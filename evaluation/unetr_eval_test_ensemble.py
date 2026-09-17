"""
unetr_eval_test_ensemble.py
============================
Ensemble evaluation of a 5-fold UNETR on the NIH Pancreas-CT test set.

Inference  : 5-fold softmax probability averaging (sliding-window, SW_OVERLAP=0.75)
Post-proc  : Largest Connected Component (LCC) at each threshold
Thresholds : 0.10 – 0.90 (step 0.05)  →  17 values
Metrics    : Dice, Jaccard, HD95, Sensitivity, Specificity, Precision, Accuracy,
             TP, FP, FN, TN, US, OS, US-OS  (Mohammadi et al. 2025)

The threshold sweep is provided for sensitivity analysis only. No operating threshold is selected from the held-out evaluation set. Any deployment threshold should be prespecified or selected using validation data.

Outputs (inside ENSEMBLE_DIR):
  metrics/
    test_ensemble_threshold_metrics_by_case.csv
    test_ensemble_threshold_summary.csv
  nifti_masks/
    <case_id>_unetr_ensemble_probability.nii.gz
  figures_2d/per_patient_per_threshold/<case_id>/
    <case_id>_thr_<T>_5axial_dice_<D>.png  (5 axial slices, all thresholds)
  logs/
    unetr_eval_run_<timestamp>.log

Usage
-----
# 1. Set DATA_DIR and verify EXPERIMENT_NAME below.
# 2. Run:
python unetr_eval_test_ensemble.py
"""

from pathlib import Path
import warnings
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import nibabel as nib

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNETR
from monai.metrics import HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("/path/to/preprocessed_dataset")

EXPERIMENT_NAME = "unetr_3d_5folds"
EXPERIMENT_DIR  = DATA_DIR / "experiments" / EXPERIMENT_NAME
ENSEMBLE_DIR    = EXPERIMENT_DIR / "test_ensemble_5folds"

PATCH_SIZE     = (96, 96, 96)
FOLDS_ENSEMBLE = [1, 2, 3, 4, 5]
BATCH_SIZE     = 1
NUM_WORKERS    = 4
SW_BATCH_SIZE  = 2
SW_OVERLAP     = 0.75
USE_AMP        = False  # overridden in main() based on CUDA availability

# Threshold sweep — all thresholds evaluated; no automatic selection
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)

USE_LARGEST_COMPONENT = True
SAVE_NIFTI_PROB       = True   # save ensemble probability map per case
GENERATE_2D_FIGURES   = True

# ============================================================
# LOGGING
# ============================================================
_LOG_FILE = None


def setup_logging():
    global _LOG_FILE
    log_dir   = ENSEMBLE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"unetr_eval_run_{timestamp}.log"
    _LOG_FILE = open(log_path, "a", buffering=1, encoding="utf-8")
    return log_path


def log(*args, **kwargs):
    print(*args, **kwargs)
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        try:
            _LOG_FILE.write(sep.join(str(a) for a in args) + end)
            _LOG_FILE.flush()
        except Exception:
            pass


def close_log():
    global _LOG_FILE
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        try:
            _LOG_FILE.close()
        except Exception:
            pass


def print_system_info(device):
    log("=" * 60)
    log("SYSTEM INFORMATION")
    log("=" * 60)
    log("Hostname       :", platform.node())
    log("OS             :", platform.platform())
    log("Physical cores :", psutil.cpu_count(logical=False))
    log("Logical cores  :", psutil.cpu_count(logical=True))
    ram = psutil.virtual_memory()
    log("Total RAM (GB) :", round(ram.total / 1024**3, 2))
    log("Avail RAM (GB) :", round(ram.available / 1024**3, 2))
    log("Device         :", device)
    if torch.cuda.is_available():
        log("GPU            :", torch.cuda.get_device_name(0))
        log("GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    else:
        log("No GPU detected — inference will use CPU.")
    log("AMP enabled    :", USE_AMP)
    log("=" * 60)


# ============================================================
# DATA
# ============================================================
def binarize_label(x):
    return (x > 0).astype(np.uint8)


def load_split(split_file):
    with open(split_file, "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return [
        {
            "image": str(DATA_DIR / "images" / f"{cid}.nii.gz"),
            "label": str(DATA_DIR / "labels" / f"{cid}.nii.gz"),
        }
        for cid in case_ids
    ]


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    EnsureTyped(keys=["image", "label"]),
])


def create_test_loader(files):
    ds = CacheDataset(data=files, transform=val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )
    return ds, loader


# ============================================================
# MODEL
# ============================================================
def create_unetr_model():
    return UNETR(
        in_channels=1,
        out_channels=2,
        img_size=PATCH_SIZE,
        feature_size=24,
        hidden_size=512,
        mlp_dim=2048,
        num_heads=8,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.2,
        spatial_dims=3,
    )


def load_unetr_model_for_fold(fold):
    model_path = EXPERIMENT_DIR / f"fold_{fold}" / "best_metric_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for fold {fold}: {model_path}")
    model = create_unetr_model()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log(f"[WARNING] Fold {fold} — missing keys: {missing}")
    if unexpected:
        log(f"[WARNING] Fold {fold} — unexpected keys: {unexpected}")
    model.eval()
    log(f"UNETR fold {fold} loaded from: {model_path}")
    return model


# ============================================================
# ENSEMBLE INFERENCE
# ============================================================
def ensemble_predict(data, device, models):
    """Mean softmax probability across all fold models."""
    inputs = data["image"].to(device, non_blocking=True)
    labels = data["label"].to(device, non_blocking=True)

    prob_sum_cpu = None
    with torch.no_grad():
        for model in models:
            model.to(device)
            model.eval()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and USE_AMP)):
                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=SW_OVERLAP,
                )
                probs = torch.softmax(logits, dim=1)
            probs_cpu    = probs.detach().cpu()
            prob_sum_cpu = probs_cpu if prob_sum_cpu is None else prob_sum_cpu + probs_cpu
            del logits, probs
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prob_mean     = prob_sum_cpu / len(models)
    prob_pancreas = prob_mean[0, 1].numpy().astype(np.float32)
    label_onehot  = AsDiscrete(to_onehot=2)(labels[0].detach().cpu())
    gt_mask       = label_onehot[1].detach().cpu().numpy().astype(np.uint8)
    ct            = inputs[0, 0].detach().cpu().numpy().astype(np.float32)
    return ct, gt_mask, prob_pancreas


# ============================================================
# METRICS
# ============================================================
def compute_binary_confusion_metrics(pred_mask, label_mask):
    """
    Standard segmentation metrics + Mohammadi et al. (2025) US / OS / US-OS.

    US    = FN / (TP + FN)          — Under-Segmentation
    OS    = FP / (TP + FN)          — Over-Segmentation
    US-OS = (FP + FN) / (TP + FN)  — Combined error
    """
    pred  = pred_mask.astype(bool)
    label = label_mask.astype(bool)
    tp = np.logical_and(pred,  label).sum()
    fp = np.logical_and(pred,  ~label).sum()
    fn = np.logical_and(~pred, label).sum()
    tn = np.logical_and(~pred, ~label).sum()
    eps = 1e-8
    return {
        "dice":        (2 * tp) / (2 * tp + fp + fn + eps),
        "jaccard":     tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "precision":   tp / (tp + fp + eps),
        "accuracy":    (tp + tn) / (tp + fp + fn + tn + eps),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "us":    float(fn / (tp + fn + eps)),
        "os":    float(fp / (tp + fn + eps)),
        "us_os": float((fp + fn) / (tp + fn + eps)),
    }


def compute_hd95(pred_mask, gt_mask):
    pred_onehot = torch.from_numpy(np.stack([1 - pred_mask, pred_mask], axis=0)).float()
    gt_onehot   = torch.from_numpy(np.stack([1 - gt_mask,   gt_mask],   axis=0)).float()
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    try:
        hd95_metric(y_pred=[pred_onehot], y=[gt_onehot])
        value = hd95_metric.aggregate().item()
        hd95_metric.reset()
        return value
    except Exception as e:
        log(f"Warning: HD95 could not be computed: {e}")
        return float("nan")


def make_pred_mask(prob_pancreas, threshold):
    pred_mask = (prob_pancreas >= threshold).astype(np.uint8)
    if USE_LARGEST_COMPONENT and pred_mask.max() > 0:
        pred_onehot = torch.from_numpy(
            np.stack([1 - pred_mask, pred_mask], axis=0)
        ).float()
        pred_onehot = KeepLargestConnectedComponent(
            applied_labels=[1], is_onehot=True, independent=False
        )(pred_onehot)
        pred_mask = pred_onehot[1].detach().cpu().numpy().astype(np.uint8)
    return pred_mask


def evaluate_all_thresholds(prob_pancreas, gt_mask, case_id):
    """Compute metrics at every threshold for one case."""
    rows = []
    for thr in THRESHOLDS:
        pred_mask = make_pred_mask(prob_pancreas, thr)
        m         = compute_binary_confusion_metrics(pred_mask, gt_mask)
        hd95      = compute_hd95(pred_mask, gt_mask)
        fpr       = m["fp"] / (m["fp"] + m["tn"] + 1e-8)
        rows.append({
            "case_id":        case_id,
            "threshold":      float(thr),
            "postprocessing": "largest_connected_component" if USE_LARGEST_COMPONENT else "none",
            "dice":           m["dice"],
            "jaccard":        m["jaccard"],
            "hd95":           hd95,
            "sensitivity":    m["sensitivity"],
            "specificity":    m["specificity"],
            "precision":      m["precision"],
            "accuracy":       m["accuracy"],
            "fpr":            fpr,
            "tpr":            m["sensitivity"],
            "tp":             m["tp"],
            "fp":             m["fp"],
            "fn":             m["fn"],
            "tn":             m["tn"],
            "us":             m["us"],
            "os":             m["os"],
            "us_os":          m["us_os"],
            "prob_pancreas_mean": float(prob_pancreas.mean()),
            "prob_pancreas_max":  float(prob_pancreas.max()),
            "prob_pancreas_p95":  float(np.percentile(prob_pancreas, 95)),
        })
    return rows


# ============================================================
# 2-D VISUALISATION
# ============================================================
def _orient(img2d):
    return np.flipud(np.rot90(img2d, k=1))


def _norm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _overlay_mask(ct_slice, mask_slice, color=(1, 0, 0), alpha=0.35):
    rgb  = np.stack([_norm(ct_slice)] * 3, axis=-1)
    mask = mask_slice > 0
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 1)


def _overlay_prob(ct_slice, prob_slice, cmap_name="inferno", alpha=0.55):
    rgb       = np.stack([_norm(ct_slice)] * 3, axis=-1)
    prob      = np.clip(prob_slice.astype(np.float32), 0, 1)
    heat_rgb  = cm.get_cmap(cmap_name)(prob)[..., :3]
    alpha_map = alpha * prob[..., None]
    return np.clip((1 - alpha_map) * rgb + alpha_map * heat_rgb, 0, 1)


def _overlay_pred_and_prob(ct_slice, pred_slice, prob_slice):
    base      = _overlay_prob(ct_slice, prob_slice, cmap_name="inferno", alpha=0.45)
    pred_bool = pred_slice > 0
    blue      = np.array([0.0, 0.45, 1.0])
    base[pred_bool] = 0.45 * base[pred_bool] + 0.55 * blue
    return np.clip(base, 0, 1)


def _select_5_axial_slices(gt_mask, pred_mask):
    union   = np.logical_or(gt_mask > 0, pred_mask > 0)
    z_area  = union.sum(axis=(0, 1))
    nonzero = np.where(z_area > 0)[0]
    if len(nonzero) == 0:
        return [gt_mask.shape[2] // 2]
    if len(nonzero) <= 5:
        return nonzero.tolist()
    top5 = np.argsort(z_area)[::-1][:5]
    return sorted(top5.tolist())


def save_5_axial_slices_figure(ct, gt_mask, pred_mask, prob_mask,
                                case_id, threshold, dice_value, save_path):
    z_slices  = _select_5_axial_slices(gt_mask, pred_mask)
    fig, axes = plt.subplots(nrows=len(z_slices), ncols=5,
                             figsize=(20, 4 * len(z_slices)))
    if len(z_slices) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["CT", "CT + Ground Truth", "CT + Ensemble Probability",
                  "CT + Prediction", "Prediction + Probability"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, z in enumerate(z_slices):
        ct_s   = _orient(ct[:, :, z])
        gt_s   = _orient(gt_mask[:, :, z])
        pred_s = _orient(pred_mask[:, :, z])
        prob_s = _orient(prob_mask[:, :, z])
        axes[row, 0].imshow(_norm(ct_s), cmap="gray")
        axes[row, 0].set_ylabel(f"Axial z={z}", fontsize=10)
        axes[row, 1].imshow(_overlay_mask(ct_s, gt_s,   color=(1, 0, 0),    alpha=0.35))
        axes[row, 2].imshow(_overlay_prob(ct_s, prob_s, cmap_name="inferno", alpha=0.60))
        axes[row, 3].imshow(_overlay_mask(ct_s, pred_s, color=(0, 0.45, 1),  alpha=0.45))
        axes[row, 4].imshow(_overlay_pred_and_prob(ct_s, pred_s, prob_s))
        for col in range(5):
            axes[row, col].axis("off")

    fig.suptitle(
        f"UNETR | {case_id} | thr={threshold:.2f} | Dice={dice_value:.4f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# NIfTI SAVING
# ============================================================
def save_nifti(array, reference_path, save_path, dtype=np.float32):
    ref = nib.load(str(reference_path))
    img = nib.Nifti1Image(array.astype(dtype), affine=ref.affine, header=ref.header)
    nib.save(img, str(save_path))
    log(f"NIfTI saved: {save_path}")


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate_test_ensemble(device):
    """
    Run the full test-set ensemble evaluation across all thresholds.
    No threshold is selected automatically — inspect the summary CSV
    to choose the threshold you want to report.
    """
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir      = ENSEMBLE_DIR / "metrics"
    nifti_dir        = ENSEMBLE_DIR / "nifti_masks"
    fig2d_per_pt_dir = ENSEMBLE_DIR / "figures_2d" / "per_patient_per_threshold"
    for d in [metrics_dir, nifti_dir, fig2d_per_pt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    test_files = load_split(DATA_DIR / "splits" / "test.txt")
    log(f"Test cases: {len(test_files)}")

    _, test_loader = create_test_loader(test_files)

    log("Loading 5 UNETR fold models into CPU ...")
    models = [load_unetr_model_for_fold(fold) for fold in FOLDS_ENSEMBLE]

    threshold_rows = []

    log("=" * 80)
    log("Ensemble inference + threshold sweep for all cases")
    log("=" * 80)

    for idx, data in enumerate(tqdm(test_loader, desc="UNETR ensemble")):
        case_id = Path(test_files[idx]["image"]).stem.replace(".nii", "")
        log(f"\n{'=' * 80}")
        log(f"Processing: {case_id}")

        ct, gt_mask, prob_pancreas = ensemble_predict(data, device, models)

        # Evaluate all thresholds
        case_rows = evaluate_all_thresholds(prob_pancreas, gt_mask, case_id)
        threshold_rows.extend(case_rows)

        # Print Dice at each threshold for quick inspection
        log(f"  {'Thr':>5}  {'Dice':>6}  {'Prec':>6}  {'Sens':>6}  {'US':>6}  {'OS':>6}")
        for r in case_rows:
            log(f"  {r['threshold']:>5.2f}  {r['dice']:>6.4f}  {r['precision']:>6.4f}"
                f"  {r['sensitivity']:>6.4f}  {r['us']:>6.4f}  {r['os']:>6.4f}")

        # Save probability NIfTI
        if SAVE_NIFTI_PROB:
            save_nifti(
                prob_pancreas, test_files[idx]["image"],
                nifti_dir / f"{case_id}_unetr_ensemble_probability.nii.gz",
                np.float32,
            )

        # 2D figures at every threshold
        if GENERATE_2D_FIGURES:
            case_fig_dir = fig2d_per_pt_dir / case_id
            case_fig_dir.mkdir(parents=True, exist_ok=True)
            for r in case_rows:
                thr      = r["threshold"]
                pred_thr = make_pred_mask(prob_pancreas, thr)
                save_5_axial_slices_figure(
                    ct=ct, gt_mask=gt_mask, pred_mask=pred_thr, prob_mask=prob_pancreas,
                    case_id=case_id, threshold=thr, dice_value=r["dice"],
                    save_path=case_fig_dir / f"{case_id}_thr_{thr:.2f}_5axial_dice_{r['dice']:.4f}.png",
                )

    # ----------------------------------------------------------------
    # Save per-case × per-threshold CSV
    # ----------------------------------------------------------------
    df_by_case   = pd.DataFrame(threshold_rows)
    by_case_path = metrics_dir / "test_ensemble_threshold_metrics_by_case.csv"
    df_by_case.to_csv(by_case_path, index=False)
    log(f"\nPer-case metrics saved: {by_case_path}")

    # ----------------------------------------------------------------
    # Aggregate summary per threshold
    # ----------------------------------------------------------------
    summary = (
        df_by_case.groupby("threshold").agg(
            n_cases          =("case_id",     "count"),
            dice_mean        =("dice",        "mean"),  dice_std        =("dice",        "std"),
            jaccard_mean     =("jaccard",     "mean"),  jaccard_std     =("jaccard",     "std"),
            hd95_mean        =("hd95",        "mean"),  hd95_std        =("hd95",        "std"),
            sensitivity_mean =("sensitivity", "mean"),  sensitivity_std =("sensitivity", "std"),
            specificity_mean =("specificity", "mean"),  specificity_std =("specificity", "std"),
            precision_mean   =("precision",   "mean"),  precision_std   =("precision",   "std"),
            accuracy_mean    =("accuracy",    "mean"),  accuracy_std    =("accuracy",    "std"),
            fpr_mean         =("fpr",         "mean"),  fpr_std         =("fpr",         "std"),
            tpr_mean         =("tpr",         "mean"),  tpr_std         =("tpr",         "std"),
            us_mean          =("us",          "mean"),  us_std          =("us",          "std"),
            os_mean          =("os",          "mean"),  os_std          =("os",          "std"),
            us_os_mean       =("us_os",       "mean"),  us_os_std       =("us_os",       "std"),
        ).reset_index()
    )
    summary_path = metrics_dir / "test_ensemble_threshold_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Threshold summary saved: {summary_path}")

    # ----------------------------------------------------------------
    # Print summary table for quick threshold selection
    # ----------------------------------------------------------------
    log("\n" + "=" * 80)
    log("THRESHOLD SUMMARY  (inspect to choose the threshold to report)")
    log("=" * 80)
    cols_to_print = ["threshold", "dice_mean", "dice_std",
                     "precision_mean", "sensitivity_mean",
                     "hd95_mean", "us_mean", "os_mean"]
    log(summary[cols_to_print].to_string(index=False, float_format="{:.4f}".format))
    log("=" * 80)

    return df_by_case, summary


# ============================================================
# MAIN
# ============================================================
def main():
    global USE_AMP

    log_path = setup_logging()
    log("=" * 80)
    log("LOG STARTED")
    log(f"Log file: {log_path}")
    log("=" * 80)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = bool(device.type == "cuda")
    print_system_info(device)

    log(f"Experiment    : {EXPERIMENT_NAME}")
    log(f"Exp dir       : {EXPERIMENT_DIR}")
    log(f"Ensemble dir  : {ENSEMBLE_DIR}")

    evaluate_test_ensemble(device)

    log("=" * 80)
    log("UNETR TEST ENSEMBLE EVALUATION COMPLETE")
    log(f"Log: {log_path}")
    log("=" * 80)

    close_log()


if __name__ == "__main__":
    main()
