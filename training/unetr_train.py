"""
unetr_train.py
==============
Final UNETR training script for 3D pancreas CT segmentation.
5-fold cross-validation on the NIH Pancreas-CT dataset.

Architecture : UNETR (feature_size=24, hidden_size=512, mlp_dim=2048, num_heads=8)
Loss         : DiceCELoss (lambda_dice=1.5, lambda_ce=0.5)
Optimizer    : AdamW (lr=1e-4, weight_decay=1e-5)
Scheduler    : LinearLR warmup (25 epochs) + CosineAnnealingLR via SequentialLR
Training     : 500 epochs, AMP, gradient clipping, CacheDataset
Augmentation : RandFlip × 3, RandZoom, RandAffine, Rand3DElastic,
               RandScaleIntensity, RandShiftIntensity, RandGaussianNoise
Post-train   : Per-case metrics on the internal validation set (with / without LCC)

Outputs (per fold):
  fold_<N>/
    best_metric_model.pth
    checkpoint_latest.pth
    training_log.csv
    internal_validation_case_metrics.csv
    internal_validation_fold_summary.csv

Outputs (experiment root):
  folds_training_summary.csv
  all_folds_internal_validation_case_metrics.csv
  all_folds_internal_validation_fold_summary.csv
  all_folds_train_loss_curve.png
  all_folds_internal_val_dice_curve.png
  logs/unetr_run_<timestamp>.log
"""

from pathlib import Path
import csv
import random
import time
import warnings
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import nibabel as nib  # noqa: F401 — kept for NIfTI compatibility in the pipeline

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    Rand3DElasticd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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

PATCH_SIZE  = (96, 96, 96)
SEED        = 42
FOLDS_TO_RUN = [1, 2, 3, 4, 5]

MAX_EPOCHS          = 500
WARMUP_EPOCHS       = 25
VAL_INTERVAL        = 10
PATIENCE            = 80
CHECKPOINT_INTERVAL = 10

BATCH_SIZE   = 1
NUM_WORKERS  = 4
SW_BATCH_SIZE = 1
SW_OVERLAP   = 0.75

USE_AMP = False  # overridden in main() based on CUDA availability

# ============================================================
# LOGGING  (no sys.stdout/stderr redirection — full tracebacks
#            remain visible on the terminal)
# ============================================================
_LOG_FILE = None


def setup_logging():
    """Open a timestamped log file; return its path."""
    global _LOG_FILE
    log_dir = EXPERIMENT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"unetr_run_{timestamp}.log"
    _LOG_FILE = open(log_path, "a", buffering=1, encoding="utf-8")
    return log_path


def log(*args, **kwargs):
    """Print to console and write to the log file simultaneously."""
    print(*args, **kwargs)
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        try:
            _LOG_FILE.write(sep.join(str(a) for a in args) + end)
            _LOG_FILE.flush()
        except Exception:
            pass  # never crash training over a logging failure


def close_log():
    global _LOG_FILE
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        try:
            _LOG_FILE.close()
        except Exception:
            pass


def print_system_info(device):
    """Log hardware and environment details at startup."""
    log("=" * 60)
    log("SYSTEM INFORMATION")
    log("=" * 60)
    log("Hostname:", platform.node())
    log("OS:", platform.platform())
    log("Architecture:", platform.machine())
    log("Processor:", platform.processor())
    log("Physical CPU cores:", psutil.cpu_count(logical=False))
    log("Logical CPU cores:", psutil.cpu_count(logical=True))
    ram = psutil.virtual_memory()
    log("Total RAM (GB):", round(ram.total / 1024**3, 2))
    log("Available RAM (GB):", round(ram.available / 1024**3, 2))
    log("RAM used (%):", ram.percent)
    try:
        disk = psutil.disk_usage("/")
        log("Total disk (GB):", round(disk.total / 1024**3, 2))
        log("Free disk (GB):", round(disk.free / 1024**3, 2))
        log("Disk used (%):", disk.percent)
    except Exception as e:
        log("Could not retrieve disk info:", e)
    log("-" * 60)
    log("Selected device:", device)
    if torch.cuda.is_available():
        log("GPU:", torch.cuda.get_device_name(0))
        log("GPU count:", torch.cuda.device_count())
        log(
            "GPU total memory (GB):",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        )
    else:
        log("No GPU detected — training will use CPU.")
    log("AMP enabled:", USE_AMP)
    log("=" * 60)


# ============================================================
# DATA TRANSFORMS
# ============================================================
def binarize_label(x):
    return (x > 0).astype(np.uint8)


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=4,
        neg=1,
        num_samples=12,
        image_key="image",
        image_threshold=0,
    ),
    RandZoomd(
        keys=["image", "label"],
        prob=0.15,
        min_zoom=0.9,
        max_zoom=1.05,
        mode=("trilinear", "nearest"),
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.3),
    RandAffined(
        keys=["image", "label"],
        prob=0.2,
        rotate_range=(0.1, 0.1, 0.1),
        mode=("trilinear", "nearest"),
    ),
    Rand3DElasticd(
        keys=["image", "label"],
        prob=0.2,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        rotate_range=(0.05, 0.05, 0.05),
        scale_range=(0.05, 0.05, 0.05),
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    ),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
    RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.01),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    EnsureTyped(keys=["image", "label"]),
])


# ============================================================
# DATA LOADING
# ============================================================
def load_split(split_file):
    """Return a list of {image, label} dicts from a text file of case IDs."""
    with open(split_file, "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return [
        {
            "image": str(DATA_DIR / "images" / f"{cid}.nii.gz"),
            "label": str(DATA_DIR / "labels" / f"{cid}.nii.gz"),
        }
        for cid in case_ids
    ]


def get_files_for_fold(fold):
    """
    Load train / internal-validation file lists for one fold.

    The internal validation set is also split in half: the first half is used
    for fast monitoring during training; the full set is used for the final
    per-case evaluation after training.
    """
    train_files = load_split(DATA_DIR / "splits" / f"fold_{fold}_train.txt")
    val_files   = load_split(DATA_DIR / "splits" / f"fold_{fold}_val.txt")
    half        = max(1, len(val_files) // 2)
    val_files_half = val_files[:half]

    log("-" * 60)
    log(f"[FOLD {fold}] Data summary:")
    log(f"  Train                      : {len(train_files)}")
    log(f"  Internal val (full)        : {len(val_files)}")
    log(f"  Internal val (first half)  : {len(val_files_half)}")
    log("-" * 60)
    return train_files, val_files, val_files_half


def create_loaders_for_fold(train_files, val_files, val_files_half, fold, check_batch=False):
    """Build CacheDatasets and DataLoaders for one fold."""
    train_ds    = CacheDataset(data=train_files,    transform=train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds      = CacheDataset(data=val_files,      transform=val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds_half = CacheDataset(data=val_files_half, transform=val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)

    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    train_loader    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=torch.cuda.is_available(),
                                 persistent_workers=(NUM_WORKERS > 0))
    val_loader      = DataLoader(val_ds,      **loader_kwargs)
    val_loader_half = DataLoader(val_ds_half, **loader_kwargs)

    log("-" * 60)
    log(f"[FOLD {fold}] DataLoader summary:")
    log(f"  Train loader              : {len(train_loader)} batches")
    log(f"  Internal val (full)       : {len(val_loader)} batches")
    log(f"  Internal val (half)       : {len(val_loader_half)} batches")
    log("-" * 60)

    if check_batch:
        log("=" * 60)
        log(f"[FOLD {fold}] Checking first training batch ...")
        batch = next(iter(train_loader))
        log("  Image shape:", batch["image"].shape)
        log("  Label shape:", batch["label"].shape)
        log("  Image min/max:", batch["image"].min().item(), "/", batch["image"].max().item())
        log("  Label unique values:", torch.unique(batch["label"]).tolist())
        log(f"[FOLD {fold}] Batch check passed.")
        log("=" * 60)

    return train_ds, val_ds, val_ds_half, train_loader, val_loader, val_loader_half


# ============================================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ============================================================
def create_unetr_model(device):
    """Instantiate the UNETR model and move it to the target device."""
    model = UNETR(
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
    ).to(device)
    return model


def create_model_loss_optimizer_scheduler(device, fold):
    model = create_unetr_model(device)

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.5,
        lambda_ce=0.5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    log("=" * 60)
    log(f"[FOLD {fold}] Model / loss / optimizer / scheduler ready.")
    log("  Architecture : UNETR 3D (MONAI)")
    log("  img_size     :", PATCH_SIZE)
    log("  feature_size : 24")
    log("  hidden_size  : 512")
    log("  mlp_dim      : 2048")
    log("  num_heads    : 8")
    log("  dropout_rate : 0.2")
    log("  loss         : DiceCELoss (lambda_dice=1.5, lambda_ce=0.5)")
    log("  optimizer    : AdamW (lr=1e-4, weight_decay=1e-5)")
    log(f"  scheduler    : LinearLR warmup ({WARMUP_EPOCHS} ep, start_factor=0.05)"
        " → CosineAnnealingLR (eta_min=1e-6)")
    log("=" * 60)

    return model, loss_function, optimizer, scheduler, dice_metric


# Post-processing transforms
post_pred_hard    = AsDiscrete(argmax=True, to_onehot=2)
post_label        = AsDiscrete(to_onehot=2)
post_pred_largest = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, independent=False),
])


# ============================================================
# METRICS
# ============================================================
def compute_binary_confusion_metrics(pred_onehot, label_onehot):
    """Compute standard segmentation metrics from one-hot tensors (channel 1 = foreground)."""
    pred  = pred_onehot[1].detach().cpu().numpy().astype(bool)
    label = label_onehot[1].detach().cpu().numpy().astype(bool)
    tp = np.logical_and(pred, label).sum()
    fp = np.logical_and(pred, ~label).sum()
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# HELPERS
# ============================================================
def create_output_dir_for_fold(fold):
    output_dir = EXPERIMENT_DIR / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[FOLD {fold}] Output directory: {output_dir}")
    return output_dir


def create_training_csv(output_dir, fold):
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "epoch", "total_epochs",
            "n_iter", "avg_iter_time_sec", "epoch_time_sec",
            "train_loss",
            "val_loss_internal", "val_dice_internal",
            "learning_rate",
            "saved_best_model", "saved_checkpoint", "early_stop",
        ])
    log(f"[FOLD {fold}] Training CSV: {csv_path}")
    return csv_path


# ============================================================
# VALIDATION LOOP (fast — first-half of internal val set)
# ============================================================
def run_internal_validation(model, loader, loss_function, dice_metric, device, fold):
    model.eval()
    dice_metric.reset()
    val_losses = []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"[FOLD {fold}] Internal val (fast)"):
            inputs = data["image"].to(device, non_blocking=True)
            labels = data["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=SW_OVERLAP,
                )
                val_loss = loss_function(outputs, labels)

            val_losses.append(val_loss.item())
            preds  = [post_pred_hard(i) for i in outputs]
            labels_ = [post_label(i)   for i in labels]
            dice_metric(y_pred=preds, y=labels_)

    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    val_loss_mean = float(np.mean(val_losses)) if val_losses else float("nan")
    return val_loss_mean, val_dice


# ============================================================
# POST-TRAINING EVALUATION (full internal val set, per case)
# ============================================================
def evaluate_best_model_internal(fold, model, loader, files, loss_function, device, output_dir):
    """
    Evaluate the best checkpoint on the full internal validation set.
    Produces per-case metrics without post-processing and with LCC.
    """
    log("=" * 80)
    log(f"[FOLD {fold}] Final per-case evaluation on the internal validation set")
    log("  Variants: without post-processing | largest connected component (LCC)")
    log("=" * 80)

    rows = []
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc=f"[FOLD {fold}] Case metrics")):
            case_id = Path(files[idx]["image"]).stem.replace(".nii", "")
            inputs  = data["image"].to(device, non_blocking=True)
            labels  = data["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=SW_OVERLAP,
                )
                case_val_loss = loss_function(outputs, labels).item()

            probs = torch.softmax(outputs, dim=1)
            prob_mean = float(probs[0, 1].mean().detach().cpu().item())
            prob_max  = float(probs[0, 1].max().detach().cpu().item())

            labels_pp  = [post_label(i) for i in labels]
            label_case = labels_pp[0]

            pred_variants = {
                "no_postprocessing":         [post_pred_hard(i)    for i in outputs],
                "largest_connected_component": [post_pred_largest(i) for i in outputs],
            }

            for variant_name, outputs_pp in pred_variants.items():
                pred_case = outputs_pp[0]
                metrics   = compute_binary_confusion_metrics(pred_case, label_case)

                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=outputs_pp, y=labels_pp)
                    hd95_case = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                except Exception as e:
                    log(f"[FOLD {fold}] Warning: HD95 skipped for {case_id} / {variant_name}: {e}")
                    hd95_case = float("nan")

                rows.append({
                    "fold":          fold,
                    "dataset":       "internal_val",
                    "case_id":       case_id,
                    "postprocessing": variant_name,
                    "val_loss":      case_val_loss,
                    "dice":          metrics["dice"],
                    "jaccard":       metrics["jaccard"],
                    "hd95":          hd95_case,
                    "sensitivity":   metrics["sensitivity"],
                    "specificity":   metrics["specificity"],
                    "precision":     metrics["precision"],
                    "accuracy":      metrics["accuracy"],
                    "tp":            metrics["tp"],
                    "fp":            metrics["fp"],
                    "fn":            metrics["fn"],
                    "tn":            metrics["tn"],
                    "prob_pancreas_mean": prob_mean,
                    "prob_pancreas_max":  prob_max,
                })

                log(
                    f"[FOLD {fold}] {case_id} | {variant_name} | "
                    f"Dice: {metrics['dice']:.4f} | "
                    f"Precision: {metrics['precision']:.4f} | "
                    f"Sensitivity: {metrics['sensitivity']:.4f} | "
                    f"HD95: {hd95_case:.4f}"
                )

    cases_df  = pd.DataFrame(rows)
    cases_csv = output_dir / "internal_validation_case_metrics.csv"
    cases_df.to_csv(cases_csv, index=False)

    # Per-fold summary grouped by post-processing variant
    summary_rows = []
    for variant_name, group in cases_df.groupby("postprocessing"):
        summary_rows.append({
            "fold":            fold,
            "dataset":         "internal_val",
            "postprocessing":  variant_name,
            "n_cases":         int(group["case_id"].count()),
            "val_loss_mean":   float(group["val_loss"].mean()),
            "dice_mean":       float(group["dice"].mean()),
            "dice_std":        float(group["dice"].std()),
            "jaccard_mean":    float(group["jaccard"].mean()),
            "jaccard_std":     float(group["jaccard"].std()),
            "hd95_mean":       float(group["hd95"].mean()),
            "hd95_std":        float(group["hd95"].std()),
            "sensitivity_mean": float(group["sensitivity"].mean()),
            "sensitivity_std":  float(group["sensitivity"].std()),
            "specificity_mean": float(group["specificity"].mean()),
            "specificity_std":  float(group["specificity"].std()),
            "precision_mean":   float(group["precision"].mean()),
            "precision_std":    float(group["precision"].std()),
            "accuracy_mean":    float(group["accuracy"].mean()),
            "accuracy_std":     float(group["accuracy"].std()),
        })

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "internal_validation_fold_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    log(f"[FOLD {fold}] Per-case metrics saved: {cases_csv}")
    log(f"[FOLD {fold}] Fold summary saved     : {summary_csv}")
    return cases_df, summary_df


# ============================================================
# FOLD TRAINING
# ============================================================
def train_one_fold(fold, device, check_batch=False):
    log("=" * 80)
    log(f"[FOLD {fold}] TRAINING START")

    # Per-fold determinism
    fold_seed = SEED + fold
    set_determinism(seed=fold_seed)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
    log(f"[FOLD {fold}] Seed: {fold_seed}")

    train_files, val_files, val_files_half = get_files_for_fold(fold)
    (train_ds, val_ds, val_ds_half,
     train_loader, val_loader, val_loader_half) = create_loaders_for_fold(
        train_files=train_files,
        val_files=val_files,
        val_files_half=val_files_half,
        fold=fold,
        check_batch=check_batch,
    )

    output_dir   = create_output_dir_for_fold(fold)
    csv_log_path = create_training_csv(output_dir=output_dir, fold=fold)

    model, loss_function, optimizer, scheduler, dice_metric = \
        create_model_loss_optimizer_scheduler(device=device, fold=fold)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))

    best_metric             = -1.0
    best_metric_epoch       = -1
    epochs_without_improvement = 0
    early_stop              = False

    log(f"[FOLD {fold}] max_epochs={MAX_EPOCHS}, val_interval={VAL_INTERVAL}, patience={PATIENCE}")

    # ----------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------
    for epoch in range(MAX_EPOCHS):
        log("=" * 80)
        log(f"[FOLD {fold}] Epoch {epoch + 1}/{MAX_EPOCHS}")

        model.train()
        epoch_loss   = 0.0
        valid_steps  = 0
        iter_times   = []
        epoch_start  = time.time()

        progress_bar = tqdm(train_loader, desc=f"[FOLD {fold}] Train epoch {epoch + 1}")

        for batch_idx, batch_data in enumerate(progress_bar):
            iter_start = time.time()

            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            if batch_idx == 0:
                log(f"[FOLD {fold}] First batch — image: {inputs.shape}, label: {labels.shape}")

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and USE_AMP)):
                outputs = model(inputs)
                loss    = loss_function(outputs, labels)

            if batch_idx == 0:
                log(f"[FOLD {fold}] First forward/loss OK — loss: {loss.item():.4f}")

            if not torch.isfinite(loss):
                log(f"[FOLD {fold}] WARNING: non-finite loss ({loss.item()}), skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if batch_idx == 0:
                log(f"[FOLD {fold}] First backward/step OK")

            epoch_loss  += loss.item()
            valid_steps += 1
            iter_time    = time.time() - iter_start
            iter_times.append(iter_time)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "iter": f"{iter_time:.2f}s"})

        if valid_steps == 0:
            raise RuntimeError(f"[FOLD {fold}] No valid batches in epoch {epoch + 1}")

        epoch_loss   /= valid_steps
        avg_iter_time = float(np.mean(iter_times)) if iter_times else float("nan")
        epoch_time    = time.time() - epoch_start
        current_lr    = optimizer.param_groups[0]["lr"]

        val_loss_internal = ""
        val_dice_internal = ""
        saved_best_model  = False
        saved_checkpoint  = False

        log(f"[FOLD {fold}] Train loss: {epoch_loss:.4f}")
        log(f"[FOLD {fold}] Avg iter time: {avg_iter_time:.2f} s")
        log(f"[FOLD {fold}] Epoch time: {epoch_time:.2f} s")
        log(f"[FOLD {fold}] LR: {current_lr:.8f}")

        # ---- Validation ----
        if (epoch + 1) % VAL_INTERVAL == 0:
            log("-" * 80)
            log(f"[FOLD {fold}] Fast internal validation (first-half subset), epoch {epoch + 1}")
            val_loss_internal, val_dice_internal = run_internal_validation(
                model=model,
                loader=val_loader_half,
                loss_function=loss_function,
                dice_metric=dice_metric,
                device=device,
                fold=fold,
            )
            log(f"[FOLD {fold}] Val loss   : {val_loss_internal:.4f}")
            log(f"[FOLD {fold}] Val Dice   : {val_dice_internal:.4f}")

            if val_dice_internal > best_metric:
                best_metric      = val_dice_internal
                best_metric_epoch = epoch + 1
                saved_best_model  = True
                epochs_without_improvement = 0
                torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                log(f"[FOLD {fold}] New best model saved — Dice: {best_metric:.4f}, epoch: {best_metric_epoch}")
            else:
                epochs_without_improvement += VAL_INTERVAL
                log(f"[FOLD {fold}] No improvement ({epochs_without_improvement}/{PATIENCE} epochs)")
                if epochs_without_improvement >= PATIENCE:
                    early_stop = True
                    log(f"[FOLD {fold}] Early stopping triggered at epoch {epoch + 1}")

        scheduler.step()

        # ---- Checkpoint ----
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = output_dir / "checkpoint_latest.pth"
            torch.save(
                {
                    "fold":               fold,
                    "epoch":              epoch + 1,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":  scaler.state_dict(),
                    "best_metric":        best_metric,
                    "best_metric_epoch":  best_metric_epoch,
                    # Architecture metadata — must match create_unetr_model()
                    "feature_size":  24,
                    "hidden_size":   512,
                    "mlp_dim":       2048,
                    "num_heads":     8,
                    "patch_size":    PATCH_SIZE,
                    "warmup_epochs": WARMUP_EPOCHS,
                },
                checkpoint_path,
            )
            saved_checkpoint = True
            log(f"[FOLD {fold}] Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

        # ---- CSV row ----
        with open(csv_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold,
                epoch + 1,
                MAX_EPOCHS,
                valid_steps,
                round(avg_iter_time, 4),
                round(epoch_time, 4),
                round(epoch_loss, 6),
                round(val_loss_internal, 6) if isinstance(val_loss_internal, float) else "",
                round(val_dice_internal, 6) if isinstance(val_dice_internal, float) else "",
                current_lr,
                saved_best_model,
                saved_checkpoint,
                early_stop,
            ])

        if early_stop:
            log(f"[FOLD {fold}] Training stopped early.")
            break

    log("=" * 80)
    log(f"[FOLD {fold}] TRAINING COMPLETE")
    log(f"[FOLD {fold}] Best internal val Dice : {best_metric:.4f}")
    log(f"[FOLD {fold}] Best epoch             : {best_metric_epoch}")
    log("=" * 80)

    # ---- Post-training evaluation on the full internal val set ----
    best_model_path = output_dir / "best_metric_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        model.eval()
        evaluate_best_model_internal(
            fold=fold,
            model=model,
            loader=val_loader,
            files=val_files,
            loss_function=loss_function,
            device=device,
            output_dir=output_dir,
        )
    else:
        log(f"[FOLD {fold}] best_metric_model.pth not found — skipping final evaluation.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold":                fold,
        "best_metric":         best_metric,
        "best_metric_epoch":   best_metric_epoch,
        "output_dir":          str(output_dir),
        "csv_log_path":        str(csv_log_path),
        "internal_metrics_csv": str(output_dir / "internal_validation_case_metrics.csv"),
        "internal_summary_csv": str(output_dir / "internal_validation_fold_summary.csv"),
    }


# ============================================================
# PLOTS AND SUMMARIES
# ============================================================
def _read_training_log(fold):
    """Return the training log DataFrame for a fold, or None if missing."""
    csv_path = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
    if not csv_path.exists():
        log(f"CSV not found for fold {fold}: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    for col in ("epoch", "train_loss", "val_loss_internal", "val_dice_internal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_training_curves():
    """Generate per-fold and combined loss / Dice curves."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    green_palette = {1: "#006400", 2: "#228B22", 3: "#32CD32", 4: "#66CDAA", 5: "#98FB98"}
    red_palette   = {1: "#8B0000", 2: "#B22222", 3: "#DC143C", 4: "#FF6347", 5: "#FFA07A"}

    # Individual fold: train loss
    for fold in FOLDS_TO_RUN:
        df = _read_training_log(fold)
        if df is None:
            continue
        val_df = df.dropna(subset=["val_loss_internal"])

        plt.figure(figsize=(8, 6))
        plt.plot(df["epoch"], df["train_loss"], color="green", linewidth=2, label="Train loss")
        if not val_df.empty:
            plt.plot(val_df["epoch"], val_df["val_loss_internal"], color="red",
                     linewidth=2, linestyle="--", label="Val loss")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Loss Curves — Fold {fold}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=11)
        plt.tight_layout()
        fig_path = EXPERIMENT_DIR / f"fold_{fold}" / f"fold_{fold}_train_val_loss.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        log(f"Loss curve saved: {fig_path}")

    # Combined train loss — all folds
    plt.figure(figsize=(10, 6))
    for fold in FOLDS_TO_RUN:
        df = _read_training_log(fold)
        if df is None:
            continue
        val_df = df.dropna(subset=["val_loss_internal"])
        plt.plot(df["epoch"], df["train_loss"],
                 color=green_palette.get(fold, "green"), linewidth=2, label=f"Train fold {fold}")
        if not val_df.empty:
            plt.plot(val_df["epoch"], val_df["val_loss_internal"],
                     color=red_palette.get(fold, "red"), linewidth=2, linestyle="--",
                     label=f"Val fold {fold}")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Curves — All Folds", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fig_path = EXPERIMENT_DIR / "all_folds_train_loss_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Combined loss curve saved: {fig_path}")

    # Combined internal val Dice — all folds
    plt.figure(figsize=(10, 6))
    for fold in FOLDS_TO_RUN:
        df = _read_training_log(fold)
        if df is None or "val_dice_internal" not in df.columns:
            continue
        dice_df = df.dropna(subset=["val_dice_internal"])
        if dice_df.empty:
            continue
        plt.plot(dice_df["epoch"], dice_df["val_dice_internal"],
                 marker="o", markersize=3, linewidth=2, label=f"Fold {fold}")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice (internal val)", fontsize=12)
    plt.title("Internal Validation Dice — All Folds", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=10)
    plt.tight_layout()
    fig_path = EXPERIMENT_DIR / "all_folds_internal_val_dice_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Dice curve saved: {fig_path}")


def merge_internal_metrics():
    """Concatenate per-case and per-fold summary CSVs from all folds."""
    case_dfs, summary_dfs = [], []
    for fold in FOLDS_TO_RUN:
        fold_dir    = EXPERIMENT_DIR / f"fold_{fold}"
        case_csv    = fold_dir / "internal_validation_case_metrics.csv"
        summary_csv = fold_dir / "internal_validation_fold_summary.csv"
        if case_csv.exists():
            case_dfs.append(pd.read_csv(case_csv))
        if summary_csv.exists():
            summary_dfs.append(pd.read_csv(summary_csv))

    if case_dfs:
        all_cases = pd.concat(case_dfs, ignore_index=True)
        out = EXPERIMENT_DIR / "all_folds_internal_validation_case_metrics.csv"
        all_cases.to_csv(out, index=False)
        log(f"Merged per-case metrics saved: {out}")

    if summary_dfs:
        all_summary = pd.concat(summary_dfs, ignore_index=True)
        out = EXPERIMENT_DIR / "all_folds_internal_validation_fold_summary.csv"
        all_summary.to_csv(out, index=False)
        log(f"Merged fold summaries saved: {out}")


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

    # Global seed (each fold sets its own seed before training)
    set_determinism(seed=SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    log("Global seed:", SEED)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("STARTING AUTOMATED FOLD TRAINING")
    log("Folds:", FOLDS_TO_RUN)
    log("=" * 80)

    fold_results = []
    for fold in FOLDS_TO_RUN:
        log("\n" + "#" * 80)
        log(f"STARTING FOLD {fold}")
        log("#" * 80)
        result = train_one_fold(
            fold=fold,
            device=device,
            check_batch=(fold == FOLDS_TO_RUN[0]),
        )
        fold_results.append(result)
        log("\n" + "#" * 80)
        log(f"FOLD {fold} DONE — best Dice: {result['best_metric']:.4f}"
            f" at epoch {result['best_metric_epoch']}")
        log("#" * 80)

    fold_summary_df = pd.DataFrame(fold_results)
    summary_path    = EXPERIMENT_DIR / "folds_training_summary.csv"
    fold_summary_df.to_csv(summary_path, index=False)
    log("\nFolds training summary:")
    log(fold_summary_df.to_string(index=False))
    log(f"Saved: {summary_path}")

    plot_training_curves()
    merge_internal_metrics()

    log("=" * 80)
    log("TRAINING COMPLETE")
    log(f"Log: {log_path}")
    log("=" * 80)

    close_log()


if __name__ == "__main__":
    main()
