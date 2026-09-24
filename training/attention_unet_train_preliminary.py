"""
Training script — 3D Attention U-Net, preliminary experiment (4 levels, 200 epochs).
GPU-ready (falls back to CPU automatically).

This is the first version of the Attention U-Net experiment, run originally on a
CPU-only server. It uses a shallower architecture (4 encoder levels) and standard
Adam optimiser. The final model used in the paper is in attention_unet_train.py.

Architecture
------------
- Attention U-Net (MONAI), 4 levels, channels=(16,32,64,128)
- Patch size 96³, 8 patches/volume, pos:neg = 3:1

Training settings
-----------------
- Loss      : DiceCELoss (λ_dice=1.0, λ_ce=1.0)
- Optimizer : Adam (lr=1e-3)
- Max epochs: 200 · Early stopping patience: 50 · Val every 5 epochs
- Seeds     : global SEED=42; per-fold seed = SEED + fold (43–47)

Outputs (written to EXPERIMENT_DIR / fold_<n>/)
-----------------------------------------------
- best_metric_model.pth
- checkpoint_latest.pth
- training_log.csv
- logs/
- all_folds_case_metrics.csv
- all_folds_metrics_summary.csv

Post-training evaluation
------------------------
- Internal-validation metrics only.
- The 16-case held-out subset is not accessed by this script.
- Held-out evaluation is performed separately using the final
  five-fold ensemble evaluation scripts.
"""

import os
from pathlib import Path
import random
import time
import csv
import sys
import traceback
from datetime import datetime
import warnings
import platform

import numpy as np
import pandas as pd
import psutil
import nibabel as nib

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandZoomd,
    RandAffined,
    Lambdad,
    AsDiscrete,
    KeepLargestConnectedComponent,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

# ============================================================
# CONFIGURATION — set DATA_DIR to your preprocessed dataset
# ============================================================
# Expected layout:
#   DATA_DIR/
#     images/   PANCREAS_XXXX.nii.gz
#     labels/   PANCREAS_XXXX.nii.gz
#     splits/   fold_1_train.txt  fold_1_val.txt  …  test.txt

DATA_DIR = Path("/path/to/preprocessed_dataset")   # <-- set this path

EXPERIMENT_NAME = "attention_unet_3d_4levels_200epochs"
EXPERIMENT_DIR  = DATA_DIR / "experiments" / EXPERIMENT_NAME

PATCH_SIZE  = (96, 96, 96)
SEED        = 42
FOLDS_TO_RUN = [1, 2, 3, 4, 5]

MAX_EPOCHS   = 200
VAL_INTERVAL = 5
PATIENCE     = 50

BATCH_SIZE    = 1
NUM_WORKERS   = 2
SW_BATCH_SIZE = 1
SW_OVERLAP    = 0.5


# ============================================================
# LOGGING
# ============================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging():
    log_dir = EXPERIMENT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = log_dir / f"attention_unet_run_{timestamp}.log"
    log_handle = open(log_file, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)

    def _hook(et, ev, tb):
        if issubclass(et, KeyboardInterrupt):
            sys.__excepthook__(et, ev, tb); return
        traceback.print_exception(et, ev, tb, file=sys.stderr)
    sys.excepthook = _hook

    print("=" * 80)
    print(f"LOG STARTED — {log_file}")
    print("=" * 80)
    return log_handle, log_file


def print_system_info(device):
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print("Hostname :", platform.node())
    print("OS       :", platform.platform())
    print("CPU cores:", psutil.cpu_count(logical=False), "physical /",
          psutil.cpu_count(logical=True), "logical")
    ram = psutil.virtual_memory()
    print(f"RAM      : {ram.total / 1024**3:.1f} GB total, "
          f"{ram.available / 1024**3:.1f} GB available")
    print("Device   :", device)
    if torch.cuda.is_available():
        print("GPU      :", torch.cuda.get_device_name(0))
        print("VRAM     :", round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    else:
        print("No GPU detected — training on CPU.")
    print("=" * 60)


# ============================================================
# DATA LOADING AND TRANSFORMS
# ============================================================
def load_split(split_file):
    with open(split_file) as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return [
        {
            "image": str(DATA_DIR / "images" / f"{cid}.nii.gz"),
            "label": str(DATA_DIR / "labels" / f"{cid}.nii.gz"),
        }
        for cid in case_ids
    ]


def get_files_for_fold(fold):
    train_files = load_split(DATA_DIR / "splits" / f"fold_{fold}_train.txt")
    val_files   = load_split(DATA_DIR / "splits" / f"fold_{fold}_val.txt")
    print(f"[FOLD {fold}] Train={len(train_files)} | Val={len(val_files)}")
    return train_files, val_files


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=lambda x: (x > 0).astype(np.uint8)),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=3,
        neg=1,
        num_samples=8,
        image_key="image",
        image_threshold=0,
    ),
    RandZoomd(keys=["image", "label"], prob=0.15,
              min_zoom=0.9, max_zoom=1.05, mode=("trilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.3),
    RandAffined(keys=["image", "label"], prob=0.2,
                rotate_range=(0.1, 0.1, 0.1), mode=("trilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=lambda x: (x > 0).astype(np.uint8)),
    EnsureTyped(keys=["image", "label"]),
])


def create_loaders(train_files, val_files, fold, check_batch=False):
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds   = Dataset(data=val_files,   transform=val_transforms)
    
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    
    if check_batch:
        batch = next(iter(train_loader))
        print(f"[FOLD {fold}] Batch check — image: {batch['image'].shape}, "
              f"label: {batch['label'].shape}, "
              f"range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

    return train_ds, val_ds, train_loader, val_loader


# ============================================================
# MODEL, LOSS, OPTIMISER
# ============================================================
def build_model(device):
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
    ).to(device)


def build_training_components(device, fold):
    model = build_model(device)
    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, include_background=False,
        lambda_dice=1.0, lambda_ce=1.0,
    )
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-3)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    print(f"[FOLD {fold}] Model: Attention U-Net 3D | channels=(16,32,64,128) | strides=(2,2,2)")
    print(f"[FOLD {fold}] Loss: DiceCELoss(λ_dice=1.0, λ_ce=1.0) | Optimizer: Adam(lr=1e-3)")
    return model, loss_fn, optimizer, dice_metric


post_pred_hard = AsDiscrete(argmax=True, to_onehot=2)
post_pred_lcc  = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, independent=False),
])
post_label = AsDiscrete(to_onehot=2)


# ============================================================
# METRICS
# ============================================================
def binary_confusion_metrics(pred_onehot, label_onehot):
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
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ============================================================
# TRAINING LOOP
# ============================================================
def train_fold(fold, device):
    print("=" * 80)
    print(f"[FOLD {fold}] START TRAINING")

    fold_seed = SEED + fold
    set_determinism(seed=fold_seed)
    random.seed(fold_seed); np.random.seed(fold_seed); torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fold_seed); torch.cuda.manual_seed_all(fold_seed)
    print(f"[FOLD {fold}] Seed: {fold_seed}")

    train_files, val_files = get_files_for_fold(fold)
    _, _, train_loader, val_loader, _ = create_loaders(
        train_files, val_files, fold,
        check_batch=(fold == FOLDS_TO_RUN[0]),
    )

    output_dir = EXPERIMENT_DIR / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, loss_fn, optimizer, dice_metric = build_training_components(device, fold)

    csv_log = output_dir / "training_log.csv"
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold", "epoch", "total_epochs", "n_iters",
            "avg_iter_time_sec", "epoch_time_sec",
            "train_loss", "val_dice_internal",
            "saved_best", "saved_checkpoint", "early_stop",
        ])

    best_metric       = -1.0
    best_metric_epoch = -1
    no_improve_epochs = 0
    early_stop        = False

    for epoch in range(MAX_EPOCHS):
        print(f"[FOLD {fold}] Epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss, step, iter_times = 0.0, 0, []
        t_epoch = time.time()

        for batch in tqdm(train_loader, desc=f"[FOLD {fold}] Train epoch {epoch+1}"):
            t_iter = time.time()
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step       += 1
            iter_times.append(time.time() - t_iter)

        epoch_loss    /= step
        avg_iter_time  = float(np.mean(iter_times))
        epoch_time     = time.time() - t_epoch

        val_dice, saved_best, saved_ckpt = "", False, False

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f"[FOLD {fold}] Internal val"):
                    inputs = data["image"].to(device, non_blocking=True)
                    labels = data["label"].to(device, non_blocking=True)
                    outputs = sliding_window_inference(
                        inputs, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP)
                    dice_metric(
                        y_pred=[post_pred_hard(i) for i in outputs],
                        y=[post_label(i) for i in labels],
                    )
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"[FOLD {fold}] Val Dice={val_dice:.4f}")

            if val_dice > best_metric:
                best_metric       = val_dice
                best_metric_epoch = epoch + 1
                saved_best        = True
                no_improve_epochs = 0
                torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                print(f"[FOLD {fold}] New best — Dice={best_metric:.4f} @ epoch {best_metric_epoch}")
            else:
                no_improve_epochs += VAL_INTERVAL
                if no_improve_epochs >= PATIENCE:
                    early_stop = True
                    print(f"[FOLD {fold}] Early stopping at epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "fold": fold, "epoch": epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric":          best_metric,
                    "best_metric_epoch":    best_metric_epoch,
                },
                output_dir / "checkpoint_latest.pth",
            )
            saved_ckpt = True

        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                fold, epoch + 1, MAX_EPOCHS, step,
                round(avg_iter_time, 4), round(epoch_time, 4),
                round(epoch_loss, 6),
                round(val_dice, 6) if isinstance(val_dice, float) else "",
                saved_best, saved_ckpt, early_stop,
            ])

        if early_stop:
            break

    print(f"[FOLD {fold}] DONE — best Dice={best_metric:.4f} @ epoch {best_metric_epoch}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"fold": fold, "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch, "output_dir": str(output_dir)}


# ============================================================
# POST-TRAINING EVALUATION (per-case, validation)
# ============================================================
def evaluate_fold(fold, device):
    train_files, val_files = get_files_for_fold(fold)
    _, _, _, val_loader = create_loaders(
        train_files, val_files, fold)

    output_dir = EXPERIMENT_DIR / f"fold_{fold}"
    model_path = output_dir / "best_metric_model.pth"
    if not model_path.exists():
        print(f"[FOLD {fold}] best_metric_model.pth not found — skipping evaluation.")
        return []

    model = build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    hd95_metric = HausdorffDistanceMetric(include_background=False,
                                           percentile=95, reduction="mean")
    rows = []

    def eval_loader(loader, files, dataset_name, postprocessor, pp_name):
        with torch.no_grad():
            for idx, data in enumerate(tqdm(loader, desc=f"[FOLD {fold}] {dataset_name} {pp_name}")):
                case_id = Path(files[idx]["image"]).stem.replace(".nii", "")
                inputs  = data["image"].to(device, non_blocking=True)
                labels  = data["label"].to(device, non_blocking=True)
                outputs = sliding_window_inference(
                    inputs, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP)
                preds_pp  = [postprocessor(i) for i in outputs]
                labels_pp = [post_label(i) for i in labels]
                m = binary_confusion_metrics(preds_pp[0], labels_pp[0])
                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=preds_pp, y=labels_pp)
                    hd95 = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                except Exception as e:
                    print(f"[FOLD {fold}] HD95 warning {case_id}: {e}")
                    hd95 = float("nan")
                rows.append({
                    "fold": fold, "dataset": dataset_name,
                    "postprocessing": pp_name, "case_id": case_id,
                    **m, "hd95": hd95,
                })
                print(f"[FOLD {fold}] {dataset_name} | {pp_name} | {case_id} | "
                      f"Dice={m['dice']:.4f} | HD95={hd95:.4f}")

    for pp_name, pp_fn in [("no_postprocessing", post_pred_hard),
                            ("largest_connected_component", post_pred_lcc)]:
        eval_loader(val_loader, val_files, "internal_val", pp_fn, pp_name)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ============================================================
# PLOTS
# ============================================================
def plot_training_curves():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for fold in FOLDS_TO_RUN:
        p = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
        plt.plot(df["epoch"], df["train_loss"], linewidth=1.5, label=f"Fold {fold}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss — 5 Folds", fontweight="bold")
    plt.grid(True, alpha=0.35); plt.legend()
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "all_folds_train_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for fold in FOLDS_TO_RUN:
        p = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["val_dice_internal"] = pd.to_numeric(df["val_dice_internal"], errors="coerce")
        dice_df = df.dropna(subset=["val_dice_internal"])
        if len(dice_df):
            plt.plot(dice_df["epoch"], dice_df["val_dice_internal"],
                     marker="o", markersize=3, linewidth=1.5, label=f"Fold {fold}")
    plt.xlabel("Epoch"); plt.ylabel("Dice (internal val)")
    plt.title("Internal Validation Dice — 5 Folds", fontweight="bold")
    plt.grid(True, alpha=0.35); plt.legend()
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "all_folds_internal_val_dice.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Training curves saved to:", EXPERIMENT_DIR)


# ============================================================
# MAIN
# ============================================================
def main():
    log_handle, log_file = setup_logging()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_system_info(device)

        set_determinism(seed=SEED)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        print("Global seed:", SEED)

        print("=" * 80)
        print("STARTING 5-FOLD TRAINING — folds:", FOLDS_TO_RUN)
        print("=" * 80)

        fold_results = []
        for fold in FOLDS_TO_RUN:
            result = train_fold(fold, device)
            fold_results.append(result)
            print(result)

        summary_df = pd.DataFrame(fold_results)
        summary_df.to_csv(EXPERIMENT_DIR / "folds_training_summary.csv", index=False)
        print(summary_df)

        plot_training_curves()

        print("=" * 80)
        print("POST-TRAINING EVALUATION (per-case metrics)")
        print("=" * 80)
        all_rows = []
        for fold in FOLDS_TO_RUN:
            all_rows.extend(evaluate_fold(fold, device))

        if all_rows:
            df = pd.DataFrame(all_rows)

            # Summary: mean ± std per fold / dataset / postprocessing
            summary = (
                df.groupby(["fold", "dataset", "postprocessing"])
                .agg(
                    n_cases=("case_id", "count"),
                    dice_mean=("dice", "mean"), dice_std=("dice", "std"),
                    jaccard_mean=("jaccard", "mean"), jaccard_std=("jaccard", "std"),
                    hd95_mean=("hd95", "mean"), hd95_std=("hd95", "std"),
                    sensitivity_mean=("sensitivity", "mean"), sensitivity_std=("sensitivity", "std"),
                    specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
                    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                )
                .reset_index()
            )
            df.to_csv(EXPERIMENT_DIR / "all_folds_case_metrics.csv", index=False)
            summary.to_csv(EXPERIMENT_DIR / "all_folds_metrics_summary.csv", index=False)
            print(summary)

        print("=" * 80)
        print("ALL FOLDS DONE — log:", log_file)
        print("=" * 80)
    finally:
        try:
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
