"""
unetr_train_preliminary.py
==========================
Preliminary / ablation training of a 3-D UNETR on the NIH Pancreas-CT dataset.
5-fold cross-validation.  Smaller architecture, fewer epochs, Adam optimizer,
no LR scheduler, no intensity normalisation.  Used to establish a performance
baseline before the final configuration.

Key differences vs. the final script (unetr_train.py)
------------------------------------------------------
Architecture : feature_size=16, hidden_size=192, mlp_dim=768, num_heads=3,
               dropout=0.1  (vs. 24/512/2048/8/0.2 in the final)
Epochs       : 200            (vs. 500)
Optimizer    : Adam lr=1e-3   (vs. AdamW lr=1e-4, wd=1e-4)
Scheduler    : none           (vs. LinearLR warmup + CosineAnnealingLR)
Dataset      : Dataset        (vs. CacheDataset)
Patches      : 8, pos:neg=3:1 (vs. 12, pos:neg=4:1)
SW overlap   : 0.5            (vs. 0.75)
Normalise    : no             (vs. NormalizeIntensityd)
Augmentation : flip+zoom+affine only (vs. + elastic + intensity)
Patience     : 60             (vs. 80)
Post-eval    : internal val + test set per fold (vs. internal val only)

Outputs  (all inside EXPERIMENT_DIR/fold_<k>/)
----------------------------------------------
best_metric_model.pth
checkpoint_latest.pth
training_log.csv
folds_training_summary.csv          (experiment root)
all_folds_case_metrics.csv          (experiment root)
all_folds_metrics_summary.csv       (experiment root)
all_folds_train_loss_curve.png      (experiment root)
all_folds_internal_val_dice_curve.png (experiment root)
logs/unetr_preliminary_run_<ts>.log

Usage
-----
# Edit DATA_DIR and EXPERIMENT_NAME below, then run:
python unetr_train_preliminary.py
"""

import os
import sys
import csv
import time
import random
import platform
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

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
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandZoomd,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

# ============================================================
# CONFIGURATION  — edit these before running
# ============================================================
DATA_DIR = Path("/path/to/preprocessed_dataset")
# Expected layout:
#   DATA_DIR/images/<case_id>.nii.gz
#   DATA_DIR/labels/<case_id>.nii.gz
#   DATA_DIR/splits/fold_<k>_train.txt
#   DATA_DIR/splits/fold_<k>_val.txt
#   DATA_DIR/splits/test.txt

EXPERIMENT_NAME = "unetr_3d_5folds_preliminary"
EXPERIMENT_DIR  = DATA_DIR / "experiments" / EXPERIMENT_NAME

PATCH_SIZE    = (96, 96, 96)
SEED          = 42
FOLDS_TO_RUN  = [1, 2, 3, 4, 5]

MAX_EPOCHS        = 200
VAL_INTERVAL      = 10
PATIENCE          = 60
CHECKPOINT_INTERVAL = 5

BATCH_SIZE    = 1
NUM_WORKERS   = 2
SW_BATCH_SIZE = 1
SW_OVERLAP    = 0.5

USE_AMP = False   # overridden in main() based on GPU availability


# ============================================================
# LOGGING
# ============================================================
class Tee:
    """Write to multiple file-like objects simultaneously."""

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
    log_file   = log_dir / f"unetr_preliminary_run_{timestamp}.log"
    log_handle = open(log_file, mode="a", buffering=1, encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        print("\n" + "=" * 80)
        print("UNHANDLED EXCEPTION")
        print("=" * 80)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("=" * 80)
        print("Error saved to:", log_file)
        print("=" * 80)

    sys.excepthook = log_uncaught_exceptions
    print("=" * 80)
    print("LOG STARTED")
    print("Log file:", log_file)
    print("=" * 80)
    return log_handle, log_file


def print_system_info(device):
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print("Hostname:", platform.node())
    print("OS:", platform.platform())
    print("Architecture:", platform.machine())
    print("Physical CPU cores:", psutil.cpu_count(logical=False))
    print("Logical CPU cores / threads:", psutil.cpu_count(logical=True))
    ram = psutil.virtual_memory()
    print("Total RAM GB:", round(ram.total / 1024 ** 3, 2))
    print("Available RAM GB:", round(ram.available / 1024 ** 3, 2))
    print("Selected device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total GPU memory GB:",
              round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
    else:
        print("No GPU detected. Training will use CPU.")
    print("AMP enabled:", USE_AMP)
    print("=" * 60)
    for sub in ("images", "labels", "splits"):
        path = DATA_DIR / sub
        print(f"  {sub}/: {'OK' if path.exists() else 'MISSING'}  ({path})")
    print("=" * 60)


# ============================================================
# DATA LOADING
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


def get_files_for_fold(fold):
    train_files    = load_split(DATA_DIR / "splits" / f"fold_{fold}_train.txt")
    val_files_int  = load_split(DATA_DIR / "splits" / f"fold_{fold}_val.txt")
    test_files     = load_split(DATA_DIR / "splits" / "test.txt")

    # Reduced internal-val subset for fast monitoring during training
    n_monitor = 5
    monitor_idx  = np.linspace(0, len(val_files_int) - 1, n_monitor, dtype=int)
    val_monitor  = [val_files_int[i] for i in monitor_idx]

    print("-" * 60)
    print(f"[FOLD {fold}] Data split:")
    print(f"  Train:                {len(train_files)}")
    print(f"  Internal val (full):  {len(val_files_int)}")
    print(f"  Internal val (monitor subset): {len(val_monitor)}  "
          f"(indices: {monitor_idx.tolist()})")
    print(f"  Test:                 {len(test_files)}")
    print("-" * 60)
    return train_files, val_files_int, val_monitor, test_files


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
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
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    EnsureTyped(keys=["image", "label"]),
])


def create_loaders_for_fold(train_files, val_files_int, val_monitor, test_files, fold,
                             check_batch=False):
    train_ds      = Dataset(data=train_files,   transform=train_transforms)
    val_int_ds    = Dataset(data=val_files_int, transform=val_transforms)
    val_monitor_ds = Dataset(data=val_monitor,  transform=val_transforms)
    test_ds       = Dataset(data=test_files,    transform=val_transforms)

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=1, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )

    train_loader      = _loader(train_ds,      shuffle=True)
    val_int_loader    = _loader(val_int_ds,    shuffle=False)
    val_monitor_loader = _loader(val_monitor_ds, shuffle=False)
    test_loader       = _loader(test_ds,       shuffle=False)

    print("-" * 60)
    print(f"[FOLD {fold}] Loaders:")
    print(f"  Train:          {len(train_loader)} batches")
    print(f"  Val internal:   {len(val_int_loader)} batches")
    print(f"  Val monitor:    {len(val_monitor_loader)} batches")
    print(f"  Test:           {len(test_loader)} batches")
    print("-" * 60)

    if check_batch:
        batch = next(iter(train_loader))
        print(f"[FOLD {fold}] First train batch — image: {batch['image'].shape}, "
              f"label: {batch['label'].shape}")
        print(f"[FOLD {fold}] Image min/max: {batch['image'].min().item():.4f} / "
              f"{batch['image'].max().item():.4f}")
        print(f"[FOLD {fold}] Label unique values: {torch.unique(batch['label']).tolist()}")

    return train_ds, val_int_ds, val_monitor_ds, test_ds, \
           train_loader, val_int_loader, val_monitor_loader, test_loader


# ============================================================
# MODEL, LOSS, OPTIMISER, METRIC
# ============================================================
def create_unetr_model(device):
    """Small UNETR for preliminary/ablation experiments."""
    return UNETR(
        in_channels=1,
        out_channels=2,
        img_size=PATCH_SIZE,
        feature_size=16,
        hidden_size=192,
        mlp_dim=768,
        num_heads=3,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
        spatial_dims=3,
    ).to(device)


def create_model_loss_optimizer_metric(device, fold):
    model = create_unetr_model(device)
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print("=" * 60)
    print(f"[FOLD {fold}] Model / loss / optimiser / metric ready.")
    print("  Model:        UNETR 3D (preliminary)")
    print("  feature_size: 16 | hidden_size: 192 | mlp_dim: 768 | num_heads: 3")
    print("  dropout:      0.1")
    print("  loss:         DiceCELoss(lambda_dice=1.0, lambda_ce=1.0)")
    print("  optimiser:    Adam lr=1e-3")
    print("=" * 60)
    return model, loss_function, optimizer, dice_metric


post_pred_hard   = AsDiscrete(argmax=True, to_onehot=2)
post_pred_lcc    = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, independent=False),
])
post_label_fn    = AsDiscrete(to_onehot=2)


# ============================================================
# METRICS
# ============================================================
def compute_binary_confusion_metrics(pred_onehot, label_onehot):
    pred  = pred_onehot[1].detach().cpu().numpy().astype(bool)
    label = label_onehot[1].detach().cpu().numpy().astype(bool)
    tp = int(np.logical_and(pred,  label).sum())
    fp = int(np.logical_and(pred,  ~label).sum())
    fn = int(np.logical_and(~pred, label).sum())
    tn = int(np.logical_and(~pred, ~label).sum())
    eps = 1e-8
    return {
        "dice":        (2 * tp) / (2 * tp + fp + fn + eps),
        "jaccard":     tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "precision":   tp / (tp + fp + eps),
        "accuracy":    (tp + tn) / (tp + fp + fn + tn + eps),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ============================================================
# TRAINING UTILITIES
# ============================================================
def create_output_dir_for_fold(fold):
    output_dir = EXPERIMENT_DIR / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_training_csv(output_dir, fold):
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "fold", "epoch", "total_epochs", "n_iter",
            "avg_iter_time_sec", "epoch_time_sec",
            "train_loss", "val_dice_monitor",
            "saved_best_model", "saved_checkpoint", "early_stop",
        ])
    return csv_path


def run_monitor_validation(model, loader, dice_metric, device, fold):
    """Quick Dice evaluation on the monitor subset."""
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for data in loader:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=SW_OVERLAP,
                )
            preds  = [post_pred_hard(i) for i in outputs]
            labels_list = [post_label_fn(i) for i in labels]
            dice_metric(y_pred=preds, y=labels_list)
    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_dice


def evaluate_loader(model, loader, files, fold, dataset_label, device):
    """Per-case evaluation with and without largest connected component."""
    print("=" * 80)
    print(f"[FOLD {fold}] Post-training evaluation — {dataset_label}")
    print("=" * 80)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    rows = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc=f"[FOLD {fold}] {dataset_label}")):
            case_id = Path(files[idx]["image"]).stem.replace(".nii", "")
            inputs  = data["image"].to(device)
            labels  = data["label"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=SW_OVERLAP,
                )
            labels_pp = [post_label_fn(i) for i in labels]
            label_case = labels_pp[0]

            for pp_name, postprocessor in [
                ("none", post_pred_hard),
                ("largest_connected_component", post_pred_lcc),
            ]:
                preds_pp   = [postprocessor(i) for i in outputs]
                pred_case  = preds_pp[0]
                metrics    = compute_binary_confusion_metrics(pred_case, label_case)
                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=preds_pp, y=labels_pp)
                    hd95_val = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                except Exception as e:
                    print(f"[FOLD {fold}] Warning HD95 not computed ({case_id} / {pp_name}): {e}")
                    hd95_val = np.nan

                rows.append({
                    "row_type":      "case_metrics",
                    "fold":          fold,
                    "dataset":       dataset_label,
                    "postprocessing": pp_name,
                    "case_id":       case_id,
                    "dice":          metrics["dice"],
                    "jaccard":       metrics["jaccard"],
                    "hd95":          hd95_val,
                    "sensitivity":   metrics["sensitivity"],
                    "specificity":   metrics["specificity"],
                    "precision":     metrics["precision"],
                    "accuracy":      metrics["accuracy"],
                    "tp": metrics["tp"], "fp": metrics["fp"],
                    "fn": metrics["fn"], "tn": metrics["tn"],
                })
                print(
                    f"[FOLD {fold}] {case_id} | {pp_name} | "
                    f"Dice={metrics['dice']:.4f} | Precision={metrics['precision']:.4f} | "
                    f"Sensitivity={metrics['sensitivity']:.4f} | HD95={hd95_val:.4f}"
                )
    return rows


# ============================================================
# TRAINING LOOP — ONE FOLD
# ============================================================
def train_one_fold(fold, device, check_batch=False):
    print("=" * 80)
    print(f"[FOLD {fold}] TRAINING START")
    fold_seed = SEED + fold
    set_determinism(seed=fold_seed)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    print(f"[FOLD {fold}] Seed: {fold_seed}")

    train_files, val_files_int, val_monitor, test_files = get_files_for_fold(fold)
    (train_ds, val_int_ds, val_monitor_ds, test_ds,
     train_loader, val_int_loader, val_monitor_loader, test_loader) = create_loaders_for_fold(
        train_files, val_files_int, val_monitor, test_files, fold, check_batch,
    )

    output_dir  = create_output_dir_for_fold(fold)
    csv_path    = create_training_csv(output_dir, fold)
    model, loss_function, optimizer, dice_metric = create_model_loss_optimizer_metric(device, fold)

    best_metric              = -1.0
    best_metric_epoch        = -1
    epochs_without_improvement = 0
    early_stop               = False

    print(f"[FOLD {fold}] max_epochs={MAX_EPOCHS} | val_interval={VAL_INTERVAL} | patience={PATIENCE}")

    for epoch in range(MAX_EPOCHS):
        print(f"[FOLD {fold}] Epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss   = 0.0
        valid_steps  = 0
        iter_times   = []
        epoch_start  = time.time()

        bar = tqdm(train_loader, desc=f"[FOLD {fold}] Training epoch {epoch + 1}")
        for batch_data in bar:
            t0     = time.time()
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_function(outputs, labels)
            if not torch.isfinite(loss):
                print(f"[FOLD {fold}] WARNING: non-finite loss, skipping batch.")
                optimizer.zero_grad()
                continue
            loss.backward()
            optimizer.step()
            epoch_loss  += loss.item()
            valid_steps += 1
            dt = time.time() - t0
            iter_times.append(dt)
            bar.set_postfix({"loss": f"{loss.item():.4f}", "t": f"{dt:.2f}s"})

        if valid_steps == 0:
            raise RuntimeError(f"[FOLD {fold}] No valid batches in epoch {epoch + 1}")

        epoch_loss    /= valid_steps
        avg_iter_time  = float(np.mean(iter_times))
        epoch_time     = time.time() - epoch_start
        val_dice_monitor = ""
        saved_best     = False
        saved_ckpt     = False

        print(f"[FOLD {fold}] Train loss: {epoch_loss:.4f} | epoch time: {epoch_time:.1f}s")

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_dice_monitor = run_monitor_validation(model, val_monitor_loader, dice_metric, device, fold)
            print(f"[FOLD {fold}] Monitor val Dice ({len(val_monitor)} cases): {val_dice_monitor:.4f}")

            if val_dice_monitor > best_metric:
                best_metric              = val_dice_monitor
                best_metric_epoch        = epoch + 1
                epochs_without_improvement = 0
                saved_best               = True
                torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                print(f"[FOLD {fold}] New best model saved — Dice={best_metric:.4f} @ epoch {best_metric_epoch}")
            else:
                epochs_without_improvement += VAL_INTERVAL
                print(f"[FOLD {fold}] No improvement ({epochs_without_improvement}/{PATIENCE})")
                if epochs_without_improvement >= PATIENCE:
                    early_stop = True
                    print(f"[FOLD {fold}] Early stopping at epoch {epoch + 1}")

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch,
                    "feature_size": 16,
                    "hidden_size": 192,
                    "mlp_dim": 768,
                    "num_heads": 3,
                    "patch_size": PATCH_SIZE,
                },
                output_dir / "checkpoint_latest.pth",
            )
            saved_ckpt = True

        with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                fold, epoch + 1, MAX_EPOCHS, valid_steps,
                round(avg_iter_time, 4), round(epoch_time, 4),
                round(epoch_loss, 6),
                round(val_dice_monitor, 6) if isinstance(val_dice_monitor, float) else "",
                saved_best, saved_ckpt, early_stop,
            ])

        if early_stop:
            break

    print("=" * 80)
    print(f"[FOLD {fold}] TRAINING DONE — best Dice={best_metric:.4f} @ epoch {best_metric_epoch}")
    print("=" * 80)

    # Post-training per-case evaluation (internal val + test set)
    all_rows = []
    best_path = output_dir / "best_metric_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        all_rows += evaluate_loader(model, val_int_loader, val_files_int, fold, "val", device)
        all_rows += evaluate_loader(model, test_loader,    test_files,    fold, "test", device)
    else:
        print(f"[FOLD {fold}] best_metric_model.pth not found — skipping evaluation.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "output_dir": str(output_dir),
        "csv_log_path": str(csv_path),
    }, all_rows


# ============================================================
# PLOTS AND SUMMARY
# ============================================================
def plot_training_curves(folds):
    plt.figure(figsize=(10, 6))
    for fold in folds:
        csv_path = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not csv_path.exists():
            print(f"CSV not found for fold {fold}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
        plt.plot(df["epoch"], df["train_loss"], linewidth=2, label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss by fold")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    out = EXPERIMENT_DIR / "all_folds_train_loss_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Training loss curve saved:", out)

    plt.figure(figsize=(10, 6))
    for fold in folds:
        csv_path = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        dice_df = df.dropna(subset=["val_dice_monitor"])
        if len(dice_df) == 0:
            continue
        plt.plot(dice_df["epoch"], dice_df["val_dice_monitor"], marker="o", linewidth=2, label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Internal val Dice (monitor subset)")
    plt.title("Internal validation Dice by fold")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    out = EXPERIMENT_DIR / "all_folds_internal_val_dice_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Val Dice curve saved:", out)


def save_metrics_csvs(all_eval_rows):
    df = pd.DataFrame(all_eval_rows)
    # Per-case CSV
    cases_csv = EXPERIMENT_DIR / "all_folds_case_metrics.csv"
    df.to_csv(cases_csv, index=False)
    print("Per-case metrics saved:", cases_csv)

    # Summary per fold × dataset × postprocessing
    summary = (
        df.groupby(["fold", "dataset", "postprocessing"]).agg(
            n_cases=("case_id", "count"),
            dice_mean=("dice", "mean"),      dice_std=("dice", "std"),
            jaccard_mean=("jaccard", "mean"), jaccard_std=("jaccard", "std"),
            hd95_mean=("hd95", "mean"),       hd95_std=("hd95", "std"),
            sensitivity_mean=("sensitivity", "mean"), sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
            precision_mean=("precision", "mean"),     precision_std=("precision", "std"),
            accuracy_mean=("accuracy", "mean"),       accuracy_std=("accuracy", "std"),
            tp_sum=("tp", "sum"), fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"), tn_sum=("tn", "sum"),
        ).reset_index()
    )
    summary_csv = EXPERIMENT_DIR / "all_folds_metrics_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Metrics summary saved:", summary_csv)
    print(summary.to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    global USE_AMP
    log_handle, log_file = setup_logging()
    try:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        USE_AMP = bool(device.type == "cuda")
        print_system_info(device)

        set_determinism(seed=SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        print("Global seed:", SEED)

        fold_results  = []
        all_eval_rows = []

        print("=" * 80)
        print("STARTING 5-FOLD TRAINING  (preliminary)")
        print("Folds:", FOLDS_TO_RUN)
        print("=" * 80)

        for fold in FOLDS_TO_RUN:
            print("\n" + "#" * 80)
            print(f"FOLD {fold}")
            print("#" * 80)
            result, eval_rows = train_one_fold(
                fold=fold,
                device=device,
                check_batch=(fold == FOLDS_TO_RUN[0]),
            )
            fold_results.append(result)
            all_eval_rows.extend(eval_rows)

        summary_df = pd.DataFrame(fold_results)
        summary_path = EXPERIMENT_DIR / "folds_training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("Fold training summary:", summary_path)
        print(summary_df.to_string(index=False))

        plot_training_curves(FOLDS_TO_RUN)
        if all_eval_rows:
            save_metrics_csvs(all_eval_rows)

        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("Log saved to:", log_file)
        print("=" * 80)
    finally:
        try:
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
