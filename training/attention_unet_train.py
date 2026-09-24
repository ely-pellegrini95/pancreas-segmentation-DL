"""
Training script — 3D Attention U-Net for pancreas CT segmentation.
5-fold cross-validation, GPU-ready (falls back to CPU automatically).

Architecture
------------
- Attention U-Net (MONAI), 5 levels, channels=(16,32,64,128,256)
- Instance normalisation, dropout=0.1
- Patch size 96³, 12 patches/volume, pos:neg = 4:1

Training settings
-----------------
- Loss      : DiceCELoss (λ_dice=1.5, λ_ce=0.5)
- Optimizer : AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler : CosineAnnealingLR (T_max=500, η_min=1e-6)
- Max epochs: 500 · Early stopping patience: 80 · Val every 10 epochs
- AMP       : enabled automatically when GPU is available
- Seeds     : global SEED=42; per-fold seed = SEED + fold (43–47)
- cudnn.benchmark = False  (reproducibility)

Outputs (written to EXPERIMENT_DIR / fold_<n>/)
-----------------------------------------------
- best_metric_model.pth             — best checkpoint by internal Dice
- checkpoint_latest.pth             — rolling checkpoint every 5 epochs
- training_log.csv                  — per-epoch metrics
- internal_validation_case_metrics.csv
- internal_validation_fold_summary.csv
- logs/                             — timestamped log file
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
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    Rand3DElasticd,
    Lambdad,
    AsDiscrete,
    KeepLargestConnectedComponent,
)
from monai.data import DataLoader, CacheDataset
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
#     splits/   fold_1_train.txt  fold_1_val.txt

DATA_DIR = Path("/path/to/preprocessed_dataset")   # <-- set this path

EXPERIMENT_NAME = "attention_unet_3d_5folds"
EXPERIMENT_DIR  = DATA_DIR / "experiments" / EXPERIMENT_NAME

PATCH_SIZE  = (96, 96, 96)
SEED        = 42
FOLDS_TO_RUN = [1, 2, 3, 4, 5]

MAX_EPOCHS          = 500
VAL_INTERVAL        = 10
PATIENCE            = 80
CHECKPOINT_INTERVAL = 5

BATCH_SIZE     = 1
NUM_WORKERS    = 4
SW_BATCH_SIZE  = 1
SW_OVERLAP     = 0.75

USE_AMP = False   # overridden in main() — True when GPU is detected


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
    log_handle = open(log_file, mode="a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)

    def log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        print("\n" + "=" * 80)
        print("UNCAUGHT EXCEPTION")
        print("=" * 80)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    sys.excepthook = log_uncaught

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
    print("AMP      :", USE_AMP)
    print("=" * 60)


# ============================================================
# DATA LOADING AND TRANSFORMS
# ============================================================
def binarize_label(x):
    return (x > 0).astype(np.uint8)


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
    print("-" * 60)
    print(f"[FOLD {fold}] Data split:")
    print(f"  Train             : {len(train_files)}")
    print(f"  Internal val      : {len(val_files)}")
    print("-" * 60)
    return train_files, val_files

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
    RandZoomd(keys=["image", "label"], prob=0.15,
              min_zoom=0.9, max_zoom=1.05, mode=("trilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.3),
    RandAffined(keys=["image", "label"], prob=0.2,
                rotate_range=(0.1, 0.1, 0.1), mode=("trilinear", "nearest")),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
    RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.01),
    Rand3DElasticd(
        keys=["image", "label"],
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        prob=0.2,
        rotate_range=(0.05, 0.05, 0.05),
        scale_range=(0.05, 0.05, 0.05),
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    ),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    EnsureTyped(keys=["image", "label"]),
])


def create_loaders(train_files, val_files, fold, check_batch=False):
    train_ds = CacheDataset(data=train_files, transform=train_transforms,
                            cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(data=val_files,   transform=val_transforms,
                            cache_rate=1.0, num_workers=NUM_WORKERS)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=True)

    if check_batch:
        batch = next(iter(train_loader))
        print(f"[FOLD {fold}] Batch check — image: {batch['image'].shape}, "
              f"label: {batch['label'].shape}, "
              f"range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

    return train_ds, val_ds, train_loader, val_loader


# ============================================================
# MODEL, LOSS, OPTIMISER, SCHEDULER
# ============================================================
def build_model(device):
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    ).to(device)


def build_training_components(device, fold):
    model = build_model(device)

    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, include_background=False,
        lambda_dice=1.5, lambda_ce=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6,
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print(f"[FOLD {fold}] Model: Attention U-Net 3D | channels=(16,32,64,128,256) | "
          f"strides=(2,2,2,2) | dropout=0.1")
    print(f"[FOLD {fold}] Loss: DiceCELoss(λ_dice=1.5, λ_ce=0.5) | "
          f"Optimizer: AdamW(lr=1e-4, wd=1e-5) | "
          f"Scheduler: CosineAnnealingLR(T_max={MAX_EPOCHS}, η_min=1e-6)")
    return model, loss_fn, optimizer, scheduler, dice_metric


post_pred_hard   = AsDiscrete(argmax=True, to_onehot=2)
post_label       = AsDiscrete(to_onehot=2)
post_pred_lcc    = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, independent=False),
])


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
# INTERNAL VALIDATION
# ============================================================
def run_internal_val(model, loader, loss_fn, dice_metric, device, fold):
    model.eval()
    dice_metric.reset()
    losses = []
    with torch.no_grad():
        for data in tqdm(loader, desc=f"[FOLD {fold}] Internal val"):
            inputs = data["image"].to(device, non_blocking=True)
            labels = data["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP)
                losses.append(loss_fn(outputs, labels).item())
            dice_metric(
                y_pred=[post_pred_hard(i) for i in outputs],
                y=[post_label(i) for i in labels],
            )
    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return float(np.mean(losses)) if losses else float("nan"), val_dice


def evaluate_best_model(fold, model, loader, files, loss_fn, device, output_dir):
    """Per-case metrics on the internal validation set, with and without LCC."""
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95,
                                          reduction="mean")
    model.eval()
    rows = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc=f"[FOLD {fold}] Per-case metrics")):
            case_id = Path(files[idx]["image"]).stem.replace(".nii", "")
            inputs  = data["image"].to(device, non_blocking=True)
            labels  = data["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs   = sliding_window_inference(
                    inputs, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP)
                case_loss = loss_fn(outputs, labels).item()

            probs        = torch.softmax(outputs, dim=1)
            labels_pp    = [post_label(i) for i in labels]
            label_case   = labels_pp[0]

            versions = {
                "no_postprocessing":          [post_pred_hard(i) for i in outputs],
                "largest_connected_component":[post_pred_lcc(i)  for i in outputs],
            }
            for post_name, preds_pp in versions.items():
                pred_case = preds_pp[0]
                m = binary_confusion_metrics(pred_case, label_case)
                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=preds_pp, y=labels_pp)
                    hd95 = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                except Exception as e:
                    print(f"[FOLD {fold}] HD95 warning {case_id}/{post_name}: {e}")
                    hd95 = float("nan")
                rows.append({
                    "fold": fold, "dataset": "internal_val", "case_id": case_id,
                    "postprocessing": post_name, "val_loss": case_loss,
                    "dice": m["dice"], "jaccard": m["jaccard"], "hd95": hd95,
                    "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                    "precision": m["precision"], "accuracy": m["accuracy"],
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                    "prob_pancreas_mean": float(probs[0, 1].mean().item()),
                    "prob_pancreas_max":  float(probs[0, 1].max().item()),
                })
                print(f"[FOLD {fold}] {case_id} | {post_name} | "
                      f"Dice={m['dice']:.4f} | Prec={m['precision']:.4f} | "
                      f"Sens={m['sensitivity']:.4f} | HD95={hd95:.4f}")

    cases_df  = pd.DataFrame(rows)
    cases_df.to_csv(output_dir / "internal_validation_case_metrics.csv", index=False)

    summary_rows = []
    for post_name, g in cases_df.groupby("postprocessing"):
        summary_rows.append({
            "fold": fold, "dataset": "internal_val", "postprocessing": post_name,
            "n_cases":          int(g["case_id"].count()),
            "val_loss_mean":    float(g["val_loss"].mean()),
            "dice_mean":        float(g["dice"].mean()),
            "dice_std":         float(g["dice"].std()),
            "jaccard_mean":     float(g["jaccard"].mean()),
            "jaccard_std":      float(g["jaccard"].std()),
            "hd95_mean":        float(g["hd95"].mean()),
            "hd95_std":         float(g["hd95"].std()),
            "sensitivity_mean": float(g["sensitivity"].mean()),
            "sensitivity_std":  float(g["sensitivity"].std()),
            "specificity_mean": float(g["specificity"].mean()),
            "specificity_std":  float(g["specificity"].std()),
            "precision_mean":   float(g["precision"].mean()),
            "precision_std":    float(g["precision"].std()),
            "accuracy_mean":    float(g["accuracy"].mean()),
            "accuracy_std":     float(g["accuracy"].std()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "internal_validation_fold_summary.csv", index=False)
    return cases_df, summary_df


# ============================================================
# TRAINING LOOP
# ============================================================
def train_fold(fold, device):
    print("=" * 80)
    print(f"[FOLD {fold}] START TRAINING")

    fold_seed = SEED + fold
    set_determinism(seed=fold_seed)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
    print(f"[FOLD {fold}] Seed: {fold_seed}")

    train_files, val_files = get_files_for_fold(fold)
    _, _, train_loader, val_loader, _ = create_loaders(
        train_files, val_files, fold,
        check_batch=(fold == FOLDS_TO_RUN[0]),
    )

    output_dir = EXPERIMENT_DIR / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, loss_fn, optimizer, scheduler, dice_metric = build_training_components(
        device, fold)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

    csv_log = output_dir / "training_log.csv"
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold", "epoch", "total_epochs", "n_iters",
            "avg_iter_time_sec", "epoch_time_sec",
            "train_loss", "val_loss_internal", "val_dice_internal",
            "learning_rate", "saved_best", "saved_checkpoint", "early_stop",
        ])

    best_metric        = -1.0
    best_metric_epoch  = -1
    no_improve_epochs  = 0
    early_stop         = False

    for epoch in range(MAX_EPOCHS):
        print(f"[FOLD {fold}] Epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss, valid_steps, iter_times = 0.0, 0, []
        t_epoch = time.time()

        for batch in tqdm(train_loader, desc=f"[FOLD {fold}] Train epoch {epoch+1}"):
            t_iter = time.time()
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = model(inputs)
                loss    = loss_fn(outputs, labels)

            if not torch.isfinite(loss):
                print(f"[FOLD {fold}] WARNING: non-finite loss, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            valid_steps += 1
            iter_times.append(time.time() - t_iter)

        if valid_steps == 0:
            raise RuntimeError(f"[FOLD {fold}] No valid batches in epoch {epoch+1}")

        epoch_loss    /= valid_steps
        avg_iter_time  = float(np.mean(iter_times))
        epoch_time     = time.time() - t_epoch
        current_lr     = optimizer.param_groups[0]["lr"]

        val_loss_int, val_dice_int = "", ""
        saved_best, saved_ckpt     = False, False

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss_int, val_dice_int = run_internal_val(
                model, val_loader, loss_fn, dice_metric, device, fold)
            print(f"[FOLD {fold}] Val loss={val_loss_int:.4f}  Val Dice={val_dice_int:.4f}")

            if val_dice_int > best_metric:
                best_metric       = val_dice_int
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

        scheduler.step()

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "fold": fold, "epoch": epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "best_metric":          best_metric,
                    "best_metric_epoch":    best_metric_epoch,
                    "channels":             (16, 32, 64, 128, 256),
                    "patch_size":           PATCH_SIZE,
                },
                output_dir / "checkpoint_latest.pth",
            )
            saved_ckpt = True

        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                fold, epoch + 1, MAX_EPOCHS, valid_steps,
                round(avg_iter_time, 4), round(epoch_time, 4),
                round(epoch_loss, 6),
                round(val_loss_int, 6) if isinstance(val_loss_int, float) else "",
                round(val_dice_int, 6) if isinstance(val_dice_int, float) else "",
                current_lr, saved_best, saved_ckpt, early_stop,
            ])

        if early_stop:
            break

    print(f"[FOLD {fold}] TRAINING DONE — best Dice={best_metric:.4f} @ epoch {best_metric_epoch}")

    best_ckpt = output_dir / "best_metric_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        evaluate_best_model(fold, model, val_loader, val_files, loss_fn, device, output_dir)
    else:
        print(f"[FOLD {fold}] best_metric_model.pth not found — skipping per-case eval.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold, "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "output_dir": str(output_dir),
    }


# ============================================================
# PLOTS
# ============================================================
def plot_training_curves():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    colors_train = {1:"#006400",2:"#228B22",3:"#32CD32",4:"#66CDAA",5:"#98FB98"}
    colors_val   = {1:"#8B0000",2:"#B22222",3:"#DC143C",4:"#FF6347",5:"#FFA07A"}

    plt.figure(figsize=(10, 6))
    for fold in FOLDS_TO_RUN:
        csv_path = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["epoch"]            = pd.to_numeric(df["epoch"],            errors="coerce")
        df["train_loss"]       = pd.to_numeric(df["train_loss"],       errors="coerce")
        df["val_loss_internal"]= pd.to_numeric(df["val_loss_internal"],errors="coerce")
        df["val_dice_internal"]= pd.to_numeric(df["val_dice_internal"],errors="coerce")

        val_df = df.dropna(subset=["val_loss_internal"])
        plt.plot(df["epoch"], df["train_loss"],
                 color=colors_train.get(fold, "green"), linewidth=1.5,
                 linestyle="-", label=f"Train fold {fold}")
        if len(val_df) > 0:
            plt.plot(val_df["epoch"], val_df["val_loss_internal"],
                     color=colors_val.get(fold, "red"), linewidth=1.5,
                     linestyle="--", label=f"Val fold {fold}")

    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves — 5 Folds", fontweight="bold")
    plt.grid(True, alpha=0.35); plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "all_folds_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for fold in FOLDS_TO_RUN:
        csv_path = EXPERIMENT_DIR / f"fold_{fold}" / "training_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
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


def merge_internal_metrics():
    case_dfs, summary_dfs = [], []
    for fold in FOLDS_TO_RUN:
        fd = EXPERIMENT_DIR / f"fold_{fold}"
        for lst, name in [(case_dfs, "internal_validation_case_metrics.csv"),
                          (summary_dfs, "internal_validation_fold_summary.csv")]:
            p = fd / name
            if p.exists():
                lst.append(pd.read_csv(p))

    if case_dfs:
        pd.concat(case_dfs, ignore_index=True).to_csv(
            EXPERIMENT_DIR / "all_folds_internal_case_metrics.csv", index=False)
    if summary_dfs:
        pd.concat(summary_dfs, ignore_index=True).to_csv(
            EXPERIMENT_DIR / "all_folds_internal_fold_summary.csv", index=False)


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

        # Global seed (per-fold seeds are set inside train_fold)
        set_determinism(seed=SEED)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False   # reproducibility
        print("Global seed:", SEED)
        print("cudnn.benchmark:", torch.backends.cudnn.benchmark)

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
        merge_internal_metrics()

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
