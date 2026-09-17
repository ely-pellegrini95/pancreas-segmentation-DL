# unetr_train_preliminary.py
# Entrenamiento preliminar de UNETR 3D en NIH Pancreas-CT
# 5-fold cross-validation. Sin NormalizeIntensityd.
# Arquitectura: feature_size=24, hidden_size=384, mlp_dim=1536, num_heads=6
# Optimizador: Adam lr=1e-4, sin weight_decay, sin scheduler
# 200 épocas, patience=60

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import nibabel as nib
import numpy as np
from monai.utils import set_determinism
import random
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandZoomd,
    RandAffined,
    Lambdad
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNETR
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import time
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.transforms import KeepLargestConnectedComponent
import sys
import traceback
from datetime import datetime
import warnings
import platform
import psutil

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated"
)

# ============================================================
# CONFIGURACIÓN — editar antes de ejecutar
# ============================================================
DATA_DIR = Path("/path/to/preprocessed_dataset")
# Estructura esperada:
#   DATA_DIR/images/<case_id>.nii.gz
#   DATA_DIR/labels/<case_id>.nii.gz
#   DATA_DIR/splits/fold_<k>_train.txt
#   DATA_DIR/splits/fold_<k>_val.txt
#   DATA_DIR/splits/test.txt

EXPERIMENT_NAME = "unetr_3d_5folds_preliminary"

# ============================================================
# LOG AUTOMÁTICO DE PRINTS Y ERRORES
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


# ============================================================
# SPLITS Y DATOS
# ============================================================
def load_split(split_file):
    with open(split_file, "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    files = []
    for case_id in case_ids:
        files.append({
            "image": str(DATA_DIR / "images" / f"{case_id}.nii.gz"),
            "label": str(DATA_DIR / "labels" / f"{case_id}.nii.gz"),
        })
    return files


def get_files_for_fold(fold):
    train_split = DATA_DIR / "splits" / f"fold_{fold}_train.txt"
    train_files = load_split(train_split)
    test_split  = DATA_DIR / "splits" / f"fold_{fold}_val.txt"
    test_files  = load_split(test_split)
    val_split   = DATA_DIR / "splits" / "test.txt"
    val_files   = load_split(val_split)
    print("-" * 60)
    print(f"[FOLD {fold}] Resumen de datos:")
    print(f"Train:              {len(train_files)}")
    print(f"Validación interna: {len(test_files)}")
    print(f"Validación externa: {len(val_files)}")
    print("-" * 60)
    return train_files, test_files, val_files


def binarize_label(x):
    return (x > 0).astype(np.uint8)


# ============================================================
# TRANSFORMACIONES
# ============================================================
patch_size = (96, 96, 96)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys="label", func=binarize_label),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=patch_size,
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


# ============================================================
# DATALOADERS
# ============================================================
def create_loaders_for_fold(train_files, test_files, val_files, fold, check_batch=False):
    train_ds = Dataset(data=train_files, transform=train_transforms)
    test_ds  = Dataset(data=test_files,  transform=val_transforms)
    val_ds   = Dataset(data=val_files,   transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())

    print("-" * 60)
    print(f"[FOLD {fold}] Resumen loaders:")
    print(f"Train loader:          {len(train_loader)} batches")
    print(f"Test interno loader:   {len(test_loader)} batches")
    print(f"Val externa loader:    {len(val_loader)} batches")
    print("-" * 60)

    if check_batch:
        print("=" * 60)
        print(f"[FOLD {fold}] Control de batch de entrenamiento...")
        batch = next(iter(train_loader))
        print("Image batch:", batch["image"].shape)
        print("Label batch:", batch["label"].shape)
        print("Image min/max:", batch["image"].min().item(), batch["image"].max().item())
        print("Label unique:", torch.unique(batch["label"]))
        print(f"[FOLD {fold}] Control de batch finalizado.")
        print("=" * 60)

    return train_ds, test_ds, val_ds, train_loader, test_loader, val_loader


# ============================================================
# MODELO, LOSS Y OPTIMIZADOR
# ============================================================
def create_model_loss_optimizer_metric(device, fold):
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=patch_size,
        feature_size=24,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=6,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
        spatial_dims=3,
    ).to(device)

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print("=" * 60)
    print(f"[FOLD {fold}] Modelo UNETR/loss/optimizador/métrica listos.")
    print("Modelo: UNETR 3D MONAI")
    print("img_size:", patch_size)
    print("feature_size: 24")
    print("hidden_size: 384")
    print("mlp_dim: 1536")
    print("num_heads: 6")
    print("dropout_rate: 0.1")
    print("optimizer: Adam lr=1e-4")
    print("=" * 60)

    return model, loss_function, optimizer, dice_metric


# ============================================================
# POSTPROCESAMIENTO
# ============================================================
post_pred  = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)
print("Postprocesamiento para validación interna listo:")
print("post_pred: argmax + one-hot")
print("post_label: one-hot")


# ============================================================
# CARPETA DE SALIDA Y CSV
# ============================================================
def create_output_dir_for_fold(fold):
    output_dir = DATA_DIR / "experiments" / EXPERIMENT_NAME / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"[FOLD {fold}] Carpeta de salida lista:")
    print(output_dir)
    print("=" * 60)
    return output_dir


def create_training_csv(output_dir, fold):
    csv_log_path = output_dir / "training_log.csv"
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "epoch", "total_epochs", "n_iter",
            "avg_iter_time_sec", "epoch_time_sec", "train_loss",
            "monitor_dice", "saved_best_model", "saved_checkpoint", "early_stop",
        ])
    print("=" * 60)
    print(f"[FOLD {fold}] CSV creado: {csv_log_path}")
    print("=" * 60)
    return csv_log_path


# ============================================================
# ENTRENAMIENTO POR FOLD
# ============================================================
def train_one_fold(fold, check_batch=False):
    print("=" * 80)
    print(f"[FOLD {fold}] INICIO DEL ENTRENAMIENTO")

    fold_seed = SEED + fold
    set_determinism(seed=fold_seed)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
    print(f"[FOLD {fold}] Seed usada: {fold_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FOLD {fold}] Device seleccionado: {device}")
    if torch.cuda.is_available():
        print(f"[FOLD {fold}] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[FOLD {fold}] Etapa 1/6: cargando splits...")
    train_files, test_files, val_files = get_files_for_fold(fold)

    n_monitor_cases = 5
    monitor_indices = np.linspace(0, len(test_files) - 1, n_monitor_cases, dtype=int)
    test_files_monitor = [test_files[i] for i in monitor_indices]
    print(f"[FOLD {fold}] Casos de validación interna total: {len(test_files)}")
    print(f"[FOLD {fold}] Casos usados para monitoreo: {len(test_files_monitor)}")
    print(f"[FOLD {fold}] Índices usados para monitoreo: {monitor_indices.tolist()}")

    print(f"[FOLD {fold}] Etapa 2/6: creando datasets y dataloaders...")
    train_ds, test_ds, val_ds, train_loader, test_loader, val_loader = create_loaders_for_fold(
        train_files=train_files, test_files=test_files, val_files=val_files,
        fold=fold, check_batch=check_batch,
    )

    test_monitor_ds = Dataset(data=test_files_monitor, transform=val_transforms)
    test_monitor_loader = DataLoader(
        test_monitor_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    print(f"[FOLD {fold}] Test monitor loader: {len(test_monitor_loader)} batches")

    print(f"[FOLD {fold}] Etapa 3/6: creando carpeta de salida...")
    output_dir = create_output_dir_for_fold(fold)

    print(f"[FOLD {fold}] Etapa 4/6: creando CSV de entrenamiento...")
    csv_log_path = create_training_csv(output_dir=output_dir, fold=fold)

    print(f"[FOLD {fold}] Etapa 5/6: creando modelo/loss/optimizador/métrica...")
    model, loss_function, optimizer, dice_metric = create_model_loss_optimizer_metric(
        device=device, fold=fold,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

    print(f"[FOLD {fold}] Etapa 6/6: configurando loop de entrenamiento...")
    max_epochs   = 200
    val_interval = 10
    patience     = 60
    best_metric  = -1
    best_metric_epoch = -1
    epochs_without_improvement = 0
    early_stop   = False

    print("-" * 80)
    print(f"[FOLD {fold}] Configuración de entrenamiento:")
    print(f"max_epochs: {max_epochs}")
    print(f"val_interval: {val_interval}")
    print(f"patience early stopping: {patience}")
    print(f"output_dir: {output_dir}")
    print("-" * 80)

    for epoch in range(max_epochs):
        print("=" * 80)
        print(f"[FOLD {fold}] Epoch {epoch + 1}/{max_epochs}")
        print("=" * 80)

        model.train()
        epoch_loss = 0
        step = 0
        iter_times = []
        epoch_start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"[FOLD {fold}] Entrenamiento epoch {epoch + 1}")
        for batch_data in progress_bar:
            iter_start_time = time.time()
            step += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                if not torch.isfinite(loss):
                    print(f"[FOLD {fold}] WARNING: loss no finita ({loss.item()}) en step {step}, saltando batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            iter_time = time.time() - iter_start_time
            iter_times.append(iter_time)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "iter": f"{iter_time:.2f}s"})

        epoch_loss /= step
        avg_iter_time = np.mean(iter_times)
        epoch_time = time.time() - epoch_start_time
        print(f"[FOLD {fold}] Train loss epoch {epoch + 1}: {epoch_loss:.4f}")
        print(f"[FOLD {fold}] Tiempo promedio/iteración: {avg_iter_time:.2f} s")
        print(f"[FOLD {fold}] Tiempo total epoch: {epoch_time:.2f} s")

        metric = ""
        saved_best_model  = False
        saved_checkpoint  = False

        if (epoch + 1) % val_interval == 0:
            print("-" * 80)
            print(f"[FOLD {fold}] Validación interna MONITOR epoch {epoch + 1}")
            print("-" * 80)
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for test_data in tqdm(test_monitor_loader, desc=f"[FOLD {fold}] Validación interna monitor"):
                    test_inputs = test_data["image"].to(device, non_blocking=True)
                    test_labels = test_data["label"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                        test_outputs = sliding_window_inference(
                            inputs=test_inputs,
                            roi_size=patch_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                        )
                    test_outputs = [post_pred(i) for i in test_outputs]
                    test_labels_list = [post_label(i) for i in test_labels]
                    dice_metric(y_pred=test_outputs, y=test_labels_list)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()

            print(f"[FOLD {fold}] Dice validación interna monitor ({len(test_files_monitor)} casos): {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                saved_best_model = True
                epochs_without_improvement = 0
                torch.save(model.state_dict(), output_dir / "best_metric_model.pth")
                print(f"[FOLD {fold}] Nuevo mejor modelo guardado.")
                print(f"[FOLD {fold}] Mejor Dice actual: {best_metric:.4f}")
                print(f"[FOLD {fold}] Mejor epoch actual: {best_metric_epoch}")
            else:
                epochs_without_improvement += val_interval
                print(f"[FOLD {fold}] No hubo mejora.")
                print(f"[FOLD {fold}] Epochs sin mejora: {epochs_without_improvement}/{patience}")
                if epochs_without_improvement >= patience:
                    early_stop = True
                    print(f"[FOLD {fold}] Early stopping activado en epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / "checkpoint_latest.pth"
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch,
                    "n_monitor_cases": n_monitor_cases,
                    "monitor_indices": monitor_indices.tolist(),
                },
                checkpoint_path,
            )
            saved_checkpoint = True
            print(f"[FOLD {fold}] Checkpoint guardado en epoch {epoch + 1}: {checkpoint_path}")

        with open(csv_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold, epoch + 1, max_epochs, step,
                round(avg_iter_time, 4), round(epoch_time, 4),
                round(epoch_loss, 6),
                round(metric, 6) if isinstance(metric, float) else "",
                saved_best_model, saved_checkpoint, early_stop,
            ])
        print(f"[FOLD {fold}] Log actualizado en CSV.")

        if early_stop:
            print(f"[FOLD {fold}] Entrenamiento detenido anticipadamente.")
            break

    print("=" * 80)
    print(f"[FOLD {fold}] ENTRENAMIENTO FINALIZADO")
    print(f"[FOLD {fold}] Mejor Dice validación interna: {best_metric:.4f}")
    print(f"[FOLD {fold}] Mejor epoch: {best_metric_epoch}")
    print(f"[FOLD {fold}] Modelo guardado en: {output_dir / 'best_metric_model.pth'}")
    print(f"[FOLD {fold}] CSV guardado en: {csv_log_path}")
    print("=" * 80)

    del model
    torch.cuda.empty_cache()

    return {
        "fold": fold,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "output_dir": str(output_dir),
        "csv_log_path": str(csv_log_path),
    }


# ============================================================
# POSTPROCESAMIENTO PARA EVALUACIÓN FINAL
# ============================================================
post_pred_no_pp = AsDiscrete(argmax=True, to_onehot=2)
post_pred_largest = Compose([
    AsDiscrete(argmax=True, to_onehot=2),
    KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, independent=False),
])
post_label_eval = AsDiscrete(to_onehot=2)


def compute_binary_confusion_metrics(pred_onehot, label_onehot):
    pred  = pred_onehot[1].detach().cpu().numpy().astype(bool)
    label = label_onehot[1].detach().cpu().numpy().astype(bool)
    tp = np.logical_and(pred == 1, label == 1).sum()
    fp = np.logical_and(pred == 1, label == 0).sum()
    fn = np.logical_and(pred == 0, label == 1).sum()
    tn = np.logical_and(pred == 0, label == 0).sum()
    eps = 1e-8
    dice        = (2 * tp) / (2 * tp + fp + fn + eps)
    jaccard     = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision   = tp / (tp + fp + eps)
    accuracy    = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "dice": dice, "jaccard": jaccard,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "accuracy": accuracy,
    }


def load_best_model_for_fold(fold, device):
    output_dir = create_output_dir_for_fold(fold)
    model_path = output_dir / "best_metric_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    print("=" * 60)
    print(f"[FOLD {fold}] Cargando mejor modelo: {model_path}")
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=patch_size,
        feature_size=24,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=6,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
        spatial_dims=3,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[FOLD {fold}] Modelo cargado correctamente.")
    print("=" * 60)
    return model, output_dir, model_path


def evaluate_model_on_loader(model, loader, files, fold, dataset_name,
                              postprocessor, postprocessing_name, device):
    print("-" * 60)
    print(f"[FOLD {fold}] Evaluando: {dataset_name} | {postprocessing_name}")
    print("-" * 60)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    rows = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc=f"Fold {fold} | {dataset_name} | {postprocessing_name}")):
            case_id = Path(files[idx]["image"]).stem.replace(".nii", "")
            inputs  = data["image"].to(device, non_blocking=True)
            labels  = data["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                outputs = sliding_window_inference(
                    inputs=inputs, roi_size=patch_size,
                    sw_batch_size=1, predictor=model, overlap=0.5,
                )
            outputs_pp = [postprocessor(i) for i in outputs]
            labels_pp  = [post_label_eval(i) for i in labels]
            pred_case  = outputs_pp[0]
            label_case = labels_pp[0]
            metrics = compute_binary_confusion_metrics(pred_case, label_case)
            try:
                hd95_metric.reset()
                hd95_metric(y_pred=outputs_pp, y=labels_pp)
                hd95_case = hd95_metric.aggregate().item()
                hd95_metric.reset()
            except Exception as e:
                print(f"[FOLD {fold}] Warning HD95 no calculado en {case_id}: {e}")
                hd95_case = np.nan
            row = {
                "row_type": "case_metrics", "fold": fold,
                "dataset": dataset_name, "postprocessing": postprocessing_name,
                "case_id": case_id,
                "dice": metrics["dice"], "jaccard": metrics["jaccard"],
                "hd95": hd95_case,
                "tp": metrics["tp"], "fp": metrics["fp"],
                "fn": metrics["fn"], "tn": metrics["tn"],
                "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                "precision": metrics["precision"], "accuracy": metrics["accuracy"],
                "dice_range": "", "n_cases": "", "percentage": "",
            }
            rows.append(row)
            print(
                f"[FOLD {fold}] {dataset_name} | {postprocessing_name} | {case_id} | "
                f"Dice: {metrics['dice']:.4f} | Jaccard: {metrics['jaccard']:.4f} | HD95: {hd95_case:.4f}"
            )
    return rows


# ============================================================
# BLOQUE PRINCIPAL
# ============================================================
if __name__ == '__main__':
    LOG_DIR = DATA_DIR / "experiments" / EXPERIMENT_NAME / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE  = LOG_DIR / f"unetr_preliminary_run_{timestamp}.log"

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    _log_file = open(LOG_FILE, mode="a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(_original_stdout, _log_file)
    sys.stderr = Tee(_original_stderr, _log_file)

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        print("\n" + "=" * 80)
        print("ERROR NO CONTROLADO")
        print("=" * 80)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("=" * 80)
        print("El error fue guardado en:", LOG_FILE)
        print("=" * 80)

    sys.excepthook = log_uncaught_exceptions
    print("=" * 80)
    print("LOG INICIADO")
    print("Archivo log:", LOG_FILE)
    print("=" * 80)

    print((DATA_DIR / "images").exists())
    print((DATA_DIR / "labels").exists())
    print((DATA_DIR / "splits").exists())

    print("=" * 60)
    print("CARACTERÍSTICAS DEL SERVIDOR / CPU")
    print("=" * 60)
    print("Nodo / hostname:", platform.node())
    print("Sistema operativo:", platform.platform())
    print("Arquitectura:", platform.machine())
    print("Procesador:", platform.processor())
    print("-" * 60)
    print("CPU")
    print("-" * 60)
    print("CPU cores físicos:", psutil.cpu_count(logical=False))
    print("CPU cores lógicos / threads:", psutil.cpu_count(logical=True))
    print("-" * 60)
    print("RAM")
    print("-" * 60)
    ram = psutil.virtual_memory()
    print("RAM total GB:", round(ram.total / 1024**3, 2))
    print("RAM disponible GB:", round(ram.available / 1024**3, 2))
    print("RAM usada %:", ram.percent)
    print("-" * 60)
    print("DISCO")
    print("-" * 60)
    disk = psutil.disk_usage("/")
    print("Disco total GB:", round(disk.total / 1024**3, 2))
    print("Disco libre GB:", round(disk.free / 1024**3, 2))
    print("Disco usado %:", disk.percent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo seleccionado:", device)

    if torch.cuda.is_available():
        print("GPU disponible:", torch.cuda.get_device_name(0))
        print("Número de GPUs:", torch.cuda.device_count())
        print("Memoria total GPU:",
              round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
        USE_AMP = True
    else:
        print("No se detectó GPU. El entrenamiento usará CPU.")
        USE_AMP = False

    print("AMP habilitado:", USE_AMP)

    SEED = 42
    set_determinism(seed=SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    print("Seed global:", SEED)

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------
    folds_to_run = [1, 2, 3, 4, 5]
    fold_results = []
    print("=" * 80)
    print("INICIO ENTRENAMIENTO AUTOMÁTICO DE FOLDS")
    print("Folds a entrenar:", folds_to_run)
    print("=" * 80)

    for fold in folds_to_run:
        print("\n" + "#" * 80)
        print(f"COMENZANDO FOLD {fold}")
        print("#" * 80)
        result = train_one_fold(fold=fold, check_batch=(fold == folds_to_run[0]))
        fold_results.append(result)
        print("\n" + "#" * 80)
        print(f"FOLD {fold} TERMINADO")
        print("#" * 80)
        print(result)

    print("=" * 80)
    print("ENTRENAMIENTO DE TODOS LOS FOLDS COMPLETADO")
    print("=" * 80)

    fold_summary_df = pd.DataFrame(fold_results)
    summary_path = DATA_DIR / "experiments" / EXPERIMENT_NAME / "folds_training_summary.csv"
    fold_summary_df.to_csv(summary_path, index=False)
    print(fold_summary_df)
    print("Resumen general guardado en:", summary_path)
    print("=" * 80)

    # --------------------------------------------------------
    # GRÁFICAS DE ENTRENAMIENTO
    # --------------------------------------------------------
    experiment_dir = DATA_DIR / "experiments" / EXPERIMENT_NAME

    # Loss por fold
    plt.figure(figsize=(10, 6))
    for fold in folds_to_run:
        csv_log_path = experiment_dir / f"fold_{fold}" / "training_log.csv"
        if not csv_log_path.exists():
            print(f"No existe CSV para fold {fold}: {csv_log_path}")
            continue
        log_df = pd.read_csv(csv_log_path)
        plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", linewidth=2, label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Función de pérdida durante el entrenamiento por fold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = experiment_dir / "all_folds_train_loss_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Figura loss guardada en:", fig_path)

    # Dice por fold
    plt.figure(figsize=(10, 6))
    for fold in folds_to_run:
        csv_log_path = experiment_dir / f"fold_{fold}" / "training_log.csv"
        if not csv_log_path.exists():
            continue
        log_df   = pd.read_csv(csv_log_path)
        dice_df  = log_df.dropna(subset=["monitor_dice"])
        plt.plot(dice_df["epoch"], dice_df["monitor_dice"], marker="o", linewidth=2, label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice validación interna monitor")
    plt.title("Dice de validación interna monitor durante el entrenamiento por fold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = experiment_dir / "all_folds_internal_val_dice_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Figura Dice guardada en:", fig_path)

    # --------------------------------------------------------
    # EVALUACIÓN FINAL POR FOLD
    # --------------------------------------------------------
    folds_to_evaluate = [1, 2, 3, 4, 5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device evaluación final:", device)

    all_eval_rows = []
    print("=" * 80)
    print("INICIO EVALUACIÓN FINAL DE TODOS LOS FOLDS")
    print("=" * 80)

    for fold in folds_to_evaluate:
        print("\n" + "#" * 80)
        print(f"EVALUANDO FOLD {fold}")
        print("#" * 80)
        train_files, test_files, val_files = get_files_for_fold(fold)
        train_ds, test_ds, val_ds, train_loader, test_loader, val_loader = create_loaders_for_fold(
            train_files=train_files, test_files=test_files, val_files=val_files,
            fold=fold, check_batch=False,
        )
        model, output_dir, model_path = load_best_model_for_fold(fold=fold, device=device)

        for postprocessor, pp_name in [
            (post_pred_no_pp,    "sin_postprocesamiento"),
            (post_pred_largest,  "largest_connected_component"),
        ]:
            for loader, files, ds_name in [
                (test_loader, test_files, "validacion_interna"),
                (val_loader,  val_files,  "validacion_externa"),
            ]:
                rows = evaluate_model_on_loader(
                    model=model, loader=loader, files=files, fold=fold,
                    dataset_name=ds_name, postprocessor=postprocessor,
                    postprocessing_name=pp_name, device=device,
                )
                all_eval_rows.extend(rows)

        del model
        torch.cuda.empty_cache()

    print("=" * 80)
    print("EVALUACIÓN DE TODOS LOS FOLDS COMPLETADA")
    print("=" * 80)

    # --------------------------------------------------------
    # GUARDAR MÉTRICAS
    # --------------------------------------------------------
    df_eval  = pd.DataFrame(all_eval_rows)
    df_cases = df_eval[df_eval["row_type"] == "case_metrics"].copy()

    def dice_range(dice_value):
        dice_percent = dice_value * 100
        if dice_percent < 10:   return "0-10"
        elif dice_percent < 25: return "10-25"
        elif dice_percent < 50: return "25-50"
        elif dice_percent < 75: return "50-75"
        elif dice_percent < 85: return "75-85"
        else:                   return ">85"

    range_order = ["0-10", "10-25", "25-50", "50-75", "75-85", ">85"]
    df_cases["dice_range"] = df_cases["dice"].apply(dice_range)

    freq_df = (
        df_cases
        .groupby(["fold", "dataset", "postprocessing", "dice_range"])
        .size()
        .reset_index(name="n_cases")
    )
    freq_df["percentage"] = (
        freq_df
        .groupby(["fold", "dataset", "postprocessing"])["n_cases"]
        .transform(lambda x: 100 * x / x.sum())
    )

    all_combinations = pd.MultiIndex.from_product(
        [
            sorted(df_cases["fold"].unique()),
            sorted(df_cases["dataset"].unique()),
            sorted(df_cases["postprocessing"].unique()),
            range_order,
        ],
        names=["fold", "dataset", "postprocessing", "dice_range"],
    )
    freq_df = (
        freq_df
        .set_index(["fold", "dataset", "postprocessing", "dice_range"])
        .reindex(all_combinations, fill_value=0)
        .reset_index()
    )
    freq_df["percentage"] = (
        freq_df
        .groupby(["fold", "dataset", "postprocessing"])["n_cases"]
        .transform(lambda x: 100 * x / x.sum() if x.sum() > 0 else 0)
    )

    freq_rows = []
    for _, r in freq_df.iterrows():
        freq_rows.append({
            "row_type": "dice_frequency", "fold": r["fold"],
            "dataset": r["dataset"], "postprocessing": r["postprocessing"],
            "case_id": "", "dice": "", "jaccard": "", "hd95": "",
            "tp": "", "fp": "", "fn": "", "tn": "",
            "sensitivity": "", "specificity": "", "precision": "", "accuracy": "",
            "dice_range": r["dice_range"],
            "n_cases": int(r["n_cases"]),
            "percentage": round(r["percentage"], 4),
        })
    df_freq_rows = pd.DataFrame(freq_rows)
    print(freq_df)

    df_cases_out = df_cases.copy()
    df_cases_out["n_cases"]    = ""
    df_cases_out["percentage"] = ""

    final_columns = [
        "row_type", "fold", "dataset", "postprocessing", "case_id",
        "dice", "jaccard", "hd95", "tp", "fp", "fn", "tn",
        "sensitivity", "specificity", "precision", "accuracy",
        "dice_range", "n_cases", "percentage",
    ]
    df_final_eval = pd.concat(
        [df_cases_out[final_columns], df_freq_rows[final_columns]],
        ignore_index=True,
    )
    final_eval_csv = experiment_dir / "all_folds_internal_external_metrics_and_dice_frequency.csv"
    df_final_eval.to_csv(final_eval_csv, index=False)
    print("=" * 80)
    print("CSV único guardado en:", final_eval_csv)
    print("=" * 80)
    print(df_final_eval.head())
    print(df_final_eval.tail())

    summary_df = (
        df_cases
        .groupby(["fold", "dataset", "postprocessing"])
        .agg(
            n_cases=("case_id", "count"),
            dice_mean=("dice", "mean"),         dice_std=("dice", "std"),
            jaccard_mean=("jaccard", "mean"),   jaccard_std=("jaccard", "std"),
            hd95_mean=("hd95", "mean"),         hd95_std=("hd95", "std"),
            sensitivity_mean=("sensitivity", "mean"), sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
            precision_mean=("precision", "mean"),     precision_std=("precision", "std"),
            accuracy_mean=("accuracy", "mean"),       accuracy_std=("accuracy", "std"),
            tp_sum=("tp", "sum"), fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"), tn_sum=("tn", "sum"),
        )
        .reset_index()
    )
    summary_csv = experiment_dir / "all_folds_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(summary_df)
    print("Resumen promedio por fold guardado en:", summary_csv)
    print("=" * 80)

    print("=" * 80)
    print("PROGRAMA FINALIZADO CORRECTAMENTE")
    print("Log guardado en:", LOG_FILE)
    print("=" * 80)

    try:
        _log_file.close()
    except Exception:
        pass
