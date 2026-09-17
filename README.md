# pancreas-segmentation-AttentionUnet-UNETR
> 3D pancreas CT segmentation — Attention U-Net vs UNETR, 5-fold cross-validation, NIH Pancreas-CT

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.0-orange)
![MONAI 1.5.2](https://img.shields.io/badge/MONAI-1.5.2-blueviolet)

---

## Work

**Effects of Pre-processing and Threshold Calibration on Local versus Global Attention Architectures for Pancreas Parenchyma Segmentation**

Elizabeth Pellegrini · Sebastián Ibarra · Nicole Roldán · Leonel Muñoz · Juan-Pablo Laguna · Paola Caprile · Cecilia Besa · Steren Chabert · Rodrigo Salas

---

## Overview

This repository contains training scripts, preprocessing pipeline, cross-validation splits, ensemble evaluation, and results for a comparison between **Attention U-Net** and **UNETR** for 3-D pancreas CT segmentation using the [NIH Pancreas-CT dataset](https://www.cancerimagingarchive.net/collection/pancreas-ct/).

The study examines the effect of preprocessing choices and probability threshold calibration (sweep 0.10–0.90, step 0.05) on segmentation performance, reporting Dice, Jaccard and the Mohammadi et al. (2025) under-/over-segmentation indices.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | ≥ 2.0 |
| MONAI | 1.5.2 |
| nibabel | — |
| pandas · numpy · matplotlib · tqdm · psutil | — |

```bash
pip install -r requirements.txt
```

Experiments were run on an NVIDIA GeForce RTX 4080 (16 GB VRAM).

---

## Dataset

We used the publicly available **NIH Pancreas-CT dataset** (80 contrast-enhanced abdominal CT volumes).  
Download: [cancerimagingarchive.net](https://www.cancerimagingarchive.net/collection/pancreas-ct/)

> Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016).  
> *Data From Pancreas-CT* (Version 2) [Data set]. The Cancer Imaging Archive.  
> [doi:10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)

This repository assumes the dataset has been preprocessed. See preprocessing/.

---

## Repository Structure

```
pancreas-segmentation-DL/
├── training/
│   ├── attention_unet_train_preliminary.py   # Attention U-Net — ablation (200 ep, Adam)
│   ├── attention_unet_train.py               # Attention U-Net — final (500 ep, AdamW)
│   ├── unetr_train_preliminary.py            # UNETR — ablation (200 ep, Adam)
│   └── unetr_train.py                        # UNETR — final (500 ep, AdamW)
├── evaluation/
│   ├── attention_unet_eval_test_ensemble.py  # Attention U-Net 5-fold ensemble eval
│   └── unetr_eval_test_ensemble.py           # UNETR 5-fold ensemble eval
├── preprocessing/
│   └── ...                                   # Preprocessing pipeline
├── splits/
│   ├── fold_1_train.txt                      # Patient IDs — fold 1 train (51 cases)
│   ├── fold_1_val.txt                        # Patient IDs — fold 1 internal val (13 cases)
│   ├── ...                                   # folds 2–5
│   └── test.txt                              # Held-out test set (N=16)
├── results/
│   ├── figure3_panels_zoomct.py              # Generates overlay panels
└── README.md
```

---

## Reproducibility

All experiments use fixed random seeds set in `torch`, `numpy`, `random`, and `monai.set_determinism`. Cross-validation patient splits are provided in `splits/`.

| Seed | Value |
|---|---|
| Global | 42 |
| Fold 1 | 43 |
| Fold 2 | 44 |
| Fold 3 | 45 |
| Fold 4 | 46 |
| Fold 5 | 47 |

---

## Preprocessing
preprocess_with.py: includes image modifications; preprocess_without.py: standard normalization only without image modifications.

## Training

Set `DATA_DIR` in each script to point to your preprocessed dataset, then run:

### Attention U-Net

```bash
# Preliminary (ablation — smaller model, 200 epochs)
python training/attention_unet_train_preliminary.py

# Final
python training/attention_unet_train.py
```

### UNETR

```bash
# Preliminary (ablation — smaller model, 200 epochs)
python training/unetr_train_preliminary.py

# Final
python training/unetr_train.py
```

| Script | Model | Epochs | Optimizer | Dataset |
|---|---|---|---|---|
| `*_preliminary` | smaller | 200 | Adam lr=1e-3 | Dataset |
| `*_train` | full | 500 | AdamW lr=1e-4 | CacheDataset |

UNETR uses a linear warm-up phase (25 epochs) followed by cosine annealing; Attention U-Net applies cosine annealing directly, as convolutional architectures do not require warm-up for attention stabilization.

---

## Ensemble Evaluation

Each evaluation script loads the 5 best-checkpoint models, averages their softmax probabilities, sweeps thresholds 0.10–0.90, and selects the best threshold by:  
**max Dice mean → highest Precision → lowest HD95 → highest threshold** (within tolerance 0.01).

```bash
# Attention U-Net
python evaluation/attention_unet_eval_test_ensemble.py

# UNETR
python evaluation/unetr_eval_test_ensemble.py
```

Outputs (inside `experiments/<name>/test_ensemble_5folds/`):

- `metrics/test_ensemble_threshold_metrics_by_case.csv` — per-case × per-threshold
- `metrics/test_ensemble_threshold_summary.csv` — aggregated across cases
- `metrics/test_ensemble_best_threshold.csv` — selected threshold + rationale
- `nifti_masks/` — ensemble probability map + hard mask (NIfTI)
- `figures_2d/per_patient_per_threshold/` — 5 axial slices × all thresholds

---

## Figure 3 — Qualitative panels

```bash
python results/figure3_panels.py \
    --dicom_dir   data/PANCREAS_0019_dicom \
    --gt          data/PANCREAS_0019_mask_prep.nii.gz \
    --attunet_prob data/PANCREAS_0019_ensemble_probability_pancreas.nii.gz \
    --unetr_prob   data/PANCREAS_0019_unetr_ensemble_probability.nii.gz \
    --output_dir  output_thr010 \
    --threshold   0.10
```

Repeat with `--threshold 0.50` and `--threshold 0.90` for the remaining rows or the threshold that you needed.



## Main Results

> **Note:** Results below are at a fixed threshold of 0.50.  
> Threshold-sweep behavior is reported in the manuscript; the main held-out comparison is reported at the fixed reference threshold of 0.50. Raw per-case metrics are available upon request..

| Model | DSC | Jaccard | Precision |
|---|---|---|---|
| Attention U-Net | 0.732 ± 0.187 | 0.607 ± 0.216 | 0.892 ± 0.041 |
| UNETR | 0.632 ± 0.251 | 0.500 ± 0.228 | 0.903 ± 0.065 |

*Test set, N=16. Ensemble of 5 folds. LCC post-processing applied.*

---

## Citation

If you use this code, please cite:

```bibtex
@misc{pellegrini2026pancreas,
  title     = {Effects of Pre-processing and Threshold Calibration on Local
               versus Global Attention Architectures for Pancreas Parenchyma Segmentation},
  author    = {Pellegrini, Elizabeth and Ibarra, Sebasti{\'a}n and Rold{\'a}n, Nicole
               and Mu{\~n}oz, Leonel and Laguna, Juan-Pablo and Caprile, Paola
               and Besa, Cecilia and Chabert, Steren and Salas, Rodrigo},
year = {2026},
note = {Manuscript under review},
}
```
