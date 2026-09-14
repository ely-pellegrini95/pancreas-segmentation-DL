# pancreas-segmentation-sipaim-2026
Pancreas CT segmentation using Attention U-Net and UNETR

Work:
**"Effects of Pre-processing and Threshold Calibration on Local versus 
Global Attention Architectures for Pancreas Parenchyma Segmentation"**  
Elizabeth Pellegrini, Sebastián Ibarra, Nicole Roldán, Leonel Muñoz,
Juan-Pablo Laguna, Paola Caprile, Cecilia Besa, Steren Chabert, Rodrigo Salas  

## Overview
This repository contains the training scripts, preprocessing pipeline,
cross-validation splits, threshold analysis, and evaluation results for
a controlled comparison between **Attention U-Net** and **UNETR** for
3D pancreas CT segmentation using the NIH Pancreas-CT dataset.

## Requirements

- Python 3.10.12
- PyTorch 2.12.0
- MONAI 1.5.2
- NVIDIA GPU (experiments run on RTX 4080 16GB)

Install dependencies:
pip install -r requirements.txt

## Dataset
We used the publicly available **NIH Pancreas-CT dataset**:
- 80 contrast-enhanced abdominal CT volumes
- Download: https://www.cancerimagingarchive.net/collection/pancreas-ct/
Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU.
The dataset must be preprocessed before training (see `preprocessing/`).

## Repository Structure
├── training/
│ ├── attention_unet_train.py # Attention U-Net 5-fold training
│ └── unetr_train.py # UNETR 5-fold training
├── preprocessing/
│ └── ... # Preprocessing pipeline scripts
├── evaluation/
│ ├── threshold_analysis.py # Threshold sweep 0.10–0.90
│ └── threshold_analysis.ipynb
├── splits/
│ ├── fold_1_train.txt # Patient IDs per fold
│ ├── fold_1_val.txt
│ ├── ...
│ └── test.txt # External validation set (N=16)
└── results/
├── threshold_summary_att_unet.csv
└── threshold_summary_unetr.csv

## Reproducibility

All experiments use fixed random seeds:

| Setting | Value |
|---|---|
| Global seed | 42 |
| Fold 1 seed | 43 |
| Fold 2 seed | 44 |
| Fold 3 seed | 45 |
| Fold 4 seed | 46 |
| Fold 5 seed | 47 |

Seeds are set in `torch`, `numpy`, `random`, and `monai.set_determinism`.
Cross-validation patient splits are provided in `splits/`.

## Training
python training/attention_unet_train.py
# UNETR
python training/unetr_train.py
Set `DATA_DIR` in each script to point to your preprocessed dataset.

## Threshold Analysis
python evaluation/threshold_analysis.py

## Main Results
| Model | DSC | Jaccard | Precision |
|---|---|---|---|
| Attention U-Net | 0.732 ± 0.187 | 0.607 ± 0.216 | 0.892 ± 0.041 |
| UNETR | 0.632 ± 0.251 | 0.500 ± 0.228 | 0.903 ± 0.065 |

Validation set (N=16, held out test set), threshold = 0.50.

## Citation
If you use this code, please cite:
```bibtex
@inproceedings{pellegrini2026pancreas,
  title={Effects of Pre-processing and Threshold Calibration on Local 
         versus Global Attention Architectures for Pancreas Parenchyma Segmentation},
  author={Pellegrini, Elizabeth and others}
}
```





