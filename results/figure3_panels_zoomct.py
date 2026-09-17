"""
figure3_panels.py — TP/FP/FN overlay panels for Figure 3
=========================================================
Generates the qualitative segmentation panels used in Figure 3 of the paper:

    "3D Pancreas Segmentation in CT: Attention U-Net vs. UNETR — A
     Comparative Study on the NIH Pancreas-CT Dataset"

Given a patient's CT volume (DICOM) and three NIfTI files (ground truth mask,
Att U-Net probability map, UNETR probability map), this script:

  1. Reconstructs the preprocessed CT volume (body mask → crop → pad-to-div8),
     matching the coordinate space in which the model outputs live.
  2. Selects the axial slice with the largest pancreas cross-section.
  3. Binarises both probability maps at the chosen threshold.
  4. Renders TP/FP/FN overlays (green / blue / red) + GT contour (pink).
  5. Saves full-FOV panels, zoomed-in panels, and a 2×4 preview grid.

To reproduce Figure 3 from the paper, run the script three times with
--threshold 0.10, 0.50, and 0.90.

REQUIREMENTS
------------
Python ≥ 3.8
  pip install numpy nibabel pydicom matplotlib scipy

INPUTS
------
  --dicom_dir     Folder containing the patient's original DICOM slices.
                  Example: data/PANCREAS_0019_dicom/
  --gt            NIfTI binary mask in preprocessed space (.nii.gz).
                  Example: data/PANCREAS_0019_mask_prep.nii.gz
  --attunet_prob  NIfTI probability map from Att U-Net ensemble (.nii.gz).
                  Example: data/PANCREAS_0019_ensemble_probability_pancreas.nii.gz
  --unetr_prob    NIfTI probability map from UNETR ensemble (.nii.gz).
                  Example: data/PANCREAS_0019_unetr_ensemble_probability.nii.gz
  --output_dir    Folder where output PNG files will be saved.
                  Created automatically if it does not exist.
                  Default: output_panels/
  --threshold     Binarisation threshold applied to both probability maps.
                  Range: 0.0–1.0.  Default: 0.10
                  Use 0.10 / 0.50 / 0.90 to reproduce the three rows in Fig. 3.

OUTPUTS (in --output_dir)
--------------------------
  ct.png               Plain CT axial slice (soft-tissue window)
  gt_contour.png       CT + GT contour (pink)
  attunet_overlay.png  CT + TP/FP/FN overlay for Att U-Net
  unetr_overlay.png    CT + TP/FP/FN overlay for UNETR
  ct_zoom.png          Zoomed-in versions of the above (×4 panels)
  gt_zoom.png
  attunet_zoom.png
  unetr_zoom.png
  _preview.png         2×4 review grid at 150 DPI

COLOR LEGEND
------------
  🟢 TP  (green)  — correctly predicted pancreas voxel
  🔵 FP  (blue)   — over-segmentation (predicted, not in GT)
  🔴 FN  (red)    — under-segmentation (missed by model)
  🩷 GT  (pink)   — ground-truth contour

EXAMPLE USAGE
-------------
  # Threshold 0.10 (low, high recall — row 1 of Figure 3)
  python figure3_panels.py \\
      --dicom_dir  data/PANCREAS_0019_dicom \\
      --gt         data/PANCREAS_0019_mask_prep.nii.gz \\
      --attunet_prob data/PANCREAS_0019_ensemble_probability_pancreas.nii.gz \\
      --unetr_prob   data/PANCREAS_0019_unetr_ensemble_probability.nii.gz \\
      --output_dir output_thr010 \\
      --threshold 0.10

  # Threshold 0.50 (balanced — row 2 of Figure 3)
  python figure3_panels.py  ...same inputs...  --output_dir output_thr050 --threshold 0.50

  # Threshold 0.90 (high precision — row 3 of Figure 3)
  python figure3_panels.py  ...same inputs...  --output_dir output_thr090 --threshold 0.90
"""

import argparse
import re
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.ndimage import (binary_fill_holes, binary_opening,
                           binary_closing, label as ndlabel,
                           binary_dilation)

# ============================================================
# Visualisation parameters  (edit here to fine-tune appearance)
# ============================================================

# HU windowing for display — soft-tissue abdomen
# W=250, C=50  →  window range [-75, 175] HU
HU_WINDOW_CENTER = 50
HU_WINDOW_WIDTH  = 250

# Overlay
OVERLAY_ALPHA  = 0.50   # transparency of TP/FP/FN colour overlay (0=invisible, 1=opaque)
ZOOM_MARGIN    = 20     # extra pixels around the GT bounding box in zoomed panels
CONTOUR_WIDTH  = 2      # GT contour thickness in pixels
DPI            = 300    # output resolution

COLOR_TP = np.array([0.18, 0.80, 0.35])   # green  — TP
COLOR_FP = np.array([0.25, 0.50, 0.95])   # blue   — FP (over-segmentation)
COLOR_FN = np.array([0.95, 0.20, 0.20])   # red    — FN (under-segmentation)
COLOR_GT = np.array([1.00, 0.40, 0.75])   # pink   — GT contour

# Preprocessing constants — must match the training pipeline
BODY_THRESHOLD_HU   = -500
BODY_FILL_HU        = -1000.0
BODY_MASK_MORPH_RAD = 3
CROP_MARGIN         = (4, 20, 20)   # (z, y, x) voxels
DIVISIBLE_BY        = 8


# ============================================================
# Preprocessing functions  (mirror of the training notebook)
# ============================================================

def natural_key(text):
    """Sort strings containing numbers in natural order."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(text))]


def read_dicom_volume_hu(case_dir):
    """Read DICOM slices, sort by spatial position, return HU volume (Z, Y, X)."""
    files = sorted([p for p in Path(case_dir).rglob("*") if p.is_file()],
                   key=natural_key)
    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise RuntimeError(f"No DICOM files with pixel data found in: {case_dir}")

    # Sort along the slice-normal direction
    first = slices[0]
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None:
        iop = np.array(iop, dtype=np.float32)
        normal = np.cross(iop[:3], iop[3:])
    else:
        normal = np.array([0., 0., 1.], dtype=np.float32)

    def position(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            return float(getattr(ds, "InstanceNumber", 0))
        return float(np.dot(np.array(ipp, dtype=np.float32), normal))

    slices.sort(key=position)

    def to_hu(ds):
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    volume = np.stack([to_hu(ds) for ds in slices], axis=0)   # (Z, Y, X)
    print(f"  DICOM loaded: {volume.shape}  HU [{volume.min():.0f}, {volume.max():.0f}]")
    return volume.astype(np.float32)


def largest_cc_3d(mask):
    """Keep only the largest connected component of a binary 3-D mask."""
    lbl, n = ndlabel(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return (lbl == np.argmax(counts)).astype(np.uint8)


def create_body_mask(vol_hu, threshold=-500, morph_r=3):
    """Build a 3-D body mask via thresholding + morphological cleanup."""
    mask = vol_hu > threshold
    filled = np.zeros_like(mask, dtype=bool)
    kernel = np.ones((morph_r, morph_r), dtype=bool)
    for z in range(mask.shape[0]):
        filled[z] = binary_fill_holes(mask[z])
    clean = np.zeros_like(filled, dtype=bool)
    for z in range(filled.shape[0]):
        m = binary_opening(filled[z], structure=kernel)
        m = binary_closing(m, structure=kernel)
        clean[z] = m
    return largest_cc_3d(clean)


def compute_body_bbox(vol_hu, threshold=-500, margin=(4, 20, 20)):
    """Return (z0,z1,y0,y1,x0,x1) bounding box around non-background voxels."""
    body = vol_hu > threshold
    coords = np.argwhere(body)
    if len(coords) == 0:
        return (0, *vol_hu.shape)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    z0 = max(0, zmin - margin[0]);  z1 = min(vol_hu.shape[0], zmax + margin[0] + 1)
    y0 = max(0, ymin - margin[1]);  y1 = min(vol_hu.shape[1], ymax + margin[1] + 1)
    x0 = max(0, xmin - margin[2]);  x1 = min(vol_hu.shape[2], xmax + margin[2] + 1)
    return (z0, z1, y0, y1, x0, x1)


def crop_volume(vol, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    return vol[z0:z1, y0:y1, x0:x1]


def pad_to_div8(vol, n=8, fill=0):
    """Pad volume so each dimension is divisible by n."""
    z, y, x = vol.shape
    pz = (n - z % n) % n
    py = (n - y % n) % n
    px = (n - x % n) % n
    return np.pad(vol, ((0, pz), (0, py), (0, px)),
                  mode="constant", constant_values=fill)


def load_nifti_zyx(path):
    """Load a NIfTI file and transpose from XYZ to ZYX order."""
    nii  = nib.load(str(path))
    data = np.squeeze(np.asarray(nii.get_fdata()))
    return np.transpose(data, (2, 1, 0)).astype(np.float32)


# ============================================================
# Rendering helpers
# ============================================================

def ct_to_rgb(ct_2d):
    return np.stack([ct_2d, ct_2d, ct_2d], axis=-1).astype(np.float32)


def add_contour(ct_rgb, binary_mask, color, width=2):
    dilated = binary_dilation(binary_mask, iterations=width).astype(np.uint8)
    contour = dilated - binary_mask
    out = ct_rgb.copy()
    out[contour > 0] = color
    return np.clip(out, 0, 1)


def add_tpfpfn(ct_rgb, pred, gt, alpha=OVERLAY_ALPHA):
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out = ct_rgb.copy()
    for mask, color in [(tp, COLOR_TP), (fp, COLOR_FP), (fn, COLOR_FN)]:
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 1)


def zoom_crop(arr, zy0, zy1, zx0, zx1):
    return arr[zy0:zy1, zx0:zx1]


def save_panel(img_rgb, filepath, dpi=DPI):
    h, w = img_rgb.shape[:2]
    fig  = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax   = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_rgb, interpolation="nearest", aspect="equal")
    ax.axis("off")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  ✓  {Path(filepath).name}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate TP/FP/FN overlay panels for Figure 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--dicom_dir",    required=True,
                   help="Folder with the patient's original DICOM slices.")
    p.add_argument("--gt",           required=True,
                   help="GT binary mask in preprocessed space (.nii.gz).")
    p.add_argument("--attunet_prob", required=True,
                   help="Att U-Net ensemble probability map (.nii.gz).")
    p.add_argument("--unetr_prob",   required=True,
                   help="UNETR ensemble probability map (.nii.gz).")
    p.add_argument("--output_dir",   default="output_panels",
                   help="Directory to save output PNG files. (default: output_panels/)")
    p.add_argument("--threshold",    type=float, default=0.10,
                   help="Binarisation threshold for probability maps. "
                        "Use 0.10 / 0.50 / 0.90 to reproduce the three rows "
                        "of Figure 3. (default: 0.10)")
    return p.parse_args()


def main():
    args = parse_args()

    dicom_dir    = Path(args.dicom_dir)
    gt_path      = Path(args.gt)
    attunet_path = Path(args.attunet_prob)
    unetr_path   = Path(args.unetr_prob)
    output_dir   = Path(args.output_dir)
    threshold    = args.threshold

    print(f"\n{'='*60}")
    print(f"figure3_panels.py  |  threshold = {threshold:.2f}")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # 1. Load CT from DICOM and apply geometric preprocessing
    # ----------------------------------------------------------
    print("\n[1] Loading DICOM and applying geometric preprocessing...")
    ct_raw  = read_dicom_volume_hu(dicom_dir)

    body_mask = create_body_mask(ct_raw,
                                 threshold=BODY_THRESHOLD_HU,
                                 morph_r=BODY_MASK_MORPH_RAD)
    ct_body   = ct_raw.copy()
    ct_body[body_mask == 0] = BODY_FILL_HU

    bbox   = compute_body_bbox(ct_body,
                               threshold=BODY_THRESHOLD_HU,
                               margin=CROP_MARGIN)
    ct_crop = crop_volume(ct_body, bbox)
    ct_vol  = pad_to_div8(ct_crop, n=DIVISIBLE_BY, fill=BODY_FILL_HU)
    print(f"  Preprocessed CT shape: {ct_vol.shape}")

    # ----------------------------------------------------------
    # 2. Load NIfTI masks (already in preprocessed space)
    # ----------------------------------------------------------
    print("\n[2] Loading NIfTI masks...")
    gt_vol  = (load_nifti_zyx(gt_path) > 0.5).astype(np.uint8)
    au_prob = load_nifti_zyx(attunet_path)
    un_prob = load_nifti_zyx(unetr_path)

    au_mask = (au_prob >= threshold).astype(np.uint8)
    un_mask = (un_prob >= threshold).astype(np.uint8)

    print(f"  CT shape:       {ct_vol.shape}")
    print(f"  GT shape:       {gt_vol.shape}")
    print(f"  Att U-Net shape:{au_mask.shape}")
    print(f"  UNETR shape:    {un_mask.shape}")

    if ct_vol.shape != gt_vol.shape:
        raise RuntimeError(
            f"Shape mismatch: CT {ct_vol.shape} vs GT {gt_vol.shape}\n"
            "Verify that the preprocessing parameters match those used during training.")

    # ----------------------------------------------------------
    # 3. Select best axial slice (largest GT cross-section)
    # ----------------------------------------------------------
    gt_areas = gt_vol.sum(axis=(1, 2))
    best_z   = int(np.argmax(gt_areas))
    print(f"\n[3] Best axial slice: Z={best_z}  (GT area = {int(gt_areas[best_z])} px)")

    ct_slice = ct_vol[best_z]
    gt_slice = gt_vol[best_z]
    au_slice = au_mask[best_z]
    un_slice = un_mask[best_z]

    print(f"    Att U-Net TP: {int((au_slice & gt_slice).sum())} px")
    print(f"    UNETR    TP: {int((un_slice & gt_slice).sum())} px")

    # ----------------------------------------------------------
    # 4. HU windowing → [0, 1]
    # ----------------------------------------------------------
    hu_lo = HU_WINDOW_CENTER - HU_WINDOW_WIDTH / 2
    hu_hi = HU_WINDOW_CENTER + HU_WINDOW_WIDTH / 2
    ct_display = np.clip((ct_slice - hu_lo) / (hu_hi - hu_lo + 1e-8), 0, 1)
    print(f"\n[4] HU window: [{hu_lo:.0f}, {hu_hi:.0f}]  "
          f"(center={HU_WINDOW_CENTER}, width={HU_WINDOW_WIDTH})")

    # ----------------------------------------------------------
    # 5. Zoom bounding box (GT region + margin)
    # ----------------------------------------------------------
    coords = np.argwhere(gt_slice > 0)
    if len(coords) == 0:
        raise RuntimeError(f"GT is empty at Z={best_z}. Check mask alignment.")
    y0z, x0z = coords.min(axis=0)
    y1z, x1z = coords.max(axis=0)
    H, W = ct_slice.shape
    zy0 = max(0, y0z - ZOOM_MARGIN);  zy1 = min(H, y1z + ZOOM_MARGIN + 1)
    zx0 = max(0, x0z - ZOOM_MARGIN);  zx1 = min(W, x1z + ZOOM_MARGIN + 1)
    print(f"\n[5] Zoom: y=[{zy0}:{zy1}], x=[{zx0}:{zx1}]  ({zy1-zy0}×{zx1-zx0} px)")
    Z = (zy0, zy1, zx0, zx1)

    # ----------------------------------------------------------
    # 6. Build and save panels
    # ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[6] Saving panels to: {output_dir}\n")

    ct_rgb = ct_to_rgb(ct_display)
    gt_rgb = add_contour(ct_rgb,  gt_slice, COLOR_GT, width=CONTOUR_WIDTH)
    au_rgb = add_tpfpfn(ct_rgb,  au_slice, gt_slice)
    un_rgb = add_tpfpfn(ct_rgb,  un_slice, gt_slice)

    # Full-FOV panels
    save_panel(ct_rgb, output_dir / "ct.png")
    save_panel(gt_rgb, output_dir / "gt_contour.png")
    save_panel(au_rgb, output_dir / "attunet_overlay.png")
    save_panel(un_rgb, output_dir / "unetr_overlay.png")

    # Zoomed panels
    save_panel(zoom_crop(ct_rgb, *Z), output_dir / "ct_zoom.png")
    save_panel(zoom_crop(gt_rgb, *Z), output_dir / "gt_zoom.png")
    save_panel(zoom_crop(au_rgb, *Z), output_dir / "attunet_zoom.png")
    save_panel(zoom_crop(un_rgb, *Z), output_dir / "unetr_zoom.png")

    # ----------------------------------------------------------
    # 7. Review preview (2×4 grid)
    # ----------------------------------------------------------
    thr_str = f"{threshold:.2f}"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels_full = [ct_rgb, gt_rgb, au_rgb, un_rgb]
    panels_zoom = [zoom_crop(p, *Z) for p in panels_full]
    labels = ["CT", "GT contour",
              f"Att U-Net (thr={thr_str})",
              f"UNETR (thr={thr_str})"]

    for col, (full, zoom, label) in enumerate(zip(panels_full, panels_zoom, labels)):
        axes[0, col].imshow(full);   axes[0, col].set_title(label, fontsize=9)
        axes[1, col].imshow(zoom);   axes[1, col].set_title(f"{label} — zoom", fontsize=9)
        for row in [0, 1]:
            axes[row, col].axis("off")

    legend = [Patch(color=COLOR_TP, label="TP"),
              Patch(color=COLOR_FN, label="FN (under-seg)"),
              Patch(color=COLOR_FP, label="FP (over-seg)"),
              Patch(color=COLOR_GT, label="GT contour")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10, frameon=True)
    fig.suptitle(
        f"Patient 19 NIH  |  Z={best_z}  |  "
        f"HU window: center={HU_WINDOW_CENTER}, width={HU_WINDOW_WIDTH}  |  "
        f"threshold={thr_str}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_dir / "_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  _preview.png")

    n_files = len(list(output_dir.glob("*.png")))
    print(f"\n✅  Done — {n_files} files saved to {output_dir}")
    print(f"\nTips for fine-tuning appearance:")
    print(f"  HU_WINDOW_CENTER = {HU_WINDOW_CENTER}   (increase to brighten the image)")
    print(f"  HU_WINDOW_WIDTH  = {HU_WINDOW_WIDTH}  (decrease for higher contrast)")
    print(f"  OVERLAY_ALPHA    = {OVERLAY_ALPHA}  (0=transparent, 1=opaque overlay)")


if __name__ == "__main__":
    main()
