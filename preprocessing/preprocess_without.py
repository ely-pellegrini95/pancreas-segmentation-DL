"""
Preprocessing pipeline WITHOUT full preprocessing (baseline).

Steps applied to each CT volume:
  1. Read DICOM slices and convert to Hounsfield Units (HU)
  2. Load NIfTI segmentation mask
  3. Normalize to [0, 1]   ← no body crop, no pad, no clipping, no CLAHE
  4. Save CT and mask as NIfTI (.nii.gz)

Output filenames follow the convention: PANCREAS_XXXX.nii.gz
"""

import re
import numpy as np
from pathlib import Path

import pydicom
import nibabel as nib
import pandas as pd

# ===========================================================
# 0) Parameters — adjust paths before running
# ===========================================================

DICOM_ROOT  = Path(r"path/to/pancreas_ct_dcm")      # one subfolder per patient
MASK_ROOT   = Path(r"path/to/pancreas_ct_nifti")    # .nii or .nii.gz masks
OUTPUT_ROOT = Path(r"path/to/output_without_preprocessing")

# ===========================================================
# 1) General utilities
# ===========================================================

def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]

def extract_int_id(text):
    nums = re.findall(r"\d+", str(text))
    if not nums:
        return None
    try:
        return int("".join(nums))
    except ValueError:
        return None

def list_case_dirs(dicom_root):
    case_dirs = []
    for p in sorted(dicom_root.iterdir(), key=natural_key):
        if p.is_dir() and any(f.is_file() for f in p.rglob("*")):
            case_dirs.append(p)
    return case_dirs

def list_mask_files(mask_root):
    return sorted(
        list(mask_root.glob("*.nii")) + list(mask_root.glob("*.nii.gz")),
        key=natural_key
    )

def find_mask_for_case(case_dir, mask_root):
    mask_files = list_mask_files(mask_root)
    if not mask_files:
        return None
    case_name = case_dir.name.lower()
    case_id   = extract_int_id(case_name)
    if case_id is not None:
        for m in mask_files:
            if extract_int_id(m.stem) == case_id:
                return m
    for m in mask_files:
        mname = m.name.lower()
        if case_name in mname or m.stem.lower() in case_name:
            return m
    return None

# ===========================================================
# 2) DICOM reading and HU conversion
# ===========================================================

def read_dicom_slices(case_dir):
    slices = []
    for fp in sorted(case_dir.rglob("*"), key=natural_key):
        if not fp.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"No DICOM files with pixel data found in {case_dir}")
    return slices

def ds_to_hu(ds):
    arr       = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept

def get_orientation_vectors(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return (np.array([1., 0., 0.], np.float32),
                np.array([0., 1., 0.], np.float32),
                np.array([0., 0., 1.], np.float32))
    iop     = np.array(iop, np.float32)
    row_cos = iop[:3]
    col_cos = iop[3:]
    normal  = np.cross(row_cos, col_cos)
    return row_cos, col_cos, normal

def slice_position_along_normal(ds, normal):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return float(getattr(ds, "InstanceNumber", 0))
    return float(np.dot(np.array(ipp, np.float32), normal))

def get_spacing(sorted_slices, normal):
    first = sorted_slices[0]
    ps    = getattr(first, "PixelSpacing", [1.0, 1.0])
    positions = [slice_position_along_normal(ds, normal) for ds in sorted_slices]
    if len(positions) > 1:
        diffs = np.abs(np.diff(positions))
        diffs = diffs[diffs != 0]
        slice_spacing = float(np.median(diffs)) if len(diffs) > 0 else float(getattr(first, "SliceThickness", 1.0))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    return (slice_spacing, float(ps[0]), float(ps[1])), positions

def build_sorted_volume(case_dir):
    slices                   = read_dicom_slices(case_dir)
    row_cos, col_cos, normal = get_orientation_vectors(slices[0])
    sorted_slices            = sorted(slices, key=lambda ds: slice_position_along_normal(ds, normal))
    volume                   = np.stack([ds_to_hu(ds) for ds in sorted_slices], axis=0)
    spacing, positions       = get_spacing(sorted_slices, normal)
    meta = {"row_cos": row_cos, "col_cos": col_cos, "normal": normal,
            "positions": positions, "case_dir": case_dir}
    return volume, spacing, meta

# ===========================================================
# 3) NIfTI mask loading
# ===========================================================

def load_mask_zyx(mask_path):
    nii  = nib.load(str(mask_path))
    data = np.squeeze(np.asarray(nii.get_fdata()))
    if data.ndim != 3:
        raise RuntimeError(f"Mask {mask_path.name} is not 3D (shape={data.shape})")
    return (np.transpose(data, (2, 1, 0)) > 0).astype(np.uint8)

# ===========================================================
# 4) Normalization only (no clipping, no CLAHE)
# ===========================================================

def normalize_01(volume):
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), (vmin, vmax)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32), (vmin, vmax)

# ===========================================================
# 5) Preprocessing pipeline WITHOUT body crop / pad / CLAHE
# ===========================================================

def preprocess_case_without(case_dir, mask_path):
    """
    Minimal (baseline) preprocessing pipeline:
    DICOM → HU → normalize [0, 1]
    No body mask, no crop, no padding, no intensity clipping, no CLAHE.
    Returns the normalised CT array (float32, [0,1]) and the binary mask.
    """
    volume, spacing, meta = build_sorted_volume(case_dir)

    try:
        mask_zyx = load_mask_zyx(mask_path)
    except Exception as e:
        print(f"  [WARN] Could not load mask {mask_path.name}: {e}")
        mask_zyx = None

    if mask_zyx is not None and mask_zyx.shape != volume.shape:
        print(f"  [WARN] Mask shape {mask_zyx.shape} != CT shape {volume.shape}. Mask skipped.")
        mask_zyx = None

    # Normalize to [0, 1] — no clipping, no body crop, no CLAHE
    norm_01, _ = normalize_01(volume.astype(np.float32))

    return norm_01, mask_zyx, spacing

# ===========================================================
# 6) Save as NIfTI
# ===========================================================

def save_zyx_as_nifti(volume_zyx, out_path, dtype=np.float32):
    vol_xyz = np.transpose(volume_zyx, (2, 1, 0)).astype(dtype)
    nib.save(nib.Nifti1Image(vol_xyz, np.eye(4, dtype=np.float32)), str(out_path))

# ===========================================================
# 7) Batch processing
# ===========================================================

def run_batch():
    OUT_CT   = OUTPUT_ROOT / "images"
    OUT_MASK = OUTPUT_ROOT / "labels"
    for p in [OUT_CT, OUT_MASK]:
        p.mkdir(parents=True, exist_ok=True)

    case_dirs = list_case_dirs(DICOM_ROOT)
    if not case_dirs:
        raise RuntimeError(f"No patient folders found in: {DICOM_ROOT}")
    print(f"Found {len(case_dirs)} patients.")

    summary_rows = []
    saved = skipped = errors = 0

    for i, case_dir in enumerate(case_dirs, 1):
        print(f"\n[{i}/{len(case_dirs)}] {case_dir.name}")
        mask_path = find_mask_for_case(case_dir, MASK_ROOT)

        if mask_path is None:
            print("  [SKIP] No mask found.")
            summary_rows.append({"case": case_dir.name, "status": "skipped_no_mask"})
            skipped += 1
            continue

        try:
            ct_norm, mask_final, spacing = preprocess_case_without(case_dir, mask_path)

            if mask_final is None:
                print("  [SKIP] Final mask is None.")
                summary_rows.append({"case": case_dir.name, "status": "skipped_null_mask"})
                skipped += 1
                continue

            out_ct   = OUT_CT   / f"{case_dir.name}.nii.gz"
            out_mask = OUT_MASK / f"{case_dir.name}.nii.gz"
            save_zyx_as_nifti(ct_norm,    out_ct,   dtype=np.float32)
            save_zyx_as_nifti(mask_final, out_mask, dtype=np.uint8)

            print(f"  [OK] CT   -> {out_ct}")
            print(f"       MASK -> {out_mask}")
            print(f"       CT range: {ct_norm.min():.4f} – {ct_norm.max():.4f}")
            print(f"       Mask values: {np.unique(mask_final).tolist()}")

            summary_rows.append({
                "case": case_dir.name, "status": "saved",
                "spacing_z": spacing[0], "spacing_y": spacing[1], "spacing_x": spacing[2],
                "shape_z": ct_norm.shape[0], "shape_y": ct_norm.shape[1], "shape_x": ct_norm.shape[2],
                "out_ct": str(out_ct), "out_mask": str(out_mask),
            })
            saved += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            summary_rows.append({"case": case_dir.name, "status": "error", "error": str(e)})
            errors += 1

    csv_path = OUTPUT_ROOT / "preprocessing_summary_without.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    run_batch()
