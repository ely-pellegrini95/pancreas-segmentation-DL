"""
Preprocessing pipeline WITH full preprocessing.

Steps applied to each CT volume:
  1. Read DICOM slices and convert to Hounsfield Units (HU)
  2. Load NIfTI segmentation mask
  3. Generate body mask and remove scanner table
  4. Crop volume to body bounding box (+ margin)
  5. Pad to shape divisible by 8
  6. Percentile clipping (0.5 – 99.5)
  7. Multiplanar CLAHE contrast enhancement
  8. Normalize to [0, 1]
  9. Save CT and mask as NIfTI (.nii.gz)

Output filenames follow the convention: PANCREAS_XXXX.nii.gz
"""

import re
import numpy as np
from pathlib import Path

import pydicom
import nibabel as nib
import cv2
import pandas as pd
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing, label

# ===========================================================
# 0) Parameters — adjust paths before running
# ===========================================================

DICOM_ROOT  = Path(r"path/to/pancreas_ct_dcm")      # one subfolder per patient
MASK_ROOT   = Path(r"path/to/pancreas_ct_nifti")    # .nii or .nii.gz masks
OUTPUT_ROOT = Path(r"path/to/output_with_preprocessing")

# Body mask
BODY_THRESHOLD_HU    = -500
BODY_FILL_HU         = -1000.0
BODY_MASK_MORPH_RADIUS = 3

# Crop
CROP_MARGIN  = (4, 20, 20)   # (z, y, x) voxels
DIVISIBLE_BY = 8 # output shape must be divisible by this for the network

# Clipping
CLIP_P_LOW  = 0.5
CLIP_P_HIGH = 99.5

# CLAHE
CLAHE_CLIP_LIMIT    = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)

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
    slices              = read_dicom_slices(case_dir)
    row_cos, col_cos, normal = get_orientation_vectors(slices[0])
    sorted_slices       = sorted(slices, key=lambda ds: slice_position_along_normal(ds, normal))
    volume              = np.stack([ds_to_hu(ds) for ds in sorted_slices], axis=0)
    spacing, positions  = get_spacing(sorted_slices, normal)
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
# 4) Body mask and crop
# ===========================================================

def largest_connected_component_3d(mask):
    lbl, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts     = np.bincount(lbl.ravel())
    counts[0]  = 0
    return (lbl == np.argmax(counts)).astype(np.uint8)

def create_body_mask(volume_hu, threshold_hu=-500, morph_radius=3):
    mask   = volume_hu > threshold_hu
    filled = np.zeros_like(mask, dtype=bool)
    kernel = np.ones((morph_radius, morph_radius), dtype=bool)
    clean  = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        filled[z] = binary_fill_holes(mask[z])
    for z in range(filled.shape[0]):
        m       = binary_opening(filled[z], structure=kernel)
        m       = binary_closing(m,          structure=kernel)
        clean[z] = m
    return largest_connected_component_3d(clean).astype(np.uint8)

def apply_body_mask(volume_hu, body_mask, fill_value=-1000):
    out = volume_hu.copy()
    out[body_mask == 0] = fill_value
    return out.astype(np.float32)

def compute_body_bbox(volume_hu, threshold_hu=-500, margin=(4, 20, 20)):
    body   = volume_hu > threshold_hu
    coords = np.argwhere(body)
    if len(coords) == 0:
        return (0, volume_hu.shape[0], 0, volume_hu.shape[1], 0, volume_hu.shape[2])
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return (
        max(0, zmin - margin[0]),  min(volume_hu.shape[0], zmax + margin[0] + 1),
        max(0, ymin - margin[1]),  min(volume_hu.shape[1], ymax + margin[1] + 1),
        max(0, xmin - margin[2]),  min(volume_hu.shape[2], xmax + margin[2] + 1),
    )

def crop_volume(volume, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    return volume[z0:z1, y0:y1, x0:x1]

def pad_to_divisible_by_n(volume, n=8, fill_value=0):
    z, y, x = volume.shape
    pad_width = ((0, (n - z % n) % n), (0, (n - y % n) % n), (0, (n - x % n) % n))
    return np.pad(volume, pad_width, mode="constant", constant_values=fill_value), pad_width

def pad_mask_to_divisible_by_n(mask, n=8):
    if mask is None:
        return None, None
    return pad_to_divisible_by_n(mask, n=n, fill_value=0)

# ===========================================================
# 5) Clipping, CLAHE, normalization
# ===========================================================

def percentile_clip(volume, p_low=0.5, p_high=99.5):
    low     = float(np.percentile(volume, p_low))
    high    = float(np.percentile(volume, p_high))
    clipped = np.clip(volume, low, high)
    return clipped.astype(np.float32), (low, high)

def to_uint8(volume, vmin=None, vmax=None):
    vmin = float(volume.min()) if vmin is None else vmin
    vmax = float(volume.max()) if vmax is None else vmax
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.uint8)
    out = np.clip((volume - vmin) / (vmax - vmin), 0, 1)
    return (out * 255.0).round().astype(np.uint8)

def apply_clahe_2d(img2d_uint8, clip_limit=1.5, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img2d_uint8)

def apply_multiplanar_clahe(volume_uint8, clip_limit=1.5, tile_grid_size=(8, 8)):
    """Apply CLAHE on axial, coronal and sagittal planes, then average."""
    vz, vy, vx = volume_uint8.shape
    out_ax = np.zeros_like(volume_uint8, dtype=np.float32)
    out_co = np.zeros_like(volume_uint8, dtype=np.float32)
    out_sa = np.zeros_like(volume_uint8, dtype=np.float32)
    for z in range(vz):
        out_ax[z]    = apply_clahe_2d(volume_uint8[z],    clip_limit, tile_grid_size)
    for y in range(vy):
        out_co[:, y, :] = apply_clahe_2d(volume_uint8[:, y, :], clip_limit, tile_grid_size)
    for x in range(vx):
        out_sa[:, :, x] = apply_clahe_2d(volume_uint8[:, :, x], clip_limit, tile_grid_size)
    out = (out_ax + out_co + out_sa) / 3.0
    return np.clip(out, 0, 255).astype(np.uint8)

def normalize_01(volume):
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), (vmin, vmax)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32), (vmin, vmax)

# ===========================================================
# 6) Full preprocessing pipeline (with CLAHE)
# ===========================================================

def preprocess_case_with(case_dir, mask_path):
    """
    Full preprocessing pipeline:
    DICOM -> HU -> body mask/crop -> clipping -> multiplanar CLAHE -> normalize [0,1]
    Returns the final CT array (float32, [0,1]) and the cropped binary mask.
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

    # No reorientation (homogeneous NIH dataset)
    reoriented      = volume.copy()
    mask_reoriented = (mask_zyx.copy() > 0).astype(np.uint8) if mask_zyx is not None else None

    # Body mask and table removal
    body_mask = create_body_mask(reoriented, threshold_hu=BODY_THRESHOLD_HU,
                                 morph_radius=BODY_MASK_MORPH_RADIUS)
    body_only = apply_body_mask(reoriented, body_mask, fill_value=BODY_FILL_HU)

    # Crop
    bbox              = compute_body_bbox(body_only, threshold_hu=BODY_THRESHOLD_HU, margin=CROP_MARGIN)
    cropped           = crop_volume(body_only, bbox)
    body_mask_cropped = crop_volume(body_mask, bbox)
    mask_cropped      = crop_volume(mask_reoriented, bbox) if mask_reoriented is not None else None

    # Pad to divisible by 8
    cropped_div8, _   = pad_to_divisible_by_n(cropped, n=DIVISIBLE_BY, fill_value=BODY_FILL_HU)
    body_mask_div8, _ = pad_mask_to_divisible_by_n(body_mask_cropped, n=DIVISIBLE_BY)
    mask_div8, _      = pad_mask_to_divisible_by_n(mask_cropped, n=DIVISIBLE_BY)

    # Clipping
    clipped, clip_bounds = percentile_clip(cropped_div8, p_low=CLIP_P_LOW, p_high=CLIP_P_HIGH)

    # Multiplanar CLAHE
    clipped_u8   = to_uint8(clipped, vmin=clip_bounds[0], vmax=clip_bounds[1])
    clahe_u8     = apply_multiplanar_clahe(clipped_u8, clip_limit=CLAHE_CLIP_LIMIT,
                                           tile_grid_size=CLAHE_TILE_GRID_SIZE)

    # Normalize to [0, 1]
    norm_01, _ = normalize_01(clahe_u8.astype(np.float32))

    return norm_01, mask_div8, spacing, bbox

# ===========================================================
# 7) Save as NIfTI
# ===========================================================

def save_zyx_as_nifti(volume_zyx, out_path, dtype=np.float32):
    vol_xyz = np.transpose(volume_zyx, (2, 1, 0)).astype(dtype)
    nib.save(nib.Nifti1Image(vol_xyz, np.eye(4, dtype=np.float32)), str(out_path))

# ===========================================================
# 8) Batch processing
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
            ct_norm, mask_final, spacing, bbox = preprocess_case_with(case_dir, mask_path)

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

    csv_path = OUTPUT_ROOT / "preprocessing_summary_with.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    run_batch()
