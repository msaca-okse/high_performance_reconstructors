"""
Neutron CT preprocessor.

Pipeline:
  1. Load and average open-beam (OB) flat-field FITS images.
  2. Process each projection exactly once (parallel):
       - Average 3 triplicated exposures per angle.
       - Compute per-image dose scalar D from the beam-monitor region.
       - Log-normalise with dose correction: -(log(proj) - log(OB) + log(D0) - log(D))
       - Zero pixels outside the circular beam (beam mask).
       - Morphological spot-clean on the full detector image.
       - Detector tilt and centre-of-rotation (COR) correction (SimpleITK).
       - Extract the target slice range.
  3. Assemble sinograms and save one TIFF per vertical slice.
  4. Optional ring removal (CIL RingRemover) on the assembled sinograms.

Usage:
  python neutron/preprocessor.py --config neutron/reconstruction_settings.yaml
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import yaml
from astropy.io import fits
from multiprocessing import Pool, cpu_count

# Allow importing imageutils from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageutils as iu

# Allow importing module_auxiliary (and related helpers) from the context folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "context"))
import module_auxiliary as ma

try:
    import SimpleITK as sitk
    _SITK_AVAILABLE = True
except ImportError:
    _SITK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Open-beam loading
# ---------------------------------------------------------------------------

def load_ob(cfg: dict):
    """Average N_OB open-beam FITS files.

    Returns
    -------
    ob_mean : float32 array of shape (n_pixels, n_pixels)
    D0      : float — median intensity in the beam-monitor region of ob_mean
    """
    data_root = Path(cfg["data_root"])
    ob_dir = data_root / cfg["subfolders"]["ob"]
    prefix = cfg["filenames"]["ob_prefix"]
    suffix = cfg["filenames"]["suffix"]
    n_ob = cfg["n_ob"]
    start_idx = cfg["ob_file_start"]
    n_pixels = cfg["n_pixels"]

    ob_mean = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    for i in range(n_ob):
        idx = start_idx + i
        path = ob_dir / f"{prefix}{idx:05d}{suffix}"
        ob_mean += fits.open(path)[0].data.astype(np.float64) / n_ob
        if (i + 1) % 50 == 0:
            print(f"  Loaded OB {i + 1}/{n_ob}")

    print("Done loading OB files.")

    # Crop to projection ROI (same cut applied to every raw FITS image)
    pr = cfg.get("proj_roi", {})
    ob_mean = ob_mean[
        pr.get("row_start", 0) : pr.get("row_end", None),
        pr.get("col_start", 0) : pr.get("col_end", None),
    ]

    # ob_monitor_region is in original pixel space → shift to cropped space
    r = cfg["ob_monitor_region"]
    row_offset = pr.get("row_start", 0)
    col_offset = pr.get("col_start", 0)
    D0 = float(np.median(ob_mean[
        r["row_start"] - row_offset : r["row_end"] - row_offset,
        r["col_start"] - col_offset : r["col_end"] - col_offset,
    ]))
    return ob_mean.astype(np.float32), D0


# ---------------------------------------------------------------------------
# Beam mask
# ---------------------------------------------------------------------------

def build_beam_mask(cfg: dict) -> np.ndarray:
    """Return a float32 mask (1 inside beam, 0 outside) for the cropped detector image.

    The beam is approximated as the largest circle that fits inside the
    bounding rectangle defined by beam_roi (given in ImageJ x/y convention:
    x = column, y = row, in the original uncropped pixel space).
    """
    pr = cfg.get("proj_roi", {})
    row_offset = pr.get("row_start", 0)
    col_offset = pr.get("col_start", 0)
    n_rows = pr.get("row_end", cfg["n_pixels"]) - row_offset
    n_cols = pr.get("col_end", cfg["n_pixels"]) - col_offset

    roi = cfg["beam_roi"]
    # beam_roi in original pixel space → shift to cropped space
    col_min = roi["x_min"] - col_offset
    col_max = roi["x_max"] - col_offset
    row_min = roi["y_min"] - row_offset
    row_max = roi["y_max"] - row_offset

    center_col = (col_min + col_max) / 2.0
    center_row = (row_min + row_max) / 2.0
    radius = min(col_max - col_min, row_max - row_min) / 2.0

    rows, cols = np.ogrid[:n_rows, :n_cols]
    mask = ((rows - center_row) ** 2 + (cols - center_col) ** 2) <= radius ** 2
    return mask.astype(np.float32)

# ---------------------------------------------------------------------------
# Geometry correction (tilt + COR)
# ---------------------------------------------------------------------------

def _correct_geometry(img: np.ndarray, gc_cfg: dict) -> np.ndarray:
    """Apply detector tilt and centre-of-rotation (COR) correction via SimpleITK.

    Implements the same 2D transform as the reference 3D CTcorrector applied
    per projection.  For a fixed angle, the 3D Y-axis rotation in the
    (pixel, slice) plane reduces to a 2D Euler transform:

        p_in_col = cos(α)·(col − w/2) − sin(α)·(row − h/2) + w/2 − cor_shift_px
        p_in_row = sin(α)·(col − w/2) + cos(α)·(row − h/2) + h/2

    The ``cor_shift_px`` parameter matches the ``translation`` variable in the
    reference code; it is negated when passed to SimpleITK, matching the
    reference's ``translation = [-translation, 0, 0]``.

    Parameters (gc_cfg keys)
    ------------------------
    tilt_angle_deg : float  —  detector tilt in degrees            (reference: 0.325)
    cor_shift_px   : float  —  COR shift variable (reference: −26);
                               SimpleITK x-translation = −cor_shift_px
    """
    angle_deg = gc_cfg.get("tilt_angle_deg", 0.0)
    cor_shift = gc_cfg.get("cor_shift_px", 0.0)

    if angle_deg == 0.0 and cor_shift == 0.0:
        return img

    if not _SITK_AVAILABLE:
        raise ImportError(
            "SimpleITK is required for geometry_correction. "
            "Install it with:  pip install SimpleITK"
        )

    h, w = img.shape
    sitk_img = sitk.GetImageFromArray(img)

    transform = sitk.Euler2DTransform()
    transform.SetCenter((w / 2.0, h / 2.0))
    transform.SetAngle(np.deg2rad(angle_deg))
    # Reference: 3D translation = [-cor_shift, 0, 0]  (x = pixel/col direction).
    # In 2D per-projection this becomes SetTranslation((-cor_shift, 0)).
    transform.SetTranslation((-cor_shift, 0.0))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    return sitk.GetArrayFromImage(resampler.Execute(sitk_img)).astype(np.float32)


# Module-level globals populated in each worker process by the Pool initializer
_ob_mean = None
_D0 = None
_cfg = None
_mask = None


def _worker_init(ob_mean, D0, cfg, mask):
    global _ob_mean, _D0, _cfg, _mask
    _ob_mean = ob_mean
    _D0 = D0
    _cfg = cfg
    _mask = mask


def _process_projection(i_proj: int) -> np.ndarray:
    """Process one projection angle.

    Each physical angle is recorded as a triplet of consecutive FITS files.
    File numbering is 1-based: angle i_proj (0-based) maps to files
    3*(i_proj+1)-2, 3*(i_proj+1)-1, 3*(i_proj+1).

    Returns
    -------
    result : float32 array of shape (n_slices, n_pixels)
        Log-normalised, spot-cleaned projection, cropped to the target row range.
        axis 0 = vertical detector row (z / sinogram index), axis 1 = horizontal pixel (u).
    """
    data_root = Path(_cfg["data_root"])
    proj_dir = data_root / _cfg["subfolders"]["projections"]
    prefix = _cfg["filenames"]["proj_prefix"]
    suffix = _cfg["filenames"]["suffix"]
    r = _cfg["ob_monitor_region"]
    slice_offset = _cfg["slice_offset"]
    n_slices = _cfg["n_slices"]

    i_file = i_proj + 1  # 1-based file counter
    def _read(k):
        return fits.open(proj_dir / f"{prefix}{k:05d}{suffix}")[0].data.astype(np.float64)

    # Median across the 3 triplicated exposures rejects single-frame cosmic ray hits
    proj = np.median(
        [_read(3 * i_file - 2), _read(3 * i_file - 1), _read(3 * i_file)],
        axis=0,
    )

    # Crop to projection ROI (all coords below are in cropped space)
    pr = _cfg.get("proj_roi", {})
    row_offset = pr.get("row_start", 0)
    col_offset = pr.get("col_start", 0)
    proj = proj[row_offset : pr.get("row_end", None), col_offset : pr.get("col_end", None)]

    # Per-image dose scalar (ob_monitor_region in original pixel space → shift to cropped)
    D = float(np.median(proj[
        r["row_start"] - row_offset : r["row_end"] - row_offset,
        r["col_start"] - col_offset : r["col_end"] - col_offset,
    ]))

    # Log-normalisation with dose correction
    ob_safe = np.clip(_ob_mean.astype(np.float64), 1.0, None)
    proj_safe = np.clip(proj, 1.0, None)
    normalized = -(np.log(proj_safe) - np.log(ob_safe) + np.log(_D0) - np.log(D))

    # Zero out pixels outside the beam circle to remove amplified edge artefacts
    normalized *= _mask

    # Morphological spot cleaning on the masked detector image
    sc = _cfg.get("spot_clean", {})
    cleaned = iu.morph_spot_clean(
        normalized,
        th_peaks=sc.get("th_peaks", 0.5),
        th_holes=sc.get("th_holes", 0.5),
        method=sc.get("method", 0),
        size=sc.get("size", 7),
    )

    # Detector tilt and COR correction
    corrected = _correct_geometry(cleaned, _cfg.get("geometry_correction", {}))

    # Crop to target row range (slice_offset is in original pixel space → shift to cropped)
    cropped_offset = slice_offset - row_offset
    row_end = None if n_slices is None else cropped_offset + n_slices
    return corrected[cropped_offset:row_end, :].astype(np.float32)


def _process_projection_debug(i_proj: int, ob_mean: np.ndarray, D0: float, cfg: dict, mask: np.ndarray):
    """Process one projection and return every intermediate stage for inspection.

    Returns
    -------
    raw        : float32 (n_rows_cropped, n_cols_cropped) — median triplet, pre-OB-correction (proj_roi applied)
    normalized : float32 (n_rows_cropped, n_cols_cropped) — log-normalised with dose correction
    masked     : float32 (n_rows_cropped, n_cols_cropped) — after beam mask (outside circle = 0)
    cleaned    : float32 (n_rows_cropped, n_cols_cropped) — after spot cleaning
    corrected  : float32 (n_rows_cropped, n_cols_cropped) — after tilt/COR geometry correction
    cropped    : float32 (n_slices, n_cols_cropped) — cropped to the target row range
    """
    data_root = Path(cfg["data_root"])
    proj_dir = data_root / cfg["subfolders"]["projections"]
    prefix = cfg["filenames"]["proj_prefix"]
    suffix = cfg["filenames"]["suffix"]
    r = cfg["ob_monitor_region"]
    slice_offset = cfg["slice_offset"]
    n_slices = cfg["n_slices"]

    i_file = i_proj + 1
    def _read(k):
        return fits.open(proj_dir / f"{prefix}{k:05d}{suffix}")[0].data.astype(np.float64)

    # Median across the 3 triplicated exposures rejects single-frame cosmic ray hits
    raw = np.median(
        [_read(3 * i_file - 2), _read(3 * i_file - 1), _read(3 * i_file)],
        axis=0,
    )

    # Crop to projection ROI
    pr = cfg.get("proj_roi", {})
    row_offset = pr.get("row_start", 0)
    col_offset = pr.get("col_start", 0)
    raw = raw[row_offset : pr.get("row_end", None), col_offset : pr.get("col_end", None)]

    D = float(np.median(raw[
        r["row_start"] - row_offset : r["row_end"] - row_offset,
        r["col_start"] - col_offset : r["col_end"] - col_offset,
    ]))

    ob_safe = np.clip(ob_mean.astype(np.float64), 1.0, None)
    proj_safe = np.clip(raw, 1.0, None)
    normalized = -(np.log(proj_safe) - np.log(ob_safe) + np.log(D0) - np.log(D))

    masked = normalized * mask

    sc = cfg.get("spot_clean", {})
    cleaned = iu.morph_spot_clean(
        masked,
        th_peaks=sc.get("th_peaks", 0.5),
        th_holes=sc.get("th_holes", 0.5),
        method=sc.get("method", 0),
        size=sc.get("size", 7),
    )

    corrected = _correct_geometry(cleaned, cfg.get("geometry_correction", {}))

    # Crop to target row range (slice_offset in original pixel space → shift to cropped)
    cropped_offset = slice_offset - row_offset
    row_end = None if n_slices is None else cropped_offset + n_slices
    cropped = corrected[cropped_offset:row_end, :].astype(np.float32)

    return raw.astype(np.float32), normalized.astype(np.float32), masked.astype(np.float32), cleaned.astype(np.float32), corrected, cropped


def save_debug_images(cfg: dict, ob_mean: np.ndarray, D0: float, mask: np.ndarray):
    """Save intermediate pipeline images for a few projections.

    Processes projections sequentially (not via pool) and saves to
    results_root/debug/:

      ob_mean.tiff                    — averaged open-beam image
      beam_mask.tiff                  — circular beam mask
      proj_NNNNN_1_raw.tiff           — averaged raw detector counts
      proj_NNNNN_2_normalized.tiff    — OB-corrected, log-normalised (absorption image)
      proj_NNNNN_3_masked.tiff        — after beam mask (outside circle zeroed)
      proj_NNNNN_4_cleaned.tiff       — after spot cleaning
      proj_NNNNN_5_corrected.tiff     — after tilt/COR geometry correction

    A sinogram TIFF from the full pipeline run is copied to the debug folder
    by the caller after preprocessing completes.
    """
    debug_cfg = cfg.get("debug", {})
    n_proj = cfg["n_projections"]

    # Default: 5 evenly-spaced projection indices
    proj_indices = debug_cfg.get("proj_indices") or [
        int(n_proj * i / 4) for i in range(5)
    ]

    results_root = Path(cfg["results_root"])
    debug_dir = results_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Debug mode: saving pipeline stages to {debug_dir} ──")

    # OB mean image and beam mask
    tifffile.imwrite(str(debug_dir / "ob_mean.tiff"), ob_mean)
    tifffile.imwrite(str(debug_dir / "beam_mask.tiff"), mask)
    print("  Saved ob_mean.tiff and beam_mask.tiff")

    # Per-projection stages
    for i_proj in proj_indices:
        print(f"  Processing projection {i_proj} ...", flush=True)
        t0 = time.time()
        raw, normalized, masked_img, cleaned, corrected, _ = _process_projection_debug(
            i_proj, ob_mean, D0, cfg, mask
        )
        print(f"    done in {time.time() - t0:.1f}s")
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_1_raw.tiff"), raw)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_2_normalized.tiff"), normalized)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_3_masked.tiff"), masked_img)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_4_cleaned.tiff"), cleaned)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_5_corrected.tiff"), corrected)
        print(f"    Saved stages 1–5 for projection {i_proj}")

    print(f"── Debug images written. Sinogram will be copied after full pipeline completes. ──\n")
def preprocess(cfg: dict, ob_mean: np.ndarray = None, D0: float = None) -> np.ndarray:
    """Run the full preprocessing pipeline.

    ob_mean and D0 may be passed in if they were already computed (e.g. during
    a debug run) to avoid loading the OB files twice.

    Returns
    -------
    sinograms : float32 array of shape (n_out_proj, n_slices, n_pixels)
        axis 0 = angle, axis 1 = vertical detector row (z), axis 2 = horizontal pixel (u)
        n_out_proj = ceil(n_projections / stride)
    """
    ob_mean, D0 = load_ob(cfg) if ob_mean is None else (ob_mean, D0)
    n_proj = cfg["n_projections"]
    stride = cfg.get("stride", 1)
    proj_indices = list(range(0, n_proj, stride))
    n_out = len(proj_indices)
    n_workers = min(cpu_count(), n_out)
    n_slices = cfg["n_slices"]  # may be None → use all rows from slice_offset

    mask = build_beam_mask(cfg)

    if stride > 1:
        print(f"Stride={stride}: using {n_out}/{n_proj} projections.")
    print(f"Processing {n_out} projections with {n_workers} workers ...")

    # Pre-allocate output. When n_slices is None we don't know the row count until
    # the first result arrives, so we collect results into a list in that case.
    n_slices = cfg["n_slices"]
    if n_slices is not None:
        pr = cfg.get("proj_roi", {})
        n_cols = (pr.get("col_end") or cfg["n_pixels"]) - pr.get("col_start", 0)
        sinograms = np.empty((n_out, n_slices, n_cols), dtype=np.float32)
        use_preallocated = True
    else:
        sinograms_list = []
        use_preallocated = False

    log_every = max(1, n_out // 20)  # print ~20 progress updates
    t0 = time.time()

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(ob_mean, D0, cfg, mask),
    ) as pool:
        for done, result in enumerate(pool.imap(_process_projection, proj_indices)):
            if use_preallocated:
                sinograms[done] = result
            else:
                sinograms_list.append(result)
            if (done + 1) % log_every == 0 or done + 1 == n_out:
                elapsed = time.time() - t0
                rate = (done + 1) / elapsed
                eta = (n_out - done - 1) / rate
                print(
                    f"  [{done + 1:>{len(str(n_out))}}/{n_out}] "
                    f"{elapsed:6.0f}s elapsed, ETA {eta:5.0f}s  ({rate:.2f} proj/s)",
                    flush=True,
                )

    if not use_preallocated:
        sinograms = np.stack(sinograms_list, axis=0)
        del sinograms_list

    del ob_mean
    gc.collect()
    return sinograms


# ---------------------------------------------------------------------------
# Gaussian edge padding
# ---------------------------------------------------------------------------

def _beam_edge_indices(cfg: dict) -> list:
    """Return beam left/right edge column positions for each target sinogram slice.

    ma.determine_edge2 fits a circle to the angular-variance image to find the
    beam boundary, but it requires the *full* sinogram stack (all detector rows)
    so that the top and bottom arcs of the circular beam are visible.  When only
    a subset of rows is used (as here), the fitter has nothing to lock onto and
    returns empty index lists — causing ma.gaussian_padding to zero every slice.

    Instead we derive the column positions analytically from beam_roi (the same
    circle used to build the beam mask).  The same ``bias`` inset is applied to
    the radius, matching what determine_edge2 does internally.

    Returns
    -------
    list of length n_slices; each element is one of:
      np.array([x_left, x_right], dtype=int)  — inside the beam
      np.array([], dtype=int)                 — outside the beam (row is zeroed)
    """
    pr      = cfg.get("proj_roi", {})
    row_off = pr.get("row_start", 0)
    col_off = pr.get("col_start", 0)

    roi = cfg["beam_roi"]
    # Beam circle in cropped-image pixel space (same geometry as build_beam_mask)
    center_col = (roi["x_min"] + roi["x_max"]) / 2.0 - col_off
    center_row = (roi["y_min"] + roi["y_max"]) / 2.0 - row_off
    radius     = min(roi["x_max"] - roi["x_min"], roi["y_max"] - roi["y_min"]) / 2.0

    bias     = cfg.get("gaussian_padding", {}).get("bias", 20)
    r_biased = max(radius - bias, 0.0)

    slice_offset = cfg["slice_offset"]
    n_slices     = cfg["n_slices"] or 0

    indices = []
    for j in range(n_slices):
        row_cropped = (slice_offset - row_off) + j   # this row in cropped-image space
        dy = row_cropped - center_row
        if abs(dy) >= r_biased:
            indices.append(np.array([], dtype=int))
        else:
            half_w  = np.sqrt(r_biased ** 2 - dy ** 2)
            x_left  = max(0, int(np.round(center_col - half_w)))
            x_right = int(np.round(center_col + half_w))
            indices.append(np.array([x_left, x_right], dtype=int))
    return indices


def gaussian_pad_sinograms(cfg: dict, sinograms: np.ndarray) -> np.ndarray:
    """Smoothly attenuate sinogram values to zero outside the circular beam region.

    Follows extended_data.ExtendedData.pad_edges() from the reference pipeline.
    Beam edge column positions are computed analytically from beam_roi (rather
    than via determine_edge2, which requires the full detector-height sinogram
    to fit a circle reliably).  ma.gaussian_padding is then called directly with
    those positions, using the same parameters as the reference pad_edges() call:
      bias=20, sigma=100, cutoff=10, pad_mean_window_size=50.

    Parameters (from cfg['gaussian_padding'])
    -----------------------------------------
    enabled              : bool  — run padding (default False)
    bias                 : int   — inward pixel bias on the detected edge (default 20)
    sigma                : float — Gaussian std-dev of the ramp, pixels (default 100)
    cutoff               : int   — ramp is zero beyond cutoff*sigma px  (default 10)
    pad_mean_window_size : int   — pixels inward from edge for the reference
                                   mean level of the ramp               (default 50)

    Parameters
    ----------
    sinograms : float32 (n_proj, n_slices, n_pixels) — from preprocess()

    Returns
    -------
    float32 (n_proj, n_slices, n_pixels) — edge-padded sinograms
    """
    gp_cfg = cfg.get("gaussian_padding", {})
    sigma                = gp_cfg.get("sigma", 100)
    cutoff               = gp_cfg.get("cutoff", 10)
    pad_mean_window_size = gp_cfg.get("pad_mean_window_size", 50)

    indices = _beam_edge_indices(cfg)

    # ma.gaussian_padding expects (N_slices, N_angles, N_pixels)
    data_szp = sinograms.transpose(1, 0, 2)

    print(f"Applying Gaussian padding (sigma={sigma}, cutoff={cutoff}, "
          f"pad_mean_window_size={pad_mean_window_size}) ...")
    t0 = time.time()
    data_szp = ma.gaussian_padding(
        data_szp, indices,
        sigma=sigma,
        cutoff=cutoff,
        pad_mean_window_size=pad_mean_window_size,
    )
    print(f"  Gaussian padding done in {time.time() - t0:.1f}s")

    return data_szp.transpose(1, 0, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Ring removal
# ---------------------------------------------------------------------------

def remove_rings(cfg: dict, sinograms: np.ndarray) -> np.ndarray:
    """Apply CIL RingRemover to the assembled sinogram stack.

    Ring artefacts appear as vertical stripes in sinograms and concentric rings
    in reconstructed slices.  CIL's RingRemover applies a wavelet-FFT filter
    independently per sinogram row (horizontal line at fixed angle across all
    detector columns), which is column-wise in CIL's standard sinogram layout.

    This function mirrors the ``sinograms.remove_ring(subdata=False, ...)``
    call in the reference ``extended_data.py``.

    Parameters (from cfg['ring_removal'])
    --------------------------------------
    enabled : bool  — run ring removal (default False)
    decNum  : int   — wavelet decomposition levels (default 4; ref: 5)
    wname   : int   — wavelet order; CIL uses 'db{wname}' (default 5; ref: 10)
    sigma   : float — Tikhonov regularisation strength (default 0.1; ref: 0.3)

    Parameters
    ----------
    sinograms : float32 (n_proj, n_slices, n_pixels) — from preprocess()

    Returns
    -------
    float32 (n_proj, n_slices, n_pixels) — ring-corrected sinograms
    """
    from cil.framework import AcquisitionGeometry, AcquisitionData
    from cil.processors import RingRemover

    rr_cfg = cfg.get("ring_removal", {})
    decNum = rr_cfg.get("decNum", 4)
    wname  = f"db{rr_cfg.get('wname', 5)}"
    sigma  = rr_cfg.get("sigma", 0.1)

    n_proj, n_slices, n_pixels = sinograms.shape
    angles = np.linspace(0, 360, n_proj, endpoint=True, dtype=np.float32)

    # CIL AcquisitionData expects layout ('vertical', 'angle', 'horizontal').
    # Our sinograms: axis0=angle, axis1=vertical_slice (z), axis2=horizontal_pixel (u).
    # Transpose: (n_proj, n_slices, n_pixels) -> (n_slices, n_proj, n_pixels)
    data_cil = sinograms.transpose(1, 0, 2).copy()

    ag = (
        AcquisitionGeometry.create_Parallel3D(detector_position=[0, n_pixels // 2, 0])
        .set_angles(angles)
        .set_panel((n_pixels, n_slices), pixel_size=(1, 1))
        .set_labels(labels=("vertical", "angle", "horizontal"))
    )
    acq_data = AcquisitionData(data_cil, deep_copy=False, geometry=ag)

    print(f"Applying ring removal (decNum={decNum}, wname={wname}, sigma={sigma}) ...")
    t0 = time.time()
    ring_remover = RingRemover(decNum=decNum, wname=wname, sigma=sigma, info=True)
    ring_remover.set_input(acq_data)
    result = ring_remover.get_output()
    print(f"  Ring removal done in {time.time() - t0:.1f}s")

    # Transpose back to (n_proj, n_slices, n_pixels)
    return result.as_array().transpose(1, 0, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_sinograms(cfg: dict, sinograms: np.ndarray):
    """Save one sinogram TIFF per vertical slice.

    Each file is a 2D array (n_angles, n_pixels) named by its absolute
    detector-row index in the full detector image: sinogram_NNNNN.tiff.
    """
    scratch_root = Path(cfg["scratch_root"])
    scratch_root.mkdir(parents=True, exist_ok=True)
    slice_offset = cfg["slice_offset"]
    n_slices = sinograms.shape[1]  # axis 1 = vertical row (z)

    print(f"Saving {n_slices} sinograms to {scratch_root} ...")
    for j in range(n_slices):
        sino = sinograms[:, j, :]  # (n_angles, n_pixels)
        out_path = scratch_root / f"sinogram_{slice_offset + j:05d}.tiff"
        tifffile.imwrite(str(out_path), sino)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neutron CT preprocessing: OB correction, log-normalisation, spot cleaning."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load OB once — reused by both debug output and the main preprocessing pool
    ob_mean, D0 = load_ob(cfg)

    if cfg.get("debug", {}).get("enabled", False):
        mask = build_beam_mask(cfg)
        save_debug_images(cfg, ob_mean, D0, mask)

    sinograms = preprocess(cfg, ob_mean, D0)

    if cfg.get("gaussian_padding", {}).get("enabled", False):
        sinograms = gaussian_pad_sinograms(cfg, sinograms)

    if cfg.get("ring_removal", {}).get("enabled", False):
        sinograms = remove_rings(cfg, sinograms)

    save_sinograms(cfg, sinograms)

    # Copy one sinogram into the debug folder so the full pipeline output
    # can be inspected alongside the per-projection debug images.
    if cfg.get("debug", {}).get("enabled", False):
        import shutil
        slice_offset = cfg["slice_offset"]
        n_slices = sinograms.shape[2]
        mid_col = slice_offset + n_slices // 2
        src = Path(cfg["scratch_root"]) / f"sinogram_{mid_col:05d}.tiff"
        dst = Path(cfg["results_root"]) / "debug" / f"sinogram_col{mid_col:05d}.tiff"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Debug: copied sinogram for col {mid_col} to {dst}")
