"""
Neutron CT preprocessor.

Pipeline:
  1. Load and average open-beam (OB) flat-field FITS images.
  2. Process each projection exactly once (parallel):
       - Average 3 triplicated exposures per angle.
       - Compute per-image dose scalar D from the beam-monitor region.
       - Log-normalise with dose correction: -(log(proj) - log(OB) + log(D0) - log(D))
       - Spot-clean on the full detector image.
       - Extract the target slice range.
  3. Assemble sinograms and save one TIFF per vertical slice.

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

    r = cfg["ob_monitor_region"]
    D0 = float(np.median(ob_mean[r["row_start"]:r["row_end"], r["col_start"]:r["col_end"]]))
    return ob_mean.astype(np.float32), D0


# ---------------------------------------------------------------------------
# Beam mask
# ---------------------------------------------------------------------------

def build_beam_mask(cfg: dict) -> np.ndarray:
    """Return a float32 mask (1 inside beam, 0 outside) for the full detector.

    The beam is approximated as the largest circle that fits inside the
    bounding rectangle defined by beam_roi (given in ImageJ x/y convention:
    x = column, y = row).
    """
    n_pixels = cfg["n_pixels"]
    roi = cfg["beam_roi"]
    col_min, col_max = roi["x_min"], roi["x_max"]
    row_min, row_max = roi["y_min"], roi["y_max"]

    center_col = (col_min + col_max) / 2.0
    center_row = (row_min + row_max) / 2.0
    radius = min(col_max - col_min, row_max - row_min) / 2.0

    rows, cols = np.ogrid[:n_pixels, :n_pixels]
    mask = ((rows - center_row) ** 2 + (cols - center_col) ** 2) <= radius ** 2
    return mask.astype(np.float32)

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
    result : float32 array of shape (n_pixels, n_slices)
        Log-normalised, spot-cleaned projection, cropped to the target slice range.
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

    proj = (_read(3 * i_file - 2) + _read(3 * i_file - 1) + _read(3 * i_file)) / 3.0

    # Per-image dose scalar
    D = float(np.median(proj[r["row_start"]:r["row_end"], r["col_start"]:r["col_end"]]))

    # Log-normalisation with dose correction
    ob_safe = np.clip(_ob_mean.astype(np.float64), 1.0, None)
    proj_safe = np.clip(proj, 1.0, None)
    normalized = -(np.log(proj_safe) - np.log(ob_safe) + np.log(_D0) - np.log(D))

    # Zero out pixels outside the beam circle to remove amplified edge artefacts
    normalized *= _mask

    # Spot cleaning on the masked detector image
    cleaned = iu.spotclean(normalized, size=10)

    # Crop to target slice range (n_slices=None means take everything from slice_offset onwards)
    col_end = None if n_slices is None else slice_offset + n_slices
    return cleaned[:, slice_offset:col_end].astype(np.float32)


def _process_projection_debug(i_proj: int, ob_mean: np.ndarray, D0: float, cfg: dict, mask: np.ndarray):
    """Process one projection and return every intermediate stage for inspection.

    Returns
    -------
    raw        : float32 (n_pixels, n_pixels) — averaged triplet, uncorrected counts
    normalized : float32 (n_pixels, n_pixels) — log-normalised with dose correction
    masked     : float32 (n_pixels, n_pixels) — after beam mask (outside circle = 0)
    cleaned    : float32 (n_pixels, n_pixels) — after spot cleaning
    cropped    : float32 (n_pixels, n_slices) — cropped to the target slice range
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

    raw = (_read(3 * i_file - 2) + _read(3 * i_file - 1) + _read(3 * i_file)) / 3.0
    D = float(np.median(raw[r["row_start"]:r["row_end"], r["col_start"]:r["col_end"]]))

    ob_safe = np.clip(ob_mean.astype(np.float64), 1.0, None)
    proj_safe = np.clip(raw, 1.0, None)
    normalized = -(np.log(proj_safe) - np.log(ob_safe) + np.log(D0) - np.log(D))

    masked = normalized * mask

    cleaned = iu.spotclean(masked, size=10)

    col_end = None if n_slices is None else slice_offset + n_slices
    cropped = cleaned[:, slice_offset:col_end].astype(np.float32)

    return raw.astype(np.float32), normalized.astype(np.float32), masked.astype(np.float32), cleaned.astype(np.float32), cropped


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
        raw, normalized, masked_img, cleaned, _ = _process_projection_debug(
            i_proj, ob_mean, D0, cfg, mask
        )
        print(f"    done in {time.time() - t0:.1f}s")
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_1_raw.tiff"), raw)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_2_normalized.tiff"), normalized)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_3_masked.tiff"), masked_img)
        tifffile.imwrite(str(debug_dir / f"proj_{i_proj:05d}_4_cleaned.tiff"), cleaned)
        print(f"    Saved stages 1–4 for projection {i_proj}")

    print(f"── Debug images written. Sinogram will be copied after full pipeline completes. ──\n")

def preprocess(cfg: dict, ob_mean: np.ndarray = None, D0: float = None) -> np.ndarray:
    """Run the full preprocessing pipeline.

    ob_mean and D0 may be passed in if they were already computed (e.g. during
    a debug run) to avoid loading the OB files twice.

    Returns
    -------
    sinograms : float32 array of shape (n_out_proj, n_pixels, n_slices)
        n_out_proj = ceil(n_projections / stride)
    """
    ob_mean, D0 = load_ob(cfg) if ob_mean is None else (ob_mean, D0)
    n_proj = cfg["n_projections"]
    stride = cfg.get("stride", 1)
    proj_indices = list(range(0, n_proj, stride))
    n_out = len(proj_indices)
    n_workers = min(cpu_count(), n_out)
    n_slices = cfg["n_slices"]  # may be None → use all columns from slice_offset

    mask = build_beam_mask(cfg)

    if stride > 1:
        print(f"Stride={stride}: using {n_out}/{n_proj} projections.")
    print(f"Processing {n_out} projections with {n_workers} workers ...")

    # Pre-allocate output. When n_slices is None we don't know the column count until
    # the first result arrives, so we collect results into a list in that case.
    n_slices = cfg["n_slices"]
    if n_slices is not None:
        n_pixels = cfg["n_pixels"]
        sinograms = np.empty((n_out, n_pixels, n_slices), dtype=np.float32)
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
# Save
# ---------------------------------------------------------------------------

def save_sinograms(cfg: dict, sinograms: np.ndarray):
    """Save one sinogram TIFF per vertical slice.

    Each file is a 2D array (n_angles, n_pixels) named by its absolute
    column index in the full detector image: sinogram_NNNNN.tiff.
    """
    scratch_root = Path(cfg["scratch_root"])
    scratch_root.mkdir(parents=True, exist_ok=True)
    slice_offset = cfg["slice_offset"]
    n_slices = sinograms.shape[2]

    print(f"Saving {n_slices} sinograms to {scratch_root} ...")
    for j in range(n_slices):
        sino = sinograms[:, :, j]  # (n_angles, n_pixels)
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
