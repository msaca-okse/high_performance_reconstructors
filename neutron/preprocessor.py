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
# Per-projection worker (runs in subprocess via Pool)
# ---------------------------------------------------------------------------

# Module-level globals populated in each worker process by the Pool initializer
_ob_mean = None
_D0 = None
_cfg = None


def _worker_init(ob_mean, D0, cfg):
    global _ob_mean, _D0, _cfg
    _ob_mean = ob_mean
    _D0 = D0
    _cfg = cfg


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

    # Spot cleaning on the full detector image
    cleaned = iu.spotclean(normalized, size=10)

    # Crop to target slice range
    return cleaned[:, slice_offset: slice_offset + n_slices].astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing orchestration
# ---------------------------------------------------------------------------

def preprocess(cfg: dict) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Returns
    -------
    sinograms : float32 array of shape (n_out_proj, n_pixels, n_slices)
        n_out_proj = ceil(n_projections / stride)
    """
    ob_mean, D0 = load_ob(cfg)
    n_proj = cfg["n_projections"]
    stride = cfg.get("stride", 1)
    proj_indices = list(range(0, n_proj, stride))
    n_out = len(proj_indices)
    n_workers = min(cpu_count(), n_out)
    n_pixels = cfg["n_pixels"]
    n_slices = cfg["n_slices"]

    if stride > 1:
        print(f"Stride={stride}: using {n_out}/{n_proj} projections.")
    print(f"Processing {n_out} projections with {n_workers} workers ...")

    # Pre-allocate output so we never hold all results in a list simultaneously
    sinograms = np.empty((n_out, n_pixels, n_slices), dtype=np.float32)
    log_every = max(1, n_out // 20)  # print ~20 progress updates
    t0 = time.time()

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(ob_mean, D0, cfg),
    ) as pool:
        for done, result in enumerate(pool.imap(_process_projection, proj_indices)):
            sinograms[done] = result
            if (done + 1) % log_every == 0 or done + 1 == n_out:
                elapsed = time.time() - t0
                rate = (done + 1) / elapsed
                eta = (n_out - done - 1) / rate
                print(
                    f"  [{done + 1:>{len(str(n_out))}}/{n_out}] "
                    f"{elapsed:6.0f}s elapsed, ETA {eta:5.0f}s  ({rate:.2f} proj/s)",
                    flush=True,
                )

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
    sinograms = preprocess(cfg)
    save_sinograms(cfg, sinograms)
