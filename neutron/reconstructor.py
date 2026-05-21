"""
Neutron CT reconstructor.

Pipeline:
  Load sinogram TIFFs -> TV-regularised FISTA reconstruction (GPU) -> save slice TIFFs.

Usage:
  python neutron/reconstructor.py --config neutron/reconstruction_settings.yaml
"""

import argparse
import gc
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tifffile
import yaml

from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares
from cil.plugins.astra import FBP
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_sino(path: Path) -> np.ndarray:
    return tifffile.imread(str(path)).astype(np.float32)


def load_sinograms(cfg: dict) -> np.ndarray:
    """Load sinogram TIFFs from scratch_root that match the configured slice range.

    If n_slices is set in the config, only the sinograms for columns
    [slice_offset, slice_offset + n_slices) are loaded. If n_slices is null,
    all sinograms from slice_offset onwards are loaded.

    Returns
    -------
    volume : float32 array of shape (n_slices, n_angles, n_pixels)
    """
    scratch_root = Path(cfg["scratch_root"])
    slice_offset = cfg["slice_offset"]
    n_slices = cfg["n_slices"]  # may be None

    if n_slices is not None:
        # Build explicit list of expected filenames
        sino_files = [
            scratch_root / f"sinogram_{slice_offset + j:05d}.tiff"
            for j in range(n_slices)
        ]
        missing = [f for f in sino_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sinogram file(s) not found in {scratch_root}, "
                f"e.g. {missing[0].name}"
            )
    else:
        # Load all sinograms from slice_offset onwards
        all_files = sorted(scratch_root.glob("sinogram_*.tiff"))
        sino_files = [
            f for f in all_files
            if int(f.stem.split("_")[1]) >= slice_offset
        ]
        if not sino_files:
            raise FileNotFoundError(
                f"No sinogram_*.tiff files found in {scratch_root} "
                f"with index >= {slice_offset}"
            )

    n_workers = cpu_count()
    print(f"Loading {len(sino_files)} sinograms from {scratch_root} with {n_workers} workers ...")
    with Pool(processes=n_workers) as pool:
        slices = pool.map(_load_sino, sino_files)

    volume = np.stack(slices, axis=0)  # (n_slices, n_angles, n_pixels)
    print(f"Sinogram volume shape: {volume.shape}")
    return volume


# ---------------------------------------------------------------------------
# GPU chunking
# ---------------------------------------------------------------------------

def split_for_gpu(cfg: dict, data: np.ndarray) -> list:
    """Split along the slice axis so each chunk uses at most 1/3 of GPU memory."""
    safe_bytes = (cfg["gpu_size"] * 1e9) / 3
    if data.nbytes <= safe_bytes:
        print(f"Data ({data.nbytes / 1e9:.2f} GB) fits in GPU budget ({safe_bytes / 1e9:.2f} GB).")
        return [data]
    slices_per_chunk = max(1, int(safe_bytes // data[0].nbytes))
    chunks = [data[i: i + slices_per_chunk] for i in range(0, data.shape[0], slices_per_chunk)]
    print(f"Split into {len(chunks)} GPU chunks of up to {slices_per_chunk} slices each.")
    return chunks


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _build_geometry(cfg: dict, data: np.ndarray):
    """Return (AcquisitionData, AcquisitionGeometry, ImageGeometry) for a data chunk.

    If reconstruction.recon_roi is set, a Slicer is applied to the ImageGeometry
    to restrict the reconstructed FOV, reducing GPU memory and runtime.
    """
    from cil.processors import Slicer

    recon_cfg = cfg["reconstruction"]
    n_slices, n_angles, n_pixels = data.shape
    initial_angle = recon_cfg["initial_angle"]
    angles = np.linspace(0, 360, n_angles, endpoint=True, dtype=np.float32)

    ag = (
        AcquisitionGeometry.create_Parallel3D(detector_position=[0, n_pixels // 2, 0])
        .set_angles(angles)
        .set_panel((n_pixels, n_slices), pixel_size=(1, 1))
        .set_labels(('vertical', 'angle', 'horizontal'))
    )
    data_cil = AcquisitionData(data, geometry=ag)
    data_cil.reorder('astra')
    ag.set_angles(ag.angles, initial_angle=initial_angle)
    ig = ag.get_ImageGeometry()

    recon_roi = recon_cfg.get("recon_roi")
    if recon_roi:
        roi_dict = {}
        y_start = recon_roi.get("horizontal_y_start")
        y_end   = recon_roi.get("horizontal_y_end")
        x_start = recon_roi.get("horizontal_x_start")
        x_end   = recon_roi.get("horizontal_x_end")
        if y_start is not None or y_end is not None:
            roi_dict["horizontal_y"] = (y_start, y_end, 1)
        if x_start is not None or x_end is not None:
            roi_dict["horizontal_x"] = (x_start, x_end, 1)
        if roi_dict:
            slicer = Slicer(roi_dict)
            slicer.set_input(ig)
            ig = slicer.get_output()

    return data_cil, ag, ig


def reconstruct_fbp_batch(cfg: dict, data: np.ndarray) -> np.ndarray:
    """Fast FBP reconstruction — use for sanity checks before committing to TV.

    Parameters
    ----------
    data : float32 array of shape (n_slices, n_angles, n_pixels)

    Returns
    -------
    recon : float32 array of shape (n_slices, n_pixels, n_pixels)
    """
    data_cil, ag, ig = _build_geometry(cfg, data)
    fbp = FBP(ig, ag, device='gpu')
    return fbp(data_cil).as_array().astype(np.float32)


def reconstruct_tv_batch(cfg: dict, data: np.ndarray) -> np.ndarray:
    """TV-regularised FISTA reconstruction (GPU).

    Parameters
    ----------
    data : float32 array of shape (n_slices, n_angles, n_pixels)

    Returns
    -------
    recon : float32 array of shape (n_slices, n_pixels, n_pixels)
    """
    recon_cfg = cfg["reconstruction"]
    alpha = recon_cfg["alpha"]
    n_iter = recon_cfg["n_iterations"]

    data_cil, ag, ig = _build_geometry(cfg, data)
    A = ProjectionOperator(ig, ag, 'gpu')
    F = LeastSquares(A, data_cil)
    G = alpha * FGP_TV(device='gpu', nonnegativity=True)

    solver = FISTA(f=F, g=G, initial=ig.allocate(0))
    solver.run(n_iter, verbose=1)

    return solver.solution.copy().as_array().astype(np.float32)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_reconstruction(cfg: dict, volume: np.ndarray):
    """Save reconstructed volume as individual TIFF slices.

    Files are named slice_NNNNN.tiff, using the same absolute slice index
    as the corresponding input sinograms.
    """
    results_root = Path(cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    slice_offset = cfg["slice_offset"]
    n_slices = volume.shape[0]

    print(f"Saving {n_slices} slices to {results_root} ...")
    for i in range(n_slices):
        out_path = results_root / f"slice_{slice_offset + i:05d}.tiff"
        tifffile.imwrite(str(out_path), volume[i])
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neutron CT reconstruction: TV-regularised FISTA."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    method = cfg["reconstruction"].get("method", "tv").lower()
    if method not in ("tv", "fbp"):
        raise ValueError(f"reconstruction.method must be 'tv' or 'fbp', got '{method}'")
    print(f"Reconstruction method: {method.upper()}")

    reconstruct_batch = reconstruct_fbp_batch if method == "fbp" else reconstruct_tv_batch

    data = load_sinograms(cfg)
    chunks = split_for_gpu(cfg, data)
    del data
    gc.collect()

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\nReconstructing chunk {i + 1}/{len(chunks)} ...")
        results.append(reconstruct_batch(cfg, chunk))

    volume = np.concatenate(results, axis=0)
    del results, chunks
    gc.collect()

    save_reconstruction(cfg, volume)
