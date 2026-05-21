import sys
import os
import argparse
import yaml
import time
import numpy as np
import tifffile as tiff
from pathlib import Path
import gc

# Ensure sibling modules in x_ray/ are importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessor import save_volume_as_tiffs

from multiprocessing import Pool, cpu_count
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry
from cil.plugins.astra import FBP



def _load_slice(slice_path):
    """Helper for parallel loading of slices."""
    return tiff.imread(slice_path).astype(np.float32)


def load_volume_from_tiffs(cfg: dict, folder: int, mode: str = "preprocess") -> np.ndarray:
    """
    Load a 3D numpy volume from a set of TIFF slices using multiprocessing.
    
    Args:
        cfg (dict): configuration dictionary
        folder (int): folder number
        mode (str): "preprocess" → load sinograms from scratch_root
                    "reconstruction" → load slices from results_root
    
    Returns:
        np.ndarray: 3D array (N_slices, H, W)
    """
    if mode not in ["preprocess", "reconstruction"]:
        raise ValueError("mode must be either 'preprocess' or 'reconstruction'")

    # Select base root and prefix based on mode
    session_name = cfg["session_name"]

    # Select base root and prefix based on mode
    if mode == "preprocess":
        base_root = Path(cfg["scratch_root"])
        session_dir = base_root
        prefix = "sinogram"
    else:  # reconstruction
        base_root = Path(cfg["results_root"])
        session_dir = base_root / session_name
        prefix = "slice"


    # Create folder-specific subfolder
    folder_name = f"HA900_3um_mars_rock_{folder:03d}_"
    folder_dir = session_dir / folder_name

    if not folder_dir.exists():
        raise FileNotFoundError(f"Folder {folder_dir} not found")

    # Collect and sort slice files
    slice_files = sorted(folder_dir.glob(f"{prefix}_*.tiff"))
    if not slice_files:
        raise FileNotFoundError(f"No {prefix}_*.tiff files found in {folder_dir}")

    print(f"[Folder {folder}] Loading {len(slice_files)} {mode} files from {folder_dir} with {cpu_count()} workers...")

    # Parallel load
    with Pool(processes=cpu_count()) as pool:
        slices = pool.map(_load_slice, slice_files)

    # Stack into 3D array
    volume = np.stack(slices, axis=0).astype(np.float32)

    print(f"[Folder {folder}] Loaded volume with shape {volume.shape}")
    return volume



def split_for_gpu(cfg: dict, sino: np.ndarray):
    """
    Split sino into smaller batches that fit into 1/3 of GPU memory.

    Args:
        cfg (dict): configuration with key 'gpu_size' in GB
        sino (np.ndarray): input sinogram (3D float32 array)

    Returns:
        list of np.ndarray: list of sub-arrays
    """
    gpu_size_gb = cfg["gpu_size"]
    safe_bytes = (gpu_size_gb * 1e9) / 3   # safe memory budget in bytes

    sino_bytes = sino.nbytes
    if sino_bytes <= safe_bytes:
        print(f"[INFO] Sinogram size {sino_bytes/1e9:.2f} GB fits into 1/3 GPU memory ({safe_bytes/1e9:.2f} GB). No split needed.")
        return [sino]  # fits already

    # memory per slice (axis 0)
    slice_bytes = sino[0].nbytes
    slices_per_chunk = int(safe_bytes // slice_bytes)


    if slices_per_chunk < 1:
        raise MemoryError(
            f"Single slice {slice_bytes/1e6:.2f} MB is larger than 1/3 GPU memory."
        )

    N_slices = sino.shape[0]

    print('Splitted sinogram of size ')

    chunks = [
        sino[i:i+slices_per_chunk]
        for i in range(0, N_slices, slices_per_chunk)
    ]

    max_chunk_bytes = max(chunk.nbytes for chunk in chunks)

    print(
        f"[INFO] Sinogram size: {sino_bytes/1e9:.2f} GB. "
        f"Split into {len(chunks)} chunks. "
        f"Largest chunk size: {max_chunk_bytes/1e9:.2f} GB "
        f"(limit {safe_bytes/1e9:.2f} GB)."
    )

    return chunks



def reconstruct_fbp_batch(cfg: dict, data: np.ndarray) -> np.ndarray:
    """
    Perform FBP reconstruction using CIL and ASTRA.

    Args:
        cfg (dict): configuration dictionary
        data (np.ndarray): input array with shape (N_slices, N_angles, N_pixels)

    Returns:
        np.ndarray: reconstructed volume, float32
    """
    # Extract dimensions
    N_slices, N_angles, N_pixels = data.shape

    # Read reconstruction parameters from cfg
    recon_cfg = cfg["reconstruction"]["geometry"]
    voxel_num_x = recon_cfg["voxel_num_x"]
    voxel_num_y = recon_cfg["voxel_num_y"]
    initial_angle = recon_cfg["initial_angle"]
    true_pixel_size = cfg["pixel_size"]

    # Angles from config
    angles_cfg = recon_cfg["angles"]
    angles = np.linspace(
        angles_cfg["start"], 
        angles_cfg["end"], 
        N_angles
    )

    # Acquisition geometry
    ag_batch = AcquisitionGeometry.create_Parallel3D(
        detector_position=[0, N_pixels // 2, 0]
    ).set_angles(angles).set_panel(
        (N_pixels, N_slices), pixel_size=(1, 1)
    ).set_labels(labels=('vertical', 'angle', 'horizontal'))

    # Data container
    data_batch = AcquisitionData(data, geometry=ag_batch)
    data_batch.reorder('astra')

    # Apply initial angle
    ag_batch.set_angles(ag_batch.angles, initial_angle=initial_angle)

    # Image geometry
    ig_batch = ImageGeometry(
        voxel_num_x=voxel_num_x,
        voxel_num_y=voxel_num_y,
        voxel_num_z=N_slices,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1
    )

    # Run FBP on GPU
    fbp = FBP(ig_batch, ag_batch, device='gpu')
    recon_slice_FBP = fbp(data_batch).as_array().astype(np.float32)*1e4/true_pixel_size #convert to 1/cm

    return recon_slice_FBP



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Preprocessing for reconstruction")
    parser.add_argument("--config", type=str, help="Configuration file path")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    folders = cfg["folders_to_preprocess"]

    for folder in folders:

        sino = load_volume_from_tiffs(cfg, folder, mode="preprocess") # Instead of loading all slices here, one can also just load them in inside the loop
        sino_batches = split_for_gpu(cfg, sino)
        del sino
        gc.collect()

        N_batches = len(sino_batches)

        output = []
        for idx_batch in range(N_batches):
            output.append(reconstruct_fbp_batch(cfg, sino_batches[idx_batch])) # units are in 1/cm

        output = np.concatenate(output, axis=0)
        save_volume_as_tiffs(cfg, folder, output, mode="reconstruction")
