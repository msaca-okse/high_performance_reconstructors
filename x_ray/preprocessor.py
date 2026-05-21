import argparse
import yaml
import numpy as np
from pathlib import Path
from dxchange.reader import read_edf
import tifffile as tiff
from multiprocessing import Pool, cpu_count
from cil.framework import AcquisitionGeometry, AcquisitionData, ImageGeometry
from cil.processors import PaganinProcessor
import os
import argparse
import gc


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file into a dict."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_angles(cfg: dict) -> np.ndarray:
    """Construct the angle array from config."""
    return np.linspace(
        cfg["angles"]["start"],
        cfg["angles"]["end"],
        cfg["angles"]["n_projections"]
    )


def generate_paths(path, A):
    # Extract the folder and filename pattern
    folder, filename_pattern = os.path.split(path)
    
    # Determine the number of digits in the zero-padding
    num_digits = filename_pattern.count('#')
    filename_base = filename_pattern.replace('#' * num_digits, '{}')

    # Generate the list of paths
    paths = [
        os.path.join(folder, filename_base.format(str(num).zfill(num_digits)))
        for num in A
    ]
    
    return paths

def load_and_crop(file_path: Path, crop: dict) -> np.ndarray:
    """Load EDF file and apply cropping."""
    arr = read_edf(file_path).squeeze()
    return arr[
        crop["start_z"]:crop["end_z"],
        crop["start_x"]:crop["end_x"]
    ]


def load_projection(args):
    """Helper for multiprocessing (unpacks arguments)."""
    proj_file, crop = args
    return load_and_crop(proj_file, crop)


def build_filename(pattern: str, folder: int, proj_num: int | None = None) -> Path:
    """Replace ### with folder number and #### with projection number if given."""
    fname = pattern
    if proj_num is not None:
        fname = fname.replace("####", f"{proj_num:04d}")
    fname = fname.replace("###", f"{folder:03d}")
    return fname




def load_folder(cfg: dict, folder: int):
    """
    Load projections, dark, and flat for a single folder.
    Returns: projections (ndarray), raw_flat (ndarray), raw_dark (ndarray).
    """
    data_root = Path(cfg["data_root"])
    crop = cfg["crop"]
    stride = cfg["stride"]

    proj_pattern = cfg["filenames"]["projection"]
    flat_pattern = cfg["filenames"]["flat"]
    dark_pattern = cfg["filenames"]["dark"]

    # Load dark + flat
    dark_file = data_root / build_filename(dark_pattern, folder)
    flat_file = data_root / build_filename(flat_pattern, folder)
    raw_dark = load_and_crop(dark_file, crop)
    raw_flat = load_and_crop(flat_file, crop)
    print(f"[Folder {folder}] Loaded dark {raw_dark.shape}, flat {raw_flat.shape}")

    # Load projections
    n_projs = cfg["angles"]["n_projections"]
    proj_files = [
        data_root / build_filename(proj_pattern, folder, proj_num)
        for proj_num in range(0, n_projs, stride)
    ]

    print(f"[Folder {folder}] Loading {len(proj_files)} projections with {cpu_count()} workers...")

    with Pool(processes=cpu_count()) as pool:
        proj_list = pool.map(load_projection, [(pf, crop) for pf in proj_files])

    projections = np.stack(proj_list, axis=0)  # shape: (N, z, x)
    print(f"[Folder {folder}] Loaded projections: {projections.shape}")

    return projections, raw_flat, raw_dark





def flat_dark_correction(cfg: dict, projs, flat, dark):
    data = np.array(-np.log((projs - dark[np.newaxis,:,:])/(flat[np.newaxis,:,:] - dark[np.newaxis,:,:])),dtype=np.float32)
    return data


def rearrange_data(cfg: dict, data):
    """
    Rearrange a 3D block matrix A into the format:
    [A11, mean(A12, flip(A22)), flip(A21)]
    
    Parameters:
        A (numpy.ndarray): Input 3D array with shape (rows, depth, columns).
        split_row (int): Index along axis=0 to split the rows into two blocks.
        split_col (int): Index along axis=2 to split the columns into two blocks.
        
    Returns:
        numpy.ndarray: Rearranged 3D array.
    """
    # Step 1: Split along rows (axis=0) and columns (axis=2)
    N_angles, N_slices, N_pixels = np.shape(data)
    axis_placement = cfg["axis_placement"]
    split_col = N_pixels-axis_placement
    split_row = int(N_angles/2)

    A11 = data[:split_row, :, :split_col]                # Top-left block
    A12 = data[:split_row, :, split_col:]                # Top-right block
    A21 = data[split_row:, :, :split_col]                # Bottom-left block
    A22 = data[split_row:, :, split_col:]                # Bottom-right block

    # Step 2: Flip A22 and A21 along axis=2 (reverse columns)
    A22_flipped = A22[:, :, ::-1]                     # Flip columns of A22
    A21_flipped = A21[:, :, ::-1]                     # Flip columns of A21

    # Step 3: Compute mean of A12 and flipped A22
    mean_block = (A12 + A22_flipped) / 2

    # Step 4: Concatenate blocks along axis=2 (columns)
    rearranged_data = np.concatenate((A11, mean_block, A21_flipped), axis=2)

    return rearranged_data



def paganin_batch(cfg: dict, data):
    # Multiprocessing should be run outside this function
    pad = cfg["paganin"]["pad"]
    alpha = cfg["paganin"]["alpha"]
    N_angles, N_slices, N_pixels = np.shape(data)
    angles = np.zeros(N_angles)

    ag = AcquisitionGeometry.create_Parallel3D(detector_position=[0,N_slices,0])\
                            .set_angles(angles)\
                            .set_panel((N_pixels,N_slices), pixel_size=(1,1))\
                            .set_labels(labels=('angle','vertical','horizontal'))
    data_batch = AcquisitionData(data, geometry=ag)
    data_batch.reorder('cil')
    data_batch.geometry.config.units = 'mm'
    processor = PaganinProcessor(full_retrieval=False,pad=pad)
    processor.set_input(data_batch)
    data_batch = processor.get_output(override_filter={'alpha':alpha})
    return data_batch.as_array()


def _paganin_worker(args):
    """Helper function for Pool.map"""
    cfg, batch = args
    return paganin_batch(cfg, batch)


def run_paganin_in_batches(cfg: dict, rearranged_data: np.ndarray, batch_size: int = 32):
    """
    Run Paganin preprocessing in parallel batches along axis=0.
    
    Args:
        cfg (dict): configuration dictionary
        rearranged_data (np.ndarray): input data, shape (N, H, W)
        batch_size (int): number of slices along axis=0 per batch
    
    Returns:
        np.ndarray: concatenated Paganin-processed data, same shape as input
    """
    N = rearranged_data.shape[0]

    # If smaller than batch_size, just run once
    if N <= batch_size:
        return paganin_batch(cfg, rearranged_data)

    # Split into batches along axis 0
    batches = [rearranged_data[i:i+batch_size] for i in range(0, N, batch_size)]

    print(f"Splitting data into {len(batches)} batches of up to {batch_size} slices each.")
    
    # Run in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_paganin_worker, [(cfg, b) for b in batches])

    # Concatenate back
    return np.concatenate(results, axis=0)



def _save_slice(args):
    """Helper for parallel saving of slices."""
    slice_array, slice_idx, folder_dir, prefix = args
    slice_path = folder_dir / f"{prefix}_{slice_idx:04d}.tiff"
    tiff.imwrite(slice_path, slice_array.astype(np.float32))
    return slice_path


def save_volume_as_tiffs(cfg: dict, folder: int, volume: np.ndarray, mode: str = "preprocess"):
    """
    Save a 3D numpy volume as individual TIFF slices along axis=0 using multiprocessing.
    
    Args:
        cfg (dict): configuration dictionary
        folder (int): folder number
        volume (np.ndarray): 3D array (N_slices, H, W)
        mode (str): "preprocess" → save sinograms to scratch_root
                    "reconstruction" → save slices to results_root
    """
    if mode not in ["preprocess", "reconstruction"]:
        raise ValueError("mode must be either 'preprocess' or 'reconstruction'")

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


    # Create session subfolder
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create folder-specific subfolder
    folder_name = f"HA900_3um_mars_rock_{folder:03d}_"
    folder_dir = session_dir / folder_name
    folder_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Folder {folder}] Saving {volume.shape[0]} {mode} files to {folder_dir} with {cpu_count()} workers...")

    # Prepare arguments for pool
    args = [(volume[i, :, :], i, folder_dir, prefix) for i in range(volume.shape[0])]

    with Pool(processes=cpu_count()) as pool:
        pool.map(_save_slice, args)

    print(f"[Folder {folder}] Done saving {volume.shape[0]} {mode} files.")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Preprocessing for reconstruction")
    parser.add_argument("--config", type=str, help="Configuration file path")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    folders = cfg["folders_to_preprocess"]

    for folder in folders:
        projs, flat, dark = load_folder(cfg, folder=folder)
        data = flat_dark_correction(cfg, projs, flat, dark)   #### Do flux correction

        # Free memory from raw inputs once not needed
        del projs, flat, dark
        gc.collect()

        rearranged_data = rearrange_data(cfg, data)
        del data
        gc.collect()

        p_data = run_paganin_in_batches(cfg, rearranged_data, batch_size = 32).transpose([1,0,2])
        del rearranged_data
        gc.collect()

        save_volume_as_tiffs(cfg, folder, p_data, mode="preprocess")
        del p_data
        gc.collect()