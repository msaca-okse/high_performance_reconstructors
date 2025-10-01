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


def run_paganin_in_batches(cfg: dict, rearranged_data: np.ndarray, batch_size: int = 100):
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
    slice_array, slice_idx, folder_dir = args
    slice_path = folder_dir / f"slice_{slice_idx:04d}.tif"
    tiff.imwrite(slice_path, slice_array.astype(np.float32))
    return slice_path


def save_volume_as_tiffs(cfg: dict, folder: int, volume: np.ndarray):
    """
    Save a 3D numpy volume as individual TIFF slices along axis=0 using multiprocessing.
    Args:
        cfg (dict): configuration dictionary
        folder (int): folder number
        volume (np.ndarray): 3D array (N_slices, H, W)
    """
    scratch_root = Path(cfg["scratch_root"])
    session_name = cfg["session_name"]

    # Create session subfolder
    session_dir = scratch_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create folder-specific subfolder
    folder_name = f"HA900_3um_mars_rock_{folder:03d}_"
    folder_dir = session_dir / folder_name
    folder_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Folder {folder}] Saving {volume.shape[0]} slices to {folder_dir} with {cpu_count()} workers...")

    # Prepare arguments for pool
    args = [(volume[i, :, :], i, folder_dir) for i in range(volume.shape[0])]

    with Pool(processes=cpu_count()) as pool:
        pool.map(_save_slice, args)

    print(f"[Folder {folder}] Done saving {volume.shape[0]} slices.")





























def folder_processor(i):
    i = i-1
    p_pre1 = '/dtu-compute/msaca/sliceA_xray_pc/HA900_3um_mars_rock_00'
    p_pre2 = '_/HA900_3um_mars_rock_00'
    q_pre1 = '/dtu-compute/msaca/sliceA_xray_pc/compressed_XA_00'
    q_pre2 = '_/compressed_XA_00'

    p_post1 = '_####.edf'
    q_post1 = '_####.tiff'

    q_post_dark = '_/dark/dark.tiff'
    q_post_dark2 = '_/dark/darkend0000.tiff'
    q_post_obeam = '_/obeam/refHST6000.tiff'

    p_post_dark = '_/dark.edf'
    p_post_dark2 = '_/darkend0000.edf'
    p_post_obeam = '_/refHST6000.edf'

    stride = 1
    A = range(1,6001, stride)
    angles = np.linspace(0,360,len(A))
    folders = [1,2,3,4,5,6,7]
    start_crop_z = 620
    end_crop_z = 1410
    start_crop_x = 300
    end_crop_x = 2048
    size_z = end_crop_z - start_crop_z
    size_x = end_crop_x - start_crop_x

    edf_path_p = p_pre1 + str(folders[i]) + p_pre2 + str(folders[i]) + p_post1
    tiff_path_q = q_pre1 + str(folders[i]) + q_pre2 + str(folders[i]) + q_post1
    paths = generate_paths(edf_path_p, A)
    tiff_paths = generate_paths(tiff_path_q, A)
    for idx in range(len(paths)):
        image = read_edf(paths[idx])
        image = np.squeeze(image)
        raw_data = image[start_crop_z:end_crop_z,start_crop_x:end_crop_x]
        tifffile.imwrite(tiff_paths[idx], raw_data)

    edf_path_dark = p_pre1 + str(folders[i]) + p_post_dark
    edf_path_obeam = p_pre1 + str(folders[i]) + p_post_obeam

    tiff_path_dark = q_pre1 + str(folders[i]) + q_post_dark
    tiff_path_obeam = q_pre1 + str(folders[i]) + q_post_obeam


    image_dark = read_edf(edf_path_dark)
    image_obeam = read_edf(edf_path_obeam)


    image_dark = np.squeeze(image_dark)
    image_obeam = np.squeeze(image_obeam)
    raw_dark = image_dark[start_crop_z:end_crop_z,start_crop_x:end_crop_x]
    raw_obeam = image_obeam[start_crop_z:end_crop_z,start_crop_x:end_crop_x]

    tifffile.imwrite(tiff_path_dark, raw_dark)
    tifffile.imwrite(tiff_path_obeam, raw_obeam)






def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess(cfg)