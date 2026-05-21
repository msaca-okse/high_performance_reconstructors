

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstructor import load_volume_from_tiffs



def correct_volume(cfg: dict) -> np.ndarray:
    return None


def stitch_reconstructions(cfg: dict) -> np.ndarray:
    """
    Stitch reconstructed slices from multiple folders into one volume.
    Uses linear blending across overlap regions.

    Args:
        cfg (dict): configuration with "stitching" section

    Returns:
        np.ndarray: stitched volume, shape (total_slices, H, W)
    """
    start_indices = cfg["stitching"]["start_indices"]
    end_indices = cfg["stitching"]["end_indices"]
    overlap = cfg["stitching"]["overlap_pixels"]
    folders = cfg["folders_to_preprocess"]

    stitched_slices = []

    for i, folder in enumerate(folders):
        print(f"[INFO] Loading reconstruction for folder {folder}")
        vol = load_volume_from_tiffs(cfg, folder, mode="reconstruction")  # (N, H, W)

        start = start_indices[i]
        end = end_indices[i]
        subvol = vol[start:end]  # slice range

        if i == 0:
            # First volume → take everything directly
            stitched_slices.append(subvol)
        else:
            # Blend with previous volume
            prev_vol = stitched_slices[-1]

            # Split off the overlap
            overlap_prev = prev_vol[-overlap:]
            overlap_curr = subvol[:overlap]

            # Linear weights: 1 → 0 for previous, 0 → 1 for current
            weights = np.linspace(0, 1, overlap, endpoint=False)[:, None, None]

            blended = (1 - weights) * overlap_prev + weights * overlap_curr

            # Replace overlap in previous
            stitched_slices[-1] = prev_vol[:-overlap]
            # Add blended + remainder of current
            stitched_slices.append(blended)
            stitched_slices.append(subvol[overlap:])

    # Stack into final array
    stitched_volume = np.concatenate(stitched_slices, axis=0).astype(np.float32)
    print(f"[INFO] Final stitched volume shape: {stitched_volume.shape}")
    return stitched_volume