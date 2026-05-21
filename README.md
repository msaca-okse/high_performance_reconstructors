# Neutron and X-ray CT Reconstruction

Reproducibility code for the paper. Two independent CT reconstruction pipelines
(neutron and x-ray) followed by a joint registration step.

## Repository structure

```
x_ray/          X-ray phase-contrast CT pipeline (preprocessing, FBP reconstruction, stitching)
neutron/        Neutron CT pipeline (preprocessing, TV-FISTA reconstruction)
registration/   Registration of x-ray and neutron volumes; export to OME-Zarr
logs/           HPC job logs (not tracked in git)
```

## Requirements

```bash
conda activate cil   # CIL + ASTRA + ccpi-regulariser environment
# Additional packages:
pip install astropy tifffile ome-zarr simpleitk
```

## X-ray pipeline

1. **Preprocess** — EDF projection loading, flat/dark correction, Paganin phase retrieval,
   save sinogram TIFFs (`x_ray/preprocessor.py`).
2. **Reconstruct** — FBP on GPU (`x_ray/reconstructor.py`).
3. **Stitch** — blend 7 overlapping sub-volumes (`x_ray/postprocessor.py`).

Configuration: `x_ray/reconstruction_settings_1.yaml` (update HPC paths as needed).

```bash
# From repo root — submits preprocessing then reconstruction as a dependent job pair
bash x_ray/jobscripts/submit_all.sh
```

## Neutron pipeline

1. **Preprocess** — FITS loading, open-beam correction, dose normalisation, spot cleaning,
   save sinogram TIFFs (`neutron/preprocessor.py`).
2. **Reconstruct** — TV-regularised FISTA on GPU (`neutron/reconstructor.py`).

Configuration: `neutron/reconstruction_settings.yaml` — replace all `PLACEHOLDER_` values
with your HPC paths and the correct TV alpha before running.

```bash
# From repo root
bash neutron/jobscripts/submit_all.sh
```

## Registration

Open `registration/stitch_and_register.ipynb` to stitch x-ray sub-volumes and register
x-ray to neutron. The result is written as an OME-Zarr multiscale store (3 pyramid levels
for neutron, 4 for x-ray). See `registration/register_to_diffraction.ipynb` for
registration to neutron diffraction data.
