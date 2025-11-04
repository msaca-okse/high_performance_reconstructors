#!/bin/bash
#BSUB -q gpuv100
#BSUB -J preproc_job
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8192]"
#BSUB -oo logs/preproc_%J.out
#BSUB -eo logs/preproc_%J.err
#BSUB -W 24:00                 # Wall time (e.g., 1 hour)


source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil
python reconstructor.py --config reconstruction_settings_1.yaml

