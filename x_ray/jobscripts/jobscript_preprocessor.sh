#!/bin/bash
#BSUB -q hpc
#BSUB -J xray_preproc
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8192]"
#BSUB -oo logs/xray_preproc_%J.out
#BSUB -eo logs/xray_preproc_%J.err
#BSUB -W 24:00

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil
python x_ray/preprocessor.py --config x_ray/reconstruction_settings_1.yaml
