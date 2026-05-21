#!/bin/bash
#BSUB -q gpuv100
#BSUB -J xray_recon
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8192]"
#BSUB -oo logs/xray_recon_%J.out
#BSUB -eo logs/xray_recon_%J.err
#BSUB -W 24:00

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil
python x_ray/reconstructor.py --config x_ray/reconstruction_settings_1.yaml
