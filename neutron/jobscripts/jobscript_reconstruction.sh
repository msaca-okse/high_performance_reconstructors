#!/bin/bash
#BSUB -q gpuv100
#BSUB -J neutron_recon
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8192]"
#BSUB -oo logs/neutron_recon_%J.out
#BSUB -eo logs/neutron_recon_%J.err
#BSUB -W 12:00

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil
python neutron/reconstructor.py --config neutron/reconstruction_settings.yaml
