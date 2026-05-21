#!/bin/bash
#BSUB -q hpc
#BSUB -J neutron_preproc
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8192]"
#BSUB -oo logs/neutron_preproc_%J.out
#BSUB -eo logs/neutron_preproc_%J.err
#BSUB -W 12:00

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil
python neutron/preprocessor.py --config neutron/reconstruction_settings.yaml
