#!/bin/bash
#OAR -n atiwari
#OAR -l walltime=12:0:0

source gpu_setVisibleDevices.sh
source "/scratch/clear/atiwari/miniconda3/etc/profile.d/conda.sh"
echo "Initialized miniconda"

conda activate grabnet2
echo "Activated conda environment"
cd /home/atiwari/projects/grabnet/
echo Running Code ...

python train.py --work-dir logs/V03_our_embeds --rhm-path ./assets/MANO_RIGHT.pkl --data-path /scratch/clear/atiwari/datasets/grabnet_extract/data/
