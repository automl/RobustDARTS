#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime
python src/evaluation/train.py --cutout --auxiliary --seed 1 --space s1 --dataset cifar10 --search_dp 0.0 --search_wd 0.0009


