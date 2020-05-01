#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
python src/search/train_search.py --unrolled --cutout --report_freq_hessian 2 --space s1 --dataset malaria --drop_path_prob 0.0 --weight_decay 0.0009

