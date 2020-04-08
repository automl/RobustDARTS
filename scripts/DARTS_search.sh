#!/bin/bash
#
# submit to the right queue
#SBATCH --gres gpu:1
#SBATCH -a 1-3
#SBATCH -J DARTS_grid
#SBATCH -D .
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime
python src/search/train_search.py --unrolled --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID --train_portion 0.1 --cutout --report_freq_hessian 2 --space $1 --dataset $2 --drop_path_prob $3 --weight_decay $4

