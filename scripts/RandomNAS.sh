#!/bin/bash
#
# submit to the right queue
#SBATCH -p meta_gpu-ti
#SBATCH --gres gpu:1
#SBATCH -a 1-3
#SBATCH -J RandomNAS
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./experiments/cluster_logs/%A_%a.o
#SBATCH -e ./experiments/cluster_logs/%A_%a.e
#
#

source activate pytorch-0.3.1-cu8-py36
python src/search/randomNAS/random_weight_share.py --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID --epochs 50 --save experiments/search_logs_RandomNAS --space $1 --dataset $2 --drop_path_prob $3 --weight_decay $4


