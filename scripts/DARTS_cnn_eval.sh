#!/bin/bash
#
# submit to the right queue
#SBATCH -p meta_gpu-ti
#SBATCH --gres=gpu:1
#SBATCH -a 1-3
#SBATCH -J DARTS_C10_eval_noisy_%a
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D ./src
#
# redirect the output/error to some files
#SBATCH -o ../logs_eval/%A_%a.o
#SBATCH -e ../logs_eval/%A_%a.e
#

source activate pytorch-0.3.1-cu8
# set init channels 16 since a cell with all sep_convs does not fit memory with 36
python train.py --cutout --auxiliary --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID --arch DARTS_$1_$SLURM_ARRAY_TASK_ID --init_channels 16

