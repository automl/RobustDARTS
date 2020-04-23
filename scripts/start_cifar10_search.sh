#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
#datasets="cifar10"
##spaces="s1 s2 s3 s4"
#spaces="s1"
#wdecay=$(awk 'BEGIN{for(i=0.0003;i<=0.0243;i*=7)print i}')
#dpath=$(awk 'BEGIN{for(j=0.0;j<=0.6;j+=0.2)print j}')
#
#
#for d in $datasets; do
#  for s in $spaces; do
#		sbatch scripts/DARTS_search.sh $s $d 0.0121 0.4
#		echo submmited job $s $d 0.0121 0.4
#  done
#done

#sbatch scripts/DARTS_search.sh s1 cifar10 0.0121 0.4
#echo submmited job s1 cifar10 0.0121 0.4

#!/bin/bash
#
# submit to the right queue
#SBATCH --gres gpu:1
#SBATCH -a 1-3
#SBATCH -J DARTS_grid
#SBATCH -D .
#python src/search/train_search.py --unrolled --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID --train_portion 0.1 --cutout --report_freq_hessian 2 --space s1 --dataset cifar10 --drop_path_prob 0.2 --weight_decay 0.0121
python src/search/train_search.py --unrolled --cutout --report_freq_hessian 2 --space s1 --dataset cifar10 --drop_path_prob 0.0 --weight_decay 0.0009

