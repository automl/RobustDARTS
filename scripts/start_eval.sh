#!/bin/bash

# 2187
datasets="cifar10 cifar100 svhn"
spaces="s1 s2 s3 s4"
wdecay=$(awk 'BEGIN{for(i=0.0009;i<=0.2187;i*=3)print i}')
dpath=$(awk 'BEGIN{for(j=0.2;j<=0.8;j+=0.2)print j}')

for dp in $dpath; do
	for wd in $wdecay; do
		sbatch scripts/DARTS_eval.sh s1 cifar10 $dp $wd
		echo submmited job s1 cifar10 $dp $wd
	done
done

for dp in $dpath; do
	for wd in $wdecay; do
		sbatch scripts/DARTS_eval.sh s3 svhn $dp $wd
		echo submmited job s3 svhn $dp $wd
	done
done

