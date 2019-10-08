#!/bin/bash

datasets="cifar10 cifar100 svhn"
spaces="s1 s2 s3 s4"
wdecay=$(awk 'BEGIN{for(i=0.0003;i<=0.0243;i*=3)print i}')
dpath=$(awk 'BEGIN{for(j=0.0;j<=0.6;j+=0.2)print j}')

for d in $datasets; do
	for s in $spaces; do
		for dp in $dpath; do
			for wd in $wdecay; do
				sbatch scripts/RandomNAS.sh $s $d $dp $wd
				echo submmited job $s $d $dp $wd
			done
		done
	done
done


