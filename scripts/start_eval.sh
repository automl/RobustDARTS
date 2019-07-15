#!/bin/bash

# start only for 6 archs. these are potentialy erroneous due to the bug
for i in 449461 449464 449467 449470
do
    sbatch DARTS_cnn_eval.sh $i
    echo submmited job $i
done
