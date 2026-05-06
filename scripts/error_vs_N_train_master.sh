#!/bin/bash

for N in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 55 60 65 70 75;
do
    export N_TRAIN=$N
    sbatch error_vs_N_train_single.sh
done