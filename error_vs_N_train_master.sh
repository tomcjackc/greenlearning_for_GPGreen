#!/bin/bash

export N_TRAIN=30

sbatch -J "Ntrain_${N_train}" error_vs_N_train_single.sh
# sh error_vs_N_train_single.sh