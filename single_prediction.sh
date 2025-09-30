#!/bin/bash
#SBATCH --job-name="greenlearning single prediction"
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH -o alleq_error_vs_N_train_logs/%j.out
#SBATCH -e alleq_error_vs_N_train_logs/%j.err

for E in "boundary_layer";
do
    export EQUATION=$E

    for N in 75;
    do
    export N_TRAIN=$N
    source /home/users/tc656/.bashrc

    eval "$(conda shell.bash hook)"

    conda activate greenlearning

    python single_prediction.py
    done
done