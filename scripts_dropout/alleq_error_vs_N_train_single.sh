#!/bin/bash
#SBATCH --job-name="dropout greenlearning all equations error vs N_train"
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH -o alleq_error_vs_N_train_logs/%j.out
#SBATCH -e alleq_error_vs_N_train_logs/%j.err

source /home/users/tc656/.bashrc

eval "$(conda shell.bash hook)"

conda activate greenlearning

python alleq_error_vs_N_train.py