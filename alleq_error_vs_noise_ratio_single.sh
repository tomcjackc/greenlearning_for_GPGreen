#!/bin/bash
#SBATCH --job-name="greenlearning all equations error vs noise_ratio"
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH -o alleq_error_vs_noise_ratio_logs/%j.out
#SBATCH -e alleq_error_vs_noise_ratio_logs/%j.err

source /home/users/tc656/.bashrc

eval "$(conda shell.bash hook)"

conda activate greenlearning

python alleq_error_vs_noise_ratio.py