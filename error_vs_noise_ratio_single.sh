#!/bin/bash
#SBATCH --job-name="greenlearning error"
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH -o %j.out
#SBATCH -e %j.err

source /home/users/tc656/.bashrc

eval "$(conda shell.bash hook)"

conda activate greenlearning

python error_vs_noise_ratio.py