#!/bin/bash
#SBATCH --job-name="greenlearning train wall time vs N_train"
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

python train_wall_time_vs_N_train.py