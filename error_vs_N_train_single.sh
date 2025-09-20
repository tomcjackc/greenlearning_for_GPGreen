#!/bin/bash
#SBATCH --job-name="My test job"
#SBATCH --time=01:00:00
#SBATCH --mem=1M
#SBATCH -o %j.out
#SBATCH -e %j.err

python error_vs_N_train.py