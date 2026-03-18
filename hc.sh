#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=hc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source env/bin/activate
srun python main_hc.py

