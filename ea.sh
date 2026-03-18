#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=ea
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source env/bin/activate
srun python main_ea.py

