#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=ea-fes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=08:00:00
#SBATCH --output=ea-fes.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source env/bin/activate
srun python main_ea-fes.py

