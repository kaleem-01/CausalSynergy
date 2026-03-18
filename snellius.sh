#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=paramsweep
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --output=paramsweep.out
module purge
module load 2025
module load Anaconda3/2025.06-1

source env/bin/activate
srun python param_sweep.py
