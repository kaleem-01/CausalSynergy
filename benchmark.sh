#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=paramsweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=pc_ges.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source env/bin/activate
srun python main.py

