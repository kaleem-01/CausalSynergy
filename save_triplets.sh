#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=paramsweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:20:00
#SBATCH --output=stdout/save_triplets.out

module purge
module load 2025
module load Anaconda3/2025.06-1

module load MPICH/4.3.0-GCC-14.2.0
source env/bin/activate
srun python -m synthetic.save_triplets

