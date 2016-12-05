#!/bin/bash
#SBATCH -J verb_tensors
#SBATCH -o vten_output_%j
##SBATCH --mem-per-cpu=3GB
#SBATCH -t 10:00:00
#SBATCH -N 2 
#SBATCH --ntasks-per-node=24

module load python/2.7.12/b2
module load openmpi/2.0.1/b2
source /home/ebigelow/.virtualenvs/np/bin/activate

mpirun SGD_parallel.py