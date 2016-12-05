#!/bin/bash
#SBATCH -J verb_tensors
#SBATCH -o vten_output_%j
#SBATCH -p standard
#SBATCH -t 1:00:00
#SBATCH -N 4 
#SBATCH --ntasks-per-node=22

module load python/2.7.12/b2
module load openmpi/2.0.1/b2
source /home/ebigelow/.virtualenvs/np/bin/activate

#mpirun -np 48 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py
mpirun -np 85 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py
