#!/bin/bash
#SBATCH -J verb_tensors_TEST
#SBATCH -o vten_output_%j
#SBATCH -p standard
#SBATCH -t 1:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=20

module load python/2.7.12/b2
module load openmpi/2.0.1/b2
source /home/ebigelow/.virtualenvs/np/bin/activate

#mpirun -np 48 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py
mpirun -np 40 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py
