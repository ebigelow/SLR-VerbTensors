#!/bin/bash
#SBATCH -J verb_tensors
#SBATCH -o data/logs/vten_output_%j
#SBATCH -p standard
###SBATCH --mem-per-cpu=1gb
#SBATCH -t 100:00:00
#SBATCH -N 6
#SBATCH --ntasks-per-node=20

module load python/2.7.12/b2
module load openmpi/2.0.1/b2
source /home/ebigelow/.virtualenvs/np/bin/activate

#mpirun -np 48 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py
mpirun -n 120 python /home/ebigelow/lib/verb-tensors/SGD_parallel.py $SLURM_JOB_ID
