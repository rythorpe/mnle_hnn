#!/bin/bash

#SBATCH -J sensitivity_job
#SBATCH -n 24
##SBATCH --mem=40g --ntasks-per-node=16 -N 25
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -A carney-sjones-condo


# module load python/3.7.4 mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2
# source ~/envs/hnn_core_env/bin/activate

# Run a command
python3 sensitivity_mn.py

# deactivate
