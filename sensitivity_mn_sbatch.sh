#!/bin/bash

#SBATCH --job-name=run_simjob --output=run_simjob.out --error=run_simjob.err
#SBATCH -n 16
##SBATCH --mem=40g --ntasks-per-node=16 -N 32
#SBATCH -m 32g
#SBATCH -t 5:00:00
#SBATCH -A carney-sjones-condo


# module load python/3.7.4 mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2
# source ~/envs/hnn_core_env/bin/activate

# Run a command
python3 sensitivity_mn.py

# deactivate
