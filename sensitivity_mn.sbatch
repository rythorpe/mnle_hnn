#!/bin/bash

#SBATCH -J run_simjob -o run_simjob.out -e run_simjob.err
##SBATCH --mem=40g --ntasks-per-node=16 -N 32
#SBATCH --mem=40g --ntasks-per-node=16 -N 3
#SBATCH -t 5:00:00
#SBATCH -A carney-sjones-condo

# load modules and activate python env
module load python/3.7.4 mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2
source $HOME/envs/hnn_core_env/bin/activate

export OMPI_MCA_rmaps_base_mapping_policy="node"

# run python script
#mpiexec -np 33 --oversubscribe python3 -m mpi4py $HOME/mnle_hnn/sensitivity_mn.py
mpiexec -np 4 --oversubscribe python3 -m mpi4py $HOME/mnle_hnn/sensitivity_mn.py

# deactivate python env
deactivate
