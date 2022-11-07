#!/bin/bash

#SBATCH -J sweep_mn_job -o sweep_mn_job.out -e sweep_mn_job.err
#SBATCH --mem=40g --ntasks-per-node=24 -N 1
##SBATCH --mem=128g -n 100
#SBATCH -t 6:00:00
#SBATCH -A carney-sjones-condo

# load modules and activate python env
module load python/3.7.4 mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2
source $HOME/envs/hnn_core_env_stable/bin/activate

export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^tcp,vader,openib"
export OMPI_MCA_routed="direct"

# run python script
python sweep_mn.py

# deactivate python env
deactivate