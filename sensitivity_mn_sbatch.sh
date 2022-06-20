#!/bin/bash

# Request an hour of runtime:

#SBATCH -n 24
##SBATCH --mem=40g --ntasks-per-node=16 -N 25
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -A carney-sjones-condo

# Specify a job name:
#SBATCH -J clock_sim


# module load python/3.7.4 mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2
# source ~/envs/hnn_core_env/bin/activate

# Run a command
python3 ~/sensitivity_mn.py

# deactivate





###############################################
#export OMPI_MCA_btl_openib_allow_ib=1
#export OMPI_MCA_mpi_warn_on_fork=0
#export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_rmaps_base_mapping_policy="node"
#export OMPI_MCA_btl_base_verbose=100
#export OMPI_MCA_rmaps_base_schedule_local=0
#export OMPI_MCA_rmaps_base_n_pernode=1
#export OMPI_MCA_mpi_yield_when_idle=0
#export OMPI_MCA_rmaps_base_oversubscribe=1
#export OMPI_MCA_rmaps_base_inherit=0
#export OMPI_MCA_btl="tcp,vader"
#export OMPI_MCA_btl="tcp"


# for openmpi-4.x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ompi/build/lib
export PATH=$PATH:$HOME/ompi/build/bin
#export UCX_LOG_LEVEL=DEBUG
#export UCX_NET_DEVICES=mlx4_0:1
#export OMPI_MCA_pml="^ucx"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^tcp,vader,openib"
#export OMPI_MCA_btl="^uct,openib"
#export OMPI_MCA_btl="^vader,openib,uct,self"
export OMPI_MCA_routed="direct"

export PARAMS_FNAME="$1"
export EXP_FNAME="$2"
#export INPUT_NAME_1="$3"
export INPUT_NAME_1="evprox_1"
export INPUT_NAME_2="evdist_1"
export INPUT_NAME_3="evprox_2"
#export INCLUDE_WEIGHTS="timing_and_weights"
export INCLUDE_WEIGHTS="timing_only"

mpiexec -np 26 --oversubscribe python3 -m mpi4py examples/run_uncertainty_analysis_mpi.py

