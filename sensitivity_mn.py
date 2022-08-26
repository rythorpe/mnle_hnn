"""
Run sensitivity analysis on the HNN-core drive parameters underlying the median
nerve evoked response. Starting with an optimized set of drive parameters (n=4
drives), this routine will sample random parameter values to see how they
impact the current dipole evoked response generated by HNN-core's
jones_2009_model(). Uses Uncertainpy
(https://uncertainpy.readthedocs.io/en/latest/index.html) and is designed to
run on Oscar across multiple nodes.

Based on an MPI processing pipeline originally developed by
Blake Caldwell <1blakecaldwell@gmail.com>.
"""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

from os import environ
from mpi4py import MPI

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend)
from hnn_core.viz import plot_dipole


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def run_sim_across_node_cores(drive_params, n_procs, mpi_info):
    # set MPI variables
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # load original parameters
    params_fname = ('/users/rthorpe/data/rthorpe/hnn_out/param/'
                    'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')
    params = hnn_core.read_params(params_fname)

    # update parameters with the current values being explored
    params['t_evdist_1'] = drive_params['t_evdist_1']
    net = jones_2009_model(params, add_drives_from_params=True)

    # run simulation distributed over multiple cores
    t_start = MPI.Wtime()
    with MPIBackend(n_procs=n_procs, mpi_comm_spawn=True,
                    mpi_comm_spawn_info=mpi_info):
        dpls = simulate_dipole(net, tstop=170., n_trials=25)
    elapsed_time = MPI.Wtime() - t_start
    print(f'Completed simulation on rank {rank} in {elapsed_time}')

    # plot dpls & save
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    scaling_factor = 40
    smooth_win = 20
    dpls = [dpl.scale(scaling_factor).smooth(smooth_win) for dpl in dpls]
    avg_dpl = average_dipoles(dpls)
    plot_dipole(dpls, ax=axes[0], show=False)
    plot_dipole(avg_dpl, ax=axes[1], show=False)
    fig.savefig(f'dipoles_mn_{rank:02.0f}.png', dpi=300)

    return avg_dpl


def master_proc(comm, rank, status, processor_name):
    """Initialize master process for setting parameters and collecting data"""

    size = comm.Get_size()  # total number of inter-node processes
    if 'SLURM_NNODES' in environ:
        # main comm is oversubscribed: don't include master process
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1

    print(f'Master starting with rank {rank} on {processor_name}, '
          f'running workers on {n_nodes}')

    # worker proc parameters
    n_workers = n_tasks = n_nodes
    closed_workers = 0
    task_idx = 0

    # define parameters for each task
    new_params = np.linspace(75, 95, n_tasks)
    agg_results = list()
    #comm.bcast(new_params, root=MPI.ROOT)

    while closed_workers < n_workers:
        latest_results = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                   status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == tags.READY:
            # worker is ready, so send it a task
            if task_idx < n_tasks:
                print("Sending task %d to worker %d" % (task_idx, source))
                comm.send(new_params[task_idx], dest=source, tag=tags.START)
                task_idx += 1
            else:
                comm.isend(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            print("Got data from worker %d" % source)
            agg_results.append(latest_results)
        elif tag == tags.EXIT:
            print("Worker %d exited (%d running)" % (source, closed_workers))
            closed_workers += 1

    fig = plot_dipole(agg_results, show=False)
    fig.savefig('dipoles_mn_master.png', dpi=300)


def worker_proc(comm, rank, status, processor_name):
    """Initialize worker process for running simulations"""

    print("Worker started with rank %d on %s." % (rank, processor_name))

    # find the number of available cores for parallel processing
    if 'SLURM_CPUS_ON_NODE' in environ:
        # XXX minus 2???
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = 1

    # limit MPI to this host only
    mpi_info = MPI.Info().Create()
    mpi_info.Set('host', processor_name.split('.')[0])
    mpi_info.Set('ompi_param', 'rmaps_base_inherit=0')
    mpi_info.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpi_info.Set('ompi_param', 'rmaps_base_oversubscribe=1')

    # receive experimental data
    #(exp_data, params_input) = comm.bcast(comm.Get_rank(), root=0)

    # send params and exp_data to spawned nrniv procs
    #simdata = (exp_data, params_input)
    #subcomm.bcast(simdata, root=MPI.ROOT)

    avg_sim_times = list()

    #subcomm.Barrier()
    print("Worker %d waiting on master to signal start" % rank)
    # tell rank 0 we are ready
    comm.isend(None, dest=0, tag=tags.READY)

    while True:

        # Receive updated params (blocking)
        new_params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        if tag == tags.EXIT:
            print('worker %d on %s has received exit signal'%(rank, processor_name))
            break

        # run simulation on timer
        t_start = MPI.Wtime()
        new_results = run_sim_across_node_cores(new_params, n_procs, mpi_info)
        elapsed_time = MPI.Wtime() - t_start
        avg_sim_times.append(elapsed_time)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (processor_name, elapsed_time, mean(avg_sim_times)))

        # send results back
        comm.isend(new_results, dest=0, tag=tags.DONE)

        # tell rank 0 we are ready (again)
        comm.isend(None, dest=0, tag=tags.READY)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.Get_rank()  # rank of this process
    status = MPI.Status()  # get MPI status object
    processor_name = MPI.Get_processor_name()

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    if rank == 0:
        master_proc(comm, rank, status, processor_name)
    else:
        worker_proc(comm, rank, status, processor_name)
