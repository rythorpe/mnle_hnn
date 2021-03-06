"""
============================================
01. Simulate Event Related Potentials (ERPs)
============================================

This example demonstrates how to simulate a threshold level tactile
evoked response, as detailed in the `HNN GUI ERP tutorial
<https://jonescompneurolab.github.io/hnn-tutorials/erp/erp>`_,
using HNN-core. We recommend you first review the GUI tutorial.

The workflow below recreates an example of the threshold level tactile
evoked response, as observed in Jones et al. J. Neuroscience 2007 [1]_
(e.g. Figure 7 in the GUI tutorial), albeit without a direct comparison
to the recorded data.
"""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>
#          Blake Caldwell <1blakecaldwell@gmail.com>

import sys
import os.path as op
from os import environ
from mpi4py import MPI

from statistics import mean
import numpy as np


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def start_master_proc(comm, rank, status, processor_name):
    """Initialize a master process for setting parameters and collecting data"""

    print(f'Master starting with rank {rank} on {processor_name}')

    size = comm.Get_size()  # total number of inter-node processes
    if 'SLURM_NNODES' in environ:
        # main comm is oversubscribed: don't include master process
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1

    print(f"Master starting sensitivity analysis on {n_nodes} nodes")

    # worker proc parameters
    n_workers = n_tasks = n_nodes
    closed_workers = 0
    task_idx = 0

    # define parameters for each task
    new_params = np.linspace(75, 95, n_tasks)
    #comm.bcast(new_params, root=MPI.ROOT)

    #results = [None for task_idx in range(len(n_tasks))]
    while closed_workers < n_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
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
        elif tag == tags.EXIT:
            print("Worker %d exited (%d running)" % (source, closed_workers))
            closed_workers += 1


def start_worker_proc(comm, rank, status, processor_name):
    """Initialize a worker process for running simulations"""

    print("Worker started with rank %d on %s." % (rank, processor_name))

    # find the number of available cores for parallel processing
    if 'SLURM_CPUS_ON_NODE' in environ:
        # XXX minus 2???
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = 1

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('host', processor_name.split('.')[0])
    mpiinfo.Set('ompi_param', 'rmaps_base_inherit=0')
    mpiinfo.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpiinfo.Set('ompi_param', 'rmaps_base_oversubscribe=1')
    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn(sys.executable,
                                  args=['run_hnn_sim.py', str(n_procs)],
                                  info=mpiinfo, maxprocs=n_procs)
    # receive experimental data
    #(exp_data, params_input) = comm.bcast(comm.Get_rank(), root=0)

    # send params and exp_data to spawned nrniv procs
    #simdata = (exp_data, params_input)
    #subcomm.bcast(simdata, root=MPI.ROOT)

    avg_sim_times = []

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

        #assert(tag == tags.START)

        #finish = MPI.Wtime() - start
        #print('worker %s waited %.2fs for param set' % (processor_name, finish))

        # Start clock
        start = MPI.Wtime()

        # send new_params to spawned nrniv procs
        subcomm.bcast(new_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(2)])
        #temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (processor_name, finish, mean(avg_sim_times)))

        # send results back
        #comm.isend(temp_results, dest=0, tag=tags.DONE)

        # tell rank 0 we are ready (again)
        comm.isend(None, dest=0, tag=tags.READY)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

    # send empty new_params to stop nrniv procs
    subcomm.bcast(None, root=MPI.ROOT)
    #subcomm.Barrier()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.Get_rank()  # rank of this process
    status = MPI.Status()  # get MPI status object
    processor_name = MPI.Get_processor_name()

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    if rank == 0:
        start_master_proc(comm, rank, status, processor_name)
    else:
        start_worker_proc(comm, rank, status, processor_name)
