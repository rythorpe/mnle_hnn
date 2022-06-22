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


def run_hnn_simulation(n_procs, rank):
    """Run a single hnn-core simulation"""
    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    import hnn_core
    from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                          MPIBackend)
    from hnn_core.viz import plot_dipole

    # load param file
    params_fname = ('/users/rthorpe/data/rthorpe/hnn_out/param/'
                    'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')
    params = hnn_core.read_params(params_fname)
    net = jones_2009_model(params, add_drives_from_params=True)

    # run simulation distributed over multiple cores
    with MPIBackend(n_procs=n_procs):
        dpls = simulate_dipole(net, tstop=170., n_trials=25)

    # plot dpls
    plt.figure()
    scaling_factor = 40
    dpls = [dpl.scale(scaling_factor).smooth(20) for dpl in dpls]  # scale in place
    avg_dpl = average_dipoles(dpls)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    plot_dipole(dpls, ax=axes[0], show=False)
    plot_dipole(avg_dpl, ax=axes[1], show=False)
    plt.savefig(f'dipoles_mn{rank:02.0f}.png', dpi=300)

def start_master_proc(comm, size, rank, status, name):
    """Initialize a master process for setting parametes and collecting data"""
    
    if 'SLURM_NNODES' in environ:
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1

    print("Master starting sensitivity analysis on %d cores" % n_nodes)

def start_worker_proc(comm, size, rank, status, name):
    """Initialize a worker process for running simulations"""

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    print("Worker started with rank %d on %s." % (rank, name))

    # receive experimental data
    (exp_data, params_input) = comm.bcast(comm.Get_rank(), root=0)

    # find the number of available cores for parallel processing
    if 'SLURM_CPUS_ON_NODE' in environ:
        n_procs = int(environ['SLURM_CPUS_ON_NODE'])
    else:
        n_procs = 1

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('host', name.split('.')[0])
    mpiinfo.Set('ompi_param', 'rmaps_base_inherit=0')
    mpiinfo.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpiinfo.Set('ompi_param', 'rmaps_base_oversubscribe=1')
    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn('nrniv',
            args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                  'examples/calculate_dipole_err.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input)
    subcomm.bcast(simdata, root=MPI.ROOT)

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
            print('worker %d on %s has received exit signal'%(rank, name))
            break

        #assert(tag == tags.START)

        #finish = MPI.Wtime() - start
        #print('worker %s waited %.2fs for param set' % (name, finish))

        # Start clock
        start = MPI.Wtime()

        # send new_params to spawned nrniv procs
        subcomm.bcast(new_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(2)])
        temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (name, finish, mean(avg_sim_times)))
   
        # send results back
        comm.isend(temp_results, dest=0, tag=tags.DONE)

        # tell rank 0 we are ready (again)
        comm.isend(None, dest=0, tag=tags.READY)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

    # send empty new_params to stop nrniv procs
    subcomm.bcast(None, root=MPI.ROOT)
    #subcomm.Barrier()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    name = MPI.Get_processor_name()

    if rank == 0:
        start_master_proc(comm, size, rank, status, name)
    else:
        start_worker_proc(comm, size, rank, status, name)
