"""Run a single hnn-core simulation"""

import sys
from mpi4py import MPI
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend)
from hnn_core.viz import plot_dipole


if __name__ == "__main__":
    # set MPI variables
    n_procs = int(sys.argv[1])
    comm = MPI.COMM.Get_parent()
    rank = comm.Get_rank()

    new_param = comm.bcast(rank, root=0)  # blocking

    # load param file
    params_fname = ('/users/rthorpe/data/rthorpe/hnn_out/param/'
                    'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')
    params = hnn_core.read_params(params_fname)

    params['t_evdist_1'] = new_param
    net = jones_2009_model(params, add_drives_from_params=True)

    start = MPI.Wtime()
    # run simulation distributed over multiple cores
    with MPIBackend(n_procs=n_procs):
        dpls = simulate_dipole(net, tstop=170., n_trials=25)
    finish = MPI.Wtime()
    elapsed_time = finish - start
    print(f'Completed simulation on rank {rank} in {elapsed_time}')

    # plot dpls & save
    plt.figure()
    scaling_factor = 40
    smooth_win = 20
    dpls = [dpl.scale(scaling_factor).smooth(smooth_win) for dpl in dpls]
    avg_dpl = average_dipoles(dpls)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    plot_dipole(dpls, ax=axes[0], show=False)
    plot_dipole(avg_dpl, ax=axes[1], show=False)
    plt.savefig(f'dipoles_mn_{rank:02.0f}.png', dpi=300)
