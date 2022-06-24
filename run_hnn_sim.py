"""Run a single hnn-core simulation"""

import sys
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend)
from hnn_core.viz import plot_dipole


if __name__ == "__main__":
    n_procs = int(sys.argv[1])
    rank = int(sys.argv[2])

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
    scaling_factor = 
    smooth_win = 20
dpls = [dpl.scale(scaling_factor).smooth(smooth_win) for dpl in dpls]
    avg_dpl = average_dipoles(dpls)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    plot_dipole(dpls, ax=axes[0], show=False)
    plot_dipole(avg_dpl, ax=axes[1], show=False)
    plt.savefig(f'dipoles_mn{rank:02.0f}.png', dpi=300)
