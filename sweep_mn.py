"""Parameter sweep for P30 deflection of MN response."""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend)

# hyper-params for parameter sweep
n_sweep_sims = 100
n_trials_per_sim = 100
params_to_vary = {''}
params_fname = ('/users/rthorpe/data/rthorpe/hnn_out/param/'
                'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')

params = hnn_core.read_params(params_fname)


net = jones_2009_model(params, add_drives_from_params=True)

with MPIBackend(n_procs=24, mpi_comm_spawn=False):
    dpls = simulate_dipole(net, tstop=170., n_trials=n_trials_per_sim)
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