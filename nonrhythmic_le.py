"""Run a nonrhythmic control simulation for the LE response."""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend, read_dipole)
from hnn_core.viz import plot_dipole

# hyper-params for parameter sweep
n_trials_per_sim = 100
seed = 1
#params_fname = 'laser_4dist_2prox_50trials_opt1_smooth.param'
params_fname = ('/home/ryan/Dropbox (Brown)/nociceptive_erp_paper_figures/'
                'scripts/sim_data_hnn_core/sim_data_all/'
                'laser_4dist_2prox_50trials_opt1_smooth/'
                'laser_4dist_2prox_50trials_opt1_smooth.param')
#write_dir = '/users/rthorpe/scratch/sweep_le_output/'
write_dir = '/home/ryan/Desktop/stuff/'


def sample_param(lb, ub):
    # uniformly sample within upper and lower bounds
    return lb + rng.random() * (ub - lb)


def get_drive_params(drive_name):

    # Proximal 1
    if drive_name == 'evprox_1':
        mu = 119.400535
        sigma = 11.990812
        weights_ampa = {'L2_basket': 0.001369, 'L2_pyramidal': 0.00247,
                        'L5_basket': 0.000875, 'L5_pyramidal': 0.00206}
        weights_nmda = {'L2_basket': 0.000775, 'L2_pyramidal': 0.000076,
                        'L5_basket': 0.000852, 'L5_pyramidal': 0.0}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1., 'L5_pyramidal': 1.}
        loc = 'proximal'

    # Distal 1
    elif drive_name == 'evdist_1':
        mu = sample_param(120.0, 200.0)
        sigma = 10.038986
        weights_ampa = {'L2_basket': 0.005218, 'L2_pyramidal': 0.004511,
                        'L5_pyramidal': 0.001218}
        weights_nmda = {'L2_basket': 0.002971, 'L2_pyramidal': 0.004739,
                        'L5_pyramidal': 0.00069}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Distal 2
    elif drive_name == 'evdist_2':
        mu = sample_param(120.0, 200.0)
        sigma = 9.996537
        weights_ampa = {'L2_basket': 0.005431, 'L2_pyramidal': 0.004463,
                        'L5_pyramidal': 0.00068}
        weights_nmda = {'L2_basket': 0.002817, 'L2_pyramidal': 0.007967,
                        'L5_pyramidal': 0.000661}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Distal 3
    elif drive_name == 'evdist_3':
        mu = sample_param(120.0, 200.0)
        sigma = 8.83142
        weights_ampa = {'L2_basket': 0.004804, 'L2_pyramidal': 0.005007,
                        'L5_pyramidal': 0.0007}
        weights_nmda = {'L2_basket': 0.003121, 'L2_pyramidal': 0.006249,
                        'L5_pyramidal': 0.000646}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Distal 4
    elif drive_name == 'evdist_4':
        mu = sample_param(120.0, 200.0)
        sigma = 10.141505
        weights_ampa = {'L2_basket': 0.004026, 'L2_pyramidal': 0.004564,
                        'L5_pyramidal': 0.000678}
        weights_nmda = {'L2_basket': 0.002731, 'L2_pyramidal': 0.004486,
                        'L5_pyramidal': 0.000695}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Proximal 2
    elif drive_name == 'evprox_2':
        mu = 252.933393
        sigma = 12.32985
        weights_ampa = {'L2_basket': 0.000784, 'L2_pyramidal': 0.003678,
                        'L5_basket': 0.000603, 'L5_pyramidal': 0.001343}
        weights_nmda = {'L2_basket': 0.000818, 'L2_pyramidal': 0.000192,
                        'L5_basket': 0.000341, 'L5_pyramidal': 0.000418}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1., 'L5_pyramidal': 1.}
        loc = 'proximal'

    return mu, sigma, weights_ampa, weights_nmda, syn_delays, loc


if __name__ == "__main__":

    emp_dpl = read_dipole('/home/ryan/Dropbox (Brown)/'
                          'nociceptive_erp_paper_figures/scripts/'
                          'sim_data_hnn_core/empirical_dpl_data/'
                          'laser_ecd_rl_avg_cropped.txt')

    params = hnn_core.read_params(params_fname)
    rng = np.random.default_rng(seed)
    all_drive_names = ['evprox_1', 'evdist_1', 'evdist_2', 'evdist_3',
                       'evdist_4', 'evprox_2']

    # instantiate with original params to set local net connectivity
    # (without automatically adding drives)
    net = jones_2009_model(params=params)

    dpls = list()
    for trial_idx in range(n_trials_per_sim):
        print(f'trial {trial_idx}')
        net.clear_drives()
        event_seed = seed + trial_idx

        for drive_name in all_drive_names:
            drive_specs = get_drive_params(drive_name)
            mu, sigma, weights_ampa, weights_nmda, syn_delays, loc = drive_specs

            # add synchronous drive
            net.add_evoked_drive(
                name=drive_name, mu=mu, sigma=sigma, numspikes=1,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda, location=loc,
                synaptic_delays=syn_delays,
                n_drive_cells=1, cell_specific=False, event_seed=event_seed)

        with MPIBackend(n_procs=10):
            dpls.extend(simulate_dipole(net, tstop=300., n_trials=1))
        dpls[-1].write(op.join(write_dir, f'dpl_{trial_idx}.txt'))
        net.cell_response.write(op.join(write_dir, f'spk_{trial_idx}.txt'))

    scaling_factor = 2500
    smooth_win = 20
    for dpl in dpls:
        dpl.scale(scaling_factor).smooth(smooth_win)
    avg_dpl = average_dipoles(dpls)

    fig, axes = plt.subplots(1, 1)
    plot_dipole(emp_dpl, ax=axes, color='r', label='empirical', show=False)
    plot_dipole(avg_dpl, ax=axes, color='k', label='simulated', show=True)
    fname_out = 'dpl_avg.txt'
    avg_dpl.write(op.join(write_dir, fname_out))
