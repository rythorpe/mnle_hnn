"""Parameter sweep for P30 deflection of MN response."""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

import os.path as op
import timeit
import numpy as np
import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend, JoblibBackend)

# hyper-params for parameter sweep
n_sweep_sims = 50
n_trials_per_sim = 100
seed = 1
params_to_vary = {'evprox_1': ['mu',
                               'L2_pyramidal_nmda',
                               'L5_basket_nmda',
                               'L5_pyramidal_nmda'],
                  'evdist_1': ['mu',
                               'L2_basket_ampa',
                               'L2_pyramidal_nmda',
                               'L5_pyramidal_nmda'],
                  'evdist_2': ['mu',
                               'L2_basket_ampa',
                               'L2_pyramidal_nmda',
                               'L5_pyramidal_nmda'],
                  'evprox_2': ['mu',
                               'L2_pyramidal_nmda',
                               'L5_basket_nmda',
                               'L5_pyramidal_nmda']}
params_fname = 'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param'
write_dir = '/users/rthorpe/scratch/sweep_mn_output_const_dx/'


def sample_param(original_val):
    # explore values <=10% change in original value
    lower_b = original_val - 0.1 * original_val
    upper_b = original_val + 0.1 * original_val
    return lower_b + rng.random() * (upper_b - lower_b)


def sample_param_const_dx(original_val, time_param=False):
    # explore values sampled more consitently across parameters
    if time_param:
        # sample uniformly +/-4 ms
        lower_b = original_val - 4
        upper_b = original_val + 4
        x = lower_b + rng.random() * (upper_b - lower_b)
    else:
        # sample on log_10 scaled
        lower_b = -5  # lower bound exponent
        upper_b = -2  # upper bound exponent
        log_x = lower_b + rng.random() * (upper_b - lower_b)
        x = 10 ** log_x
    return x


def get_drive_params(drive_name, resample_param=None):

    # Proximal 1
    if drive_name == 'evprox_1':
        mu = 20.808669
        sigma = 4.121563
        weights_ampa = {'L2_basket': 0.003617, 'L2_pyramidal': 0.003903,
                        'L5_basket': 0.003037, 'L5_pyramidal': 0.001963}
        weights_nmda = {'L2_basket': 0.002893, 'L2_pyramidal': 0.000505,
                        'L5_basket': 0.00278, 'L5_pyramidal': 0.001869}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1., 'L5_pyramidal': 1.}
        loc = 'proximal'

    # Distal 1
    elif drive_name == 'evdist_1':
        mu = 31.592845
        sigma = 2.714237
        weights_ampa = {'L2_basket': 0.004265, 'L2_pyramidal': 0.003248,
                        'L5_pyramidal': 0.000932}
        weights_nmda = {'L2_basket': 0.002884, 'L2_pyramidal': 0.005126,
                        'L5_pyramidal': 0.000983}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Distal 2
    elif drive_name == 'evdist_2':
        mu = 83.962981
        sigma = 4.356796
        weights_ampa = {'L2_basket': 0.004065, 'L2_pyramidal': 0.001884,
                        'L5_pyramidal': 0.001802}
        weights_nmda = {'L2_basket': 0.003188, 'L2_pyramidal': 0.00177,
                        'L5_pyramidal': 0.001749}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
        loc = 'distal'

    # Proximal 2
    elif drive_name == 'evprox_2':
        mu = 134.383155
        sigma = 4.5
        weights_ampa = {'L2_basket': 0.003989, 'L2_pyramidal': 0.008833,
                        'L5_basket': 0.006875, 'L5_pyramidal': 0.00238}
        weights_nmda = {'L2_basket': 0.008614, 'L2_pyramidal': 0.00926,
                        'L5_basket': 0.002055, 'L5_pyramidal': 0.003047}
        syn_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1., 'L5_pyramidal': 1.}
        loc = 'proximal'

    # resample a param if specified
    new_val = None
    if resample_param is not None:
        if resample_param == 'mu':
            mu = new_val = sample_param_const_dx(mu, time_param=True)
        elif resample_param[-4:] == 'ampa':
            original_val = weights_ampa[resample_param[:-5]]
            weights_ampa[resample_param[:-5]] = sample_param_const_dx(original_val)
            new_val = weights_ampa[resample_param[:-5]]
        elif resample_param[-4:] == 'nmda':
            original_val = weights_nmda[resample_param[:-5]]
            weights_nmda[resample_param[:-5]] = sample_param_const_dx(original_val)
            new_val = weights_nmda[resample_param[:-5]]

    return mu, sigma, weights_ampa, weights_nmda, syn_delays, loc, new_val


def run_and_save(all_drive_names, selected_drive_name, param_name,
                 params_original):
    # instantiate with original params to set local net connectivity
    # (without automatically adding drives)
    net = jones_2009_model(params=params_original)

    for name in all_drive_names:
        # only resample a given parameter (from its original value) for the
        # selected drive; all other drives default to original values
        if name == selected_drive_name:
            resample_param = param_name
        else:
            resample_param = None

        drive_params = get_drive_params(drive_name=name,
                                        resample_param=resample_param)
        mu, sigma, weights_ampa, weights_nmda, syn_delays, loc, new_val = drive_params
        if new_val is not None:
            param_val = new_val

        # add synchronous drive
        net.add_evoked_drive(
            name=name, mu=mu, sigma=sigma, numspikes=1,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda, location=loc,
            synaptic_delays=syn_delays,
            n_drive_cells=1, cell_specific=False)

    with MPIBackend(n_procs=24):
        dpls = simulate_dipole(net, tstop=170., n_trials=n_trials_per_sim)

    scaling_factor = 40
    smooth_win = 20
    for dpl in dpls:
        dpl.scale(scaling_factor).smooth(smooth_win)
    avg_dpl = average_dipoles(dpls)
    #avg_dpl.plot()
    fname_out = f'{drive_name}_{param_name}_{param_val:.6e}.txt'
    avg_dpl.write(op.join(write_dir, fname_out))


if __name__ == "__main__":
    params = hnn_core.read_params(params_fname)
    rng = np.random.default_rng(seed)
    #start_t = timeit.default_timer()
    for drive_name, drive_params in params_to_vary.items():
        for drive_param in drive_params:
            for sweep_idx in range(n_sweep_sims):
                run_and_save(params_to_vary.keys(), drive_name, drive_param,
                             params.copy())
    #stop_t = timeit.default_timer()
    #print(f'single sim run time: {stop_t - start_t}')
