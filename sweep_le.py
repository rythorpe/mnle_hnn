"""Parameter sweep for LE response."""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

import os.path as op
import timeit
import numpy as np
import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend, JoblibBackend)

# hyper-params for parameter sweep
n_sweep_sims = 1  # XXX
n_trials_per_sim = 1  # XXX
seed = 1
params_to_vary = {'evprox_1': ['mu',
                               'L2_basket_ampa',
                               'L2_basket_nmda',
                               'L5_basket_ampa',
                               'L5_basket_nmda'],
                  'evdist_1': ['mu',
                               'L2_basket_ampa',
                               'L2_basket_nmda',
                               'L5_basket_ampa',
                               'L5_basket_nmda'],
                  'dist_burst': ['idi'],
                  'evprox_2': ['mu',
                               'L5_pyramidal_ampa',
                               'L5_pyramidal_nmda']}
params_to_vary = {'dist_burst': ['idi'], 'evprox_2': ['mu']}  # XXX fix
params_fname = 'laser_4dist_2prox_50trials_opt1_smooth.param'
write_dir = '/users/rthorpe/scratch/sweep_le_output/'


def sample_param(original_val):
    # explore values <=10% change in original value
    lower_b = original_val - 0.1 * original_val
    upper_b = original_val + 0.1 * original_val
    return lower_b + rng.random() * (upper_b - lower_b)


def sample_param_const_dx(original_val, time_param=False):
    # explore values sampled more consitently across parameters
    if time_param:
        # sample uniformly +/-10 ms
        lower_b = original_val - 10
        upper_b = original_val + 10
        x = lower_b + rng.random() * (upper_b - lower_b)
    else:
        # sample on log_10 scaled
        lower_b = -5  # lower bound exponent
        upper_b = -3  # upper bound exponent
        log_x = lower_b + rng.random() * (upper_b - lower_b)
        x = 10 ** log_x
    return x


def get_drive_params(drive_name, resample_param=None):

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
        mu = 120.008737
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
        mu = 145.221516
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
        mu = 169.94315
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
        mu = 195.079004
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

    # for collective burst of distal drives, param resampling must happen
    # outside of the for loop below
    if selected_drive_name == 'dist_burst':
        burst_center_time = 157.5
        original_isi = 25.0
        resampled_isi = sample_param_const_dx(original_isi, time_param=True)
        dist_drive_times = np.arange(4) * resampled_isi
        dist_drive_times -= dist_drive_times.mean()
        dist_drive_times += burst_center_time

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

        # reset all distal drives in the burst if the inra-burst ISI is being
        # jittered this simulation
        if selected_drive_name == 'dist_burst':
            param_val = resampled_isi
            if name == 'evdist_1':
                mu = dist_drive_times[0]
            if name == 'evdist_2':
                mu = dist_drive_times[1]
            if name == 'evdist_3':
                mu = dist_drive_times[2]
            if name == 'evdist_4':
                mu = dist_drive_times[3]

        # add synchronous drive
        net.add_evoked_drive(
            name=name, mu=mu, sigma=sigma, numspikes=1,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda, location=loc,
            synaptic_delays=syn_delays,
            n_drive_cells=1, cell_specific=False)

    with MPIBackend(n_procs=24):
        dpls = simulate_dipole(net, tstop=300., n_trials=n_trials_per_sim)

    scaling_factor = 2500
    smooth_win = 20
    for dpl in dpls:
        dpl.scale(scaling_factor).smooth(smooth_win)
    avg_dpl = average_dipoles(dpls)
    avg_dpl.plot()
    fname_out = f'{drive_name}_{param_name}_{param_val:.4e}.txt'
    avg_dpl.write(op.join(write_dir, fname_out))


if __name__ == "__main__":
    params = hnn_core.read_params(params_fname)
    rng = np.random.default_rng(seed)
    
    all_drive_names = ['evprox_1', 'evdist_1', 'evdist_2', 'evdist_3',
                       'evdist_4', 'evprox_2']
    for drive_name, drive_params in params_to_vary.items():
        for drive_param in drive_params:
            for sweep_idx in range(n_sweep_sims):
                start_t = timeit.default_timer()
                run_and_save(all_drive_names, drive_name, drive_param,
                             params.copy())
                stop_t = timeit.default_timer()
                print(f'single sim run time: {stop_t - start_t}')