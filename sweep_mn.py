"""Parameter sweep for P30 deflection of MN response."""

# Authors: Ryan Thorpe <ryvthorpe@gmail.com>

import hnn_core
from hnn_core import (simulate_dipole, jones_2009_model, average_dipoles,
                      MPIBackend)

# hyper-params for parameter sweep
n_sweep_sims = 50
n_trials_per_sim = 100
params_to_vary = []
params_fname = ('/users/rthorpe/data/rthorpe/hnn_out/param/'
                'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')
#params_fname = ('/home/ryan/Dropbox (Brown)/nociceptive_erp_paper_figures/'
#                'scripts/sims_hnn_core/sim_data_initial/'
#                'med_nerve_2020_04_27_2prox_2dist_opt1_smooth/'
#                'med_nerve_2020_04_27_2prox_2dist_opt1_smooth.param')


def run_and_save(drive_name, param_name, param_val, params_original):
    # instantiate with original params to set local net connectivity
    # (without automatically adding drives)
    net = jones_2009_model(params=params_original)

    # Proximal 1 (synchronous)
    weights_ampa_p1 = {'L2_basket': 0.003617, 'L2_pyramidal': 0.003903,
                       'L5_basket': 0.003037, 'L5_pyramidal': 0.001963}
    weights_nmda_p1 = {'L2_basket': 0.002893, 'L2_pyramidal': 0.000505,
                       'L5_basket': 0.00278, 'L5_pyramidal': 0.001869}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive(
        'evprox_1', mu=20.808669, sigma=4.121563, numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=weights_nmda_p1, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=544,
        n_drive_cells=1, cell_specific=False)

    # Distal 1 (synchronous)
    weights_ampa_d1 = {'L2_basket': 0.004265, 'L2_pyramidal': .003248,
                       'L5_pyramidal': 0.000932}
    weights_nmda_d1 = {'L2_basket': 0.002884, 'L2_pyramidal': 0.005126,
                       'L5_pyramidal': 0.000983}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        'evdist_1', mu=31.592845, sigma=2.714237, numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1, location='distal',
        synaptic_delays=synaptic_delays_d1, event_seed=274,
        n_drive_cells=1, cell_specific=False)

    # Distal 2 (synchronous)
    weights_ampa_d1 = {'L2_basket': 0.004065, 'L2_pyramidal': .001884,
                       'L5_pyramidal': 0.001802}
    weights_nmda_d1 = {'L2_basket': 0.003188, 'L2_pyramidal': 0.00177,
                       'L5_pyramidal': 0.001749}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        'evdist_2', mu=83.962981, sigma=4.356796, numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1, location='distal',
        synaptic_delays=synaptic_delays_d1, event_seed=274,
        n_drive_cells=1, cell_specific=False)

    # Proximal 2 (synchronous)
    weights_ampa_p2 = {'L2_basket': 0.003989, 'L2_pyramidal': 0.008833,
                       'L5_basket': 0.006875, 'L5_pyramidal': 0.00238}
    weights_nmda_p2 = {'L2_basket': 0.008614, 'L2_pyramidal': 0.00926,
                       'L5_basket': 0.002055, 'L5_pyramidal': 0.003047}
    net.add_evoked_drive(
        'evprox_2', mu=134.383155, sigma=4.5, numspikes=1,
        weights_ampa=weights_ampa_p2,
        weights_nmda=weights_nmda_p2, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=814,
        n_drive_cells=1, cell_specific=False)

    with MPIBackend(n_procs=10):
        dpls = simulate_dipole(net, tstop=170., n_trials=n_trials_per_sim)

    scaling_factor = 40
    smooth_win = 20
    for dpl in dpls:
        dpl.scale(scaling_factor).smooth(smooth_win)
    avg_dpl = average_dipoles(dpls)
    avg_dpl.plot()
    avg_dpl.write(f'{param_name}_{param_val}.txt')


if __name__ == "__main__":
    params = hnn_core.read_params(params_fname)
    run_and_save('drive_1', 'param_1', 5, params)