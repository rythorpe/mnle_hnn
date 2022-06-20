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

import os.path as op
import timeit

import matplotlib
matplotlib.use('TkAgg')
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

# run 100 trials per simulation with varying numbers of cores
n_procs_list = [1] + list(range(2, 25, 2))
durations = list()
for n_procs in n_procs_list:
    start = timeit.default_timer()

    with MPIBackend(n_procs=n_procs):
        dpls = simulate_dipole(net, tstop=170., n_trials=25)

    stop = timeit.default_timer()
    durations.append((stop - start) / 60)
    print(f'n_procs: {n_procs}')

# plot run time vs. # of cores
plt.figure()
plt.step(n_procs_list, durations, where='post')
plt.xlabel('# of cores')
plt.ylabel('computation time (min)')
# plt.savefig('clock_hnn_core.png', dpi=300)
# plt.savefig('clock_hnn_core.eps')

# plot dpls
plt.figure()
scaling_factor = 40
dpls = [dpl.scale(scaling_factor).smooth(20) for dpl in dpls]  # scale in place
avg_dpl = average_dipoles(dpls)
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
plot_dipole(dpls, ax=axes[0], show=False)
plot_dipole(avg_dpl, ax=axes[1], show=False)
# plt.savefig('clock_hnn_core_mn.png', dpi=300)