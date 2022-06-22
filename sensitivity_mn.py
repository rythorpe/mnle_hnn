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
from os import environ
import timeit

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

# find the number of available cores for parallel processing
if 'SLURM_CPUS_ON_NODE' in environ:
    n_procs = int(environ['SLURM_CPUS_ON_NODE'])
else:
    n_procs = 1
durations = list()
start = timeit.default_timer()

# run simulation distributed over multiple cores
with MPIBackend(n_procs=n_procs):
    dpls = simulate_dipole(net, tstop=170., n_trials=25)

stop = timeit.default_timer()
durations.append((stop - start) / 60)
print(f'n_procs: {n_procs}')

# plot run time vs. # of cores
plt.figure()
plt.step(n_procs, durations, where='post')
plt.xlabel('# of cores')
plt.ylabel('computation time (min)')
plt.savefig('sensitivity_clock_time.png', dpi=300)

# plot dpls
plt.figure()
scaling_factor = 40
dpls = [dpl.scale(scaling_factor).smooth(20) for dpl in dpls]  # scale in place
avg_dpl = average_dipoles(dpls)
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
plot_dipole(dpls, ax=axes[0], show=False)
plot_dipole(avg_dpl, ax=axes[1], show=False)
plt.savefig('dipoles_mn.png', dpi=300)