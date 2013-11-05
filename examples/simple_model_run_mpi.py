#!/usr/bin/env python
"""
Run SMC on the simple model.

Author:
    Ilias Bilionis

Date:
    9/28/2013

"""


import simple_model
import pymc
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc as ps
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np


if __name__ == '__main__':
    do_init = not os.path.exists('test_db.pickle')
    # Construct the SMC sampler
    model = simple_model.make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.GaussianMixtureStep, model['mixture'])
    smc_sampler = ps.SMC(mcmc, num_particles=512,
                         num_mcmc=1, verbose=5,
                         mpi=mpi, gamma_is_an_exponent=True,
                         ess_reduction=0.9,
                         db_filename='test_db.pickle',
                         update_db=True)
    # Initialize SMC at gamma = 0.01
    if do_init:
        smc_sampler.initialize(0.01, num_mcmc_per_particle=100)
    # Move the particles to gamma = 1.0
    smc_sampler.move_to(1.)
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation().gather()
    # Plot a histogram
    if mpi.COMM_WORLD.Get_rank() == 0:
        x = np.linspace(-5, 5, 200)
        y = simple_model.eval_stochastic_variable(model['mixture'], x)
        plt.plot(x, y, linewidth=2)
        ps.hist(p, 'mixture')
        plt.show()
