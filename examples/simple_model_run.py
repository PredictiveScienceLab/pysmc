"""
Run SMC on the simple model.

Author:
    Ilias Bilionis

Date:
    9/28/2013

"""


import simple_model as model
import pymc
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import cPickle as pickle


if __name__ == '__main__':
    # Construct the MCMC sampler
    mcmc_sampler = pymc.MCMC(model)
    # Construct the SMC sampler
    smc_sampler = pysmc.SMC(mcmc_sampler, num_particles=100,
                            num_mcmc=10, verbose=1)
    # Initialize SMC at gamma = 0.01
    smc_sampler.initialize(0.01)
    # Move the particles to gamma = 1.0
    smc_sampler.move_to(1.0)
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation()
     # Plot a histogram
    plt.hist(p.mixture, weights=p.weights, bins=100, normed=True)
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$p(x)$', fontsize=16)
    plt.show()
