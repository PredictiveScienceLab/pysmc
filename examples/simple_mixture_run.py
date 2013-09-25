"""
Run some tests with the simple_mixture model.

Author:
    Ilias Bilionis

Date:
    9/22/2013
"""


import pymc
import simple_mixture_model
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from pysmc import *
import matplotlib.pyplot as plt


mcmc_sampler = pymc.MCMC(simple_mixture_model, verbose=1)
smc_sampler = SMC(mcmc_sampler=mcmc_sampler, num_particles=100,
                  num_mcmc=10,
                  verbose=3)
smc_sampler.initialize(gamma=0.001)
smc_sampler.move_to(gamma=1.)
w, r = smc_sampler.get_particle_approximation('mixture')
#print w
#print w, r
plt.hist(r, bins=10, weights=w, normed=True)
plt.show()
