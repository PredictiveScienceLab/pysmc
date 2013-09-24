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
mcmc_sampler.use_step_method(pymc.Metropolis, simple_mixture_model.mixture, proposal_sd=0.1)
#mcmc_sampler.sample(1000000, burn=1000, thin=1000, stop_tuning_after=0)
#pymc.Matplot.plot(mcmc_sampler)
#plt.show()
smc_sampler = SMC(mcmc_sampler=mcmc_sampler, num_particles=100,
                  num_mcmc=10,
                  verbose=True)
smc_sampler.sample()
w, r = smc_sampler.get_particle_approximation('mixture')
print w
print w, r
plt.hist(r, bins=10, weights=w, normed=True)
plt.show()