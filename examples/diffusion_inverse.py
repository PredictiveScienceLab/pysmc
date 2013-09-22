"""
Example: Solving an inverse problem with MCMC.
----------------------------------------------
"""


import pymc
import diffusion_inverse_model as model
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from pysmc import *


mcmc_sampler = pymc.MCMC(model, db='pickle',
                         dbname='diffusion.pickle', verbose=0)
#g = pymc.graph.dag(mcmc_sampler)
#g.write_png('diffusion_model.png')
smc_sampler = SMC(mcmc_sampler=mcmc_sampler, num_particles=5,
                  verbose=True)
smc_sampler.sample()