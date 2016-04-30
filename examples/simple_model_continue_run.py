"""
This demo demonstrates how you can continue an SMC run from the HDF5
database file.

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
import pysmc
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np


if __name__ == '__main__':
    # Construct the SMC sampler
    model = simple_model.make_model()
    mcmc = pymc.MCMC(model)
    mcmc.use_step_method(pysmc.RandomWalk, model['mixture'])
    db_filename = 'simple_model_continue.h5'
    if os.path.exists(db_filename):
        os.remove(db_filename)
    smc_sampler = pysmc.SMC(mcmc, num_particles=1000,
                            num_mcmc=1, verbose=1,
                            db_filename=db_filename)
    # Initialize SMC at gamma = 1e-4
    smc_sampler.initialize(1e-4)
    # Move the particles to gamma = 0.5
    smc_sampler.move_to(.5)
    # Now get rid of this SMC sampler.
    del smc_sampler
    del mcmc
    del model

    # Now we intialize a new sampler which will continue from the last spot.
    model = simple_model.make_model()
    mcmc = pymc.MCMC(model)
    smc_sampler = pysmc.SMC(mcmc, db_filename=db_filename)
    # Notice that no initialization is required
    smc_sampler.move_to(1.)
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation()
    #print p.mean
    #print p.variance
    # Plot a histogram
    data = [p.particles[i]['stochastics']['mixture'] for i in xrange(p.num_particles)]
    data = np.array(data)
    plt.plot(data, np.zeros(data.shape), 'ro', markersize=10)
    pysmc.hist(p, 'mixture')
    plt.show()
