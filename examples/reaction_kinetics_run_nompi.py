"""
Solve the reaction kinetics inverse problem.
"""


import reaction_kinetics_model
import sys
import os
import pysmc
import pymc
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import cPickle as pickle


if __name__ == '__main__':
    model = reaction_kinetics_model.make_model()
    # Construct the SMC sampler
    mcmc = pymc.MCMC(model)
    mcmc.use_step_method(pysmc.LognormalRandomWalk, model['k1'])
    mcmc.use_step_method(pysmc.LognormalRandomWalk, model['k2'])
    mcmc.use_step_method(pysmc.LognormalRandomWalk, model['sigma'])
    smc_sampler = pysmc.SMC(mcmc, num_particles=100,
                            num_mcmc=1, verbose=1,
                            gamma_is_an_exponent=True)
    # Initialize SMC at gamma = 0.01
    smc_sampler.initialize(0.0)
    # Move the particles to gamma = 1.0
    smc_sampler.move_to(1.)
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation()
    print p.mean
    print p.variance

    data = [p.particles[i]['stochastics']['k1'] for i in xrange(p.num_particles)]
    data = np.array(data)
    plt.plot(data, np.zeros(data.shape), 'ro', markersize=10)
    pysmc.hist(p, 'mixture')

    data = [p.particles[i]['stochastics']['k2'] for i in xrange(p.num_particles)]
    data = np.array(data)
    plt.plot(data, np.zeros(data.shape), 'bo', markersize=10)
    pysmc.hist(p, 'mixture')    
