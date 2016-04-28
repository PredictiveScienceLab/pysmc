"""
Solve the reaction kinetics inverse problem.
"""


import reaction_kinetics_model
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt
import cPickle as pickle


if __name__ == '__main__':
    model = reaction_kinetics_model.make_model()
    # Construct the SMC sampler
    smc_sampler = pysmc.SMC(model, num_particles=100,
                            num_mcmc=1, verbose=1,
                            mpi=mpi, gamma_is_an_exponent=True)
    # Initialize SMC at gamma = 0.01
    smc_sampler.initialize(0.)
    # Move the particles to gamma = 1.0
    smc_sampler.move_to(1.)
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation()
    #m = p.mean
    #v = p.variance
    #if mpi.COMM_WORLD.Get_rank() == 0:
    #    print m
    #    print v
    #lp = p.allgather()
    # Plot a histogram
    #pysmc.hist(p, 'mixture')
    p_l = p.allgather()
    if mpi.COMM_WORLD.Get_rank() == 0:
        with open('reaction_kinetics_particle_approximation.pickle', 'wb') as fd:
            pickle.dump(p_l, fd, pickle.HIGHEST_PROTOCOL)
