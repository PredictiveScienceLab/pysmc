"""
Example: Solving an inverse problem with MCMC.
----------------------------------------------
"""


import diffusion_inverse_model
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import mpi4py.MPI as mpi


if __name__ == '__main__':
    model = diffusion_inverse_model.make_model()
    smc_sampler = pysmc.SMC(model, num_particles=100, num_mcmc=10,
                            verbose=1, mpi=mpi,
                            gamma_is_an_exponent=True)
    smc_sampler.initialize(0.)
    smc_sampler.move_to(1.)
    p = smc_sampler.get_particle_approximation()
    m = p.mean
    v = p.variance
    if mpi.COMM_WORLD.Get_rank() == 0:
        print m
        print v
