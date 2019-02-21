"""
Example: Solving an inverse problem with MCMC.
----------------------------------------------
"""


import warnings
warnings.filterwarnings("ignore")
import diffusion_inverse_model
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc as ps
import pymc as pm
import mpi4py.MPI as mpi


if __name__ == '__main__':
    model = diffusion_inverse_model.make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.GaussianMixtureStep, model['location'])
    mcmc.use_step_method(ps.LognormalRandomWalk, model['alpha'],
                         proposal_sd=1.)
    mcmc.use_step_method(ps.LognormalRandomWalk, model['beta'],
                         proposal_sd=1.)
    mcmc.use_step_method(ps.LognormalRandomWalk, model['tau'],
                         proposal_sd=1.)
    try:
        smc_sampler = ps.SMC(mcmc, num_particles=64, num_mcmc=1,
                             verbose=3, mpi=mpi,
                             db_filename='diffusion_inverse.pickle',
                             update_db=True,
                             gamma_is_an_exponent=True)
        smc_sampler.initialize(0.)
        smc_sampler.move_to(1.)
    except Exception as e:
        print(str(e))
        mpi.COMM_WORLD.Abort(1)
    p = smc_sampler.get_particle_approximation()
    m = p.mean
    v = p.variance
    if mpi.COMM_WORLD.Get_rank() == 0:
        print(m)
        print(v)
