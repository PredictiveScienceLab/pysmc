"""
A simple mixture model to test the capabilities of SMC.

Author:
    Ilias Bilionis

Date:
    9/22/2013
"""


import pymc
import numpy as np


@pymc.stochastic(dtype=float)
def mixture(value=-0., gamma=.001, pi=[0.2, 0.8], mu=[-1., 2.],
            sigma=[0.01, 0.01]):
    # The number of components in the mixture
    n = len(pi)
    # pymc.normal_like requires the precision not the variance:
    tau = np.sqrt(1. / sigma ** 2)
    # The following looks a little bit awkward because of the need for
    # numerical stability:
    logp = np.log(pi)
    logp += np.array([pymc.normal_like(value, mu[i], tau[i])
                      for i in range(n)]
    logp = math.fsum(np.exp(logp))
    # logp should never be negative, but it can be zero...
    if logp <= 0.:
        return -np.inf
    return gamma * math.log(logp)
