"""
A simple mixture model to test the capabilities of SMC.

Author:
    Ilias Bilionis

Date:
    9/22/2013
"""


import pymc
import numpy as np
import math


@pymc.stochastic(dtype=float)
def mixture(value=0., gamma=1.):
    def logp(value, gamma):

        logp1 = math.log(0.2) + pymc.normal_like(value, -1., 10.)
        logp2 = math.log(0.8) + pymc.normal_like(value, 2., 10.)
        return (gamma * math.log(math.fsum([math.exp(logp1), math.exp(logp2)]))
                + (1. - gamma) * pymc.normal_like(value, 0., 1))

    def random(gamma):
        return np.random.randn()