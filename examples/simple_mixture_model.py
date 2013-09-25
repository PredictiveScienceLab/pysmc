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
def mixture(value=-0., gamma=.001, sigma=0.01):
    def logp(value, gamma, sigma):
        tau = math.sqrt(1. / sigma ** 2)
        logp1 = math.log(0.2) + pymc.normal_like(value, -1., tau)
        logp2 = math.log(0.8) + pymc.normal_like(value, 2., tau)
        tmp = math.fsum([math.exp(logp1), math.exp(logp2)])
        if tmp <= 0.:
            return -np.inf
        return gamma * math.log(tmp)
