"""
Test the step method when trying to sample a lognormal random variable.
"""


import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import pymc
import math
import matplotlib.pyplot as plt
import numpy as np


def make_model():
    #c = pymc.Container([pymc.Lognormal('x', mu=math.log(1.), tau=1.),
    #                    pymc.Lognormal('z', mu=math.log(2.), tau=3.)])
    #x = pymc.Exponential('x', beta=0.1)
    x = pymc.Lognormal('x', mu=np.array([math.log(1.), math.log(2.)]),
                       tau=np.array([1., 3.]))
    @pymc.stochastic(observed=True)
    def y(value=0.01, x=x):
        return pymc.lognormal_like(value, mu=x[0], tau=1.)
    return locals()


if __name__ == '__main__':
    m = make_model()
    mcmc = pymc.MCMC(m)
    mcmc.assign_step_methods()
    print mcmc.step_method_dict
    mcmc.sample(20000, thin=100, burn=1000)
    s = mcmc.step_method_dict[m['x']][0]
    print '\n', s.accepted / (s.accepted + s.rejected)
    pymc.Matplot.plot(mcmc)
    plt.show()
