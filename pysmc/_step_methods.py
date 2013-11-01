"""

.. _step_methods:

============
Step Methods
============

In PySMC we define a few step methods for the Metropolis-Hastings algorithm
that extend the capabilities of PyMC.

Here is a list of what we offer:

"""


__all__ = ['LognormalRandomWalk']


import pymc
import numpy as np
from sklearn import mixture


class LognormalRandomWalk(pymc.Metropolis):
    """
    This is a step method class that is good for positive random variables.
    It is a essentially a random walk in the logarithmic scale.

    **Base class:** :class:`pymc.Metropolis`
    """

    def __init__(self, stochastic, *args, **kwargs):
        """Initialize the object."""
        pymc.Metropolis.__init__(self, stochastic, *args, **kwargs)

    def propose(self):
        """
        Propose a move.
        """
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        self.stochastic.value = \
                pymc.rlognormal(np.log(self.stochastic.value), tau)

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        cur_val = self.stochastic.value
        last_val = self.stochastic.last_value

        lp_for = pymc.lognormal_like(cur_val, mu=np.log(last_val), tau=tau)
        lp_bak = pymc.lognormal_like(last_val, mu=np.log(cur_val), tau=tau)

        if self.verbose > 1:
            print self._id + ': Hastings factor %f' % (lp_bak - lp_for)
        return lp_bak - lp_for

    @staticmethod
    def competence(s):
        """
        Tell PyMC that this step method is good for Lognormal, Exponential
        and Gamma random variables. In general, it should be good for positive
        random variables.
        """
        if isinstance(s, pymc.Lognormal):
            return 3
        elif isinstance(s, pymc.Exponential):
            return 3
        elif isinstance(s, pymc.Gamma):
            return 3
        else:
            return 0


class GaussianMixtureStep(pymc.Metropolis):
    """
    This is a test.
    """

    def __init__(self, stochastic, *args, **kwargs):
        """Initialize the object."""
        pymc.Metropolis.__init__(self, stochastic, *args, **kwargs)
        self._gmm = mixture.DPGMM(n_components=5, covariance_type='full')
        self._data = []

    def propose(self):
        """
        Propose a move.
        """
        x = self._gmm.sample()
        self.stochastic.value = x.flatten('F')
        return self._gmm.score(x)[0]

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """
        cur_val = np.atleast_2d(self.stochastic.value)
        last_val = np.atleast_2d(self.stochastic.value)

        lp_for = self._gmm.score(cur_val)[0]
        lp_bak = self._gmm.score(last_val)[0]

        if self.verbose > 1:
            print self._id + ': Hastings factor %f' % (lp_bak - lp_for)
        return lp_bak - lp_for
