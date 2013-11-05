"""

.. _step_methods:

============
Step Methods
============

In PySMC we define a few step methods for the Metropolis-Hastings algorithm
that extend the capabilities of PyMC.

Here is a list of what we offer:

"""


__all__ = ['RandomWalk', 'LognormalRandomWalk', 'GaussianMixtureStep']


import pymc as pm
import numpy as np
from sklearn import mixture


class RandomWalk(pm.Metropolis):

    _adapt_increase_factor = None

    _adapt_decrease_factor = None

    _adapt_upper_ac_rate = None

    _adapt_lower_ac_rate = None

    _min_adaptive_scale_factor = None

    _max_adaptive_scale_factor = None

    _adaptive_scale_factor = None

    @property
    def adapt_increase_factor(self):
        return self._adapt_increase_factor

    @property
    def adapt_decrease_factor(self):
        return self._adapt_decrease_factor

    @property
    def adapt_upper_ac_rate(self):
        return self._adapt_upper_ac_rate

    @property
    def adapt_lower_ac_rate(self):
        return self._adapt_lower_ac_rate

    @property
    def min_adaptive_scale_factor(self):
        return self._min_adaptive_scale_factor

    @property
    def max_adaptive_scale_factor(self):
        return self._max_adaptive_scale_factor

    def __init__(self, stochastic,
                 adapt_increase_factor=1.3,
                 adapt_decrease_factor=0.7,
                 adapt_upper_ac_rate=0.7,
                 adapt_lower_ac_rate=0.3,
                 min_adaptive_scale_factor=1e-32,
                 max_adaptive_scale_factor=1e99,
                 *args, **kwargs):
        """Initialize the object."""
        assert adapt_decrease_factor <= 1.
        self._adapt_decrease_factor = adapt_decrease_factor
        assert adapt_increase_factor >= 1.
        self._adapt_increase_factor = adapt_increase_factor
        assert adapt_upper_ac_rate <= 1.
        assert adapt_lower_ac_rate >= 0.
        assert adapt_lower_ac_rate <= adapt_upper_ac_rate
        self._adapt_upper_ac_rate = adapt_upper_ac_rate
        self._adapt_lower_ac_rate = adapt_lower_ac_rate
        assert min_adaptive_scale_factor > 0.
        assert min_adaptive_scale_factor <= max_adaptive_scale_factor
        self._min_adaptive_scale_factor = min_adaptive_scale_factor
        self._max_adaptive_scale_factor = max_adaptive_scale_factor
        super(RandomWalk, self).__init__(stochastic, *args, **kwargs)
        self._adaptive_scale_factor = self.adaptive_scale_factor

    def tune(self, ac, pa, comm=None, divergence_threshold=1e10, verbose=0):
        if ac == -1:
            return False
        use_mpi = comm is not None
        rank = comm.Get_rank() if use_mpi else 0
        if ac <= self.adapt_lower_ac_rate:
            self._adaptive_scale_factor = max(self._adaptive_scale_factor *
                                             self.adapt_decrease_factor,
                                             self.min_adaptive_scale_factor)
        elif ac >= self.adapt_upper_ac_rate:
            self._adaptive_scale_factor = min(self._adaptive_scale_factor *
                                             self.adapt_increase_factor,
                                             self.max_adaptive_scale_factor)
        self.adaptive_scale_factor = self._adaptive_scale_factor
        if verbose >= 2 and rank == 0:
            print '\n\t\tadaptive_scale_factor:', self.adaptive_scale_factor
        self.accepted = 0.
        self.rejected = 0.
        return True

    def competence(s):
        """
        Tell PyMC that this step method is good for Lognormal, Exponential
        and Gamma random variables. In general, it should be good for positive
        random variables.
        """
        return 2


class LognormalRandomWalk(RandomWalk):
    """
    This is a step method class that is good for positive random variables.
    It is a essentially a random walk in the logarithmic scale.

    **Base class:** :class:`pm.Metropolis`
    """

    def __init__(self, stochastic,
                 adapt_increase_factor=1.3,
                 adapt_decrease_factor=0.7,
                 adapt_upper_ac_rate=0.7,
                 adapt_lower_ac_rate=0.3,
                 min_adaptive_scale_factor=1e-32,
                 max_adaptive_scale_factor=1e99,
                 *args, **kwargs):
        """Initialize the object."""
        super(LognormalRandomWalk, self).__init__(
                            stochastic,
                            adapt_increase_factor=adapt_increase_factor,
                            adapt_decrease_factor=adapt_decrease_factor,
                            adapt_upper_ac_rate=adapt_upper_ac_rate,
                            adapt_lower_ac_rate=adapt_lower_ac_rate,
                            min_adaptive_scale_factor=min_adaptive_scale_factor,
                            max_adaptive_scale_factor=max_adaptive_scale_factor,
                            *args, **kwargs)

    def propose(self):
        """
        Propose a move.
        """
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        self.stochastic.value = \
                pm.rlognormal(np.log(self.stochastic.value), tau)

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        cur_val = self.stochastic.value
        last_val = self.stochastic.last_value

        lp_for = pm.lognormal_like(cur_val, mu=np.log(last_val), tau=tau)
        lp_bak = pm.lognormal_like(last_val, mu=np.log(cur_val), tau=tau)

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
        if isinstance(s, pm.Lognormal):
            return 3
        elif isinstance(s, pm.Exponential):
            return 3
        elif isinstance(s, pm.Gamma):
            return 3
        else:
            return 0


class GaussianMixtureStep(RandomWalk):
    """
    This is a test.
    """

    # A gaussian mixtures model
    _gmm = None

    # Covariance type
    _covariance_type = None

    # The valid covariance types
    _VALID_COVARIANCE_TYPES = None

    # Maximum number of components for the Gaussian mixture
    _n_components = None

    # The number of iterations we should do while training the
    # Gaussian mixture
    _n_iter = None

    @property
    def n_iter(self):
        """
        Get the number of Gaussian mixture training iterations.
        """
        return self._n_iter

    @property
    def n_components(self):
        """
        Get the maximum number of Gaussian mixture components.
        """
        return self._n_components

    @property
    def VALID_COVARIANCE_TYPES(self):
        """
        Get the valid covariance types.
        """
        return self._VALID_COVARIANCE_TYPES

    @property
    def covariance_type(self):
        """
        Get the covariance type.
        """
        return self._covariance_type

    @property
    def gmm(self):
        """
        Get the Gaussian process model.
        """
        return self._gmm

    def __init__(self, stochastic,
                 adapt_upper_ac_rate=1.,
                 adapt_lower_ac_rate=0.3,
                 covariance_type='full',
                 n_components=5,
                 n_iter=1000,
                 *args, **kwargs):
        """Initialize the object."""
        super(GaussianMixtureStep, self).__init__(
                            stochastic,
                            adapt_upper_ac_rate=adapt_upper_ac_rate,
                            adapt_lower_ac_rate=adapt_lower_ac_rate,
                            *args, **kwargs)
        self._VALID_COVARIANCE_TYPES = ['diag', 'full', 'spherical', 'tied']
        assert covariance_type in self.VALID_COVARIANCE_TYPES
        self._covariance_type = covariance_type
        n_components = int(n_components)
        assert n_components >= 1
        self._n_components = n_components
        n_iter = int(n_iter)
        assert n_iter >= 0
        self._n_iter = n_iter
        self._tuned = False

    def propose(self):
        """
        Propose a move.
        """
        if not self._tuned:
            return super(GaussianMixtureStep, self).propose()
        x = self.gmm.sample() * self._std + self._mean
        self.stochastic.value = x.flatten('F')

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """

        if not self._tuned:
            return super(GaussianMixtureStep, self).hastings_factor()

        cur_val = (np.atleast_2d(self.stochastic.value) - self._mean) / self._std
        last_val = (np.atleast_2d(self.stochastic.value) - self._mean) / self._std

        lp_for = self._gmm.score(cur_val)[0]
        lp_bak = self._gmm.score(last_val)[0]

        if self.verbose > 1:
            print self._id + ': Hastings factor %f' % (lp_bak - lp_for)
        return lp_bak - lp_for

    def tune(self, ac, pa, comm=None, divergence_threshold=1e10, verbose=0):
        """
        Tune the step...
        """
        if ac == -1:
            return False
        self.accepted = 0.
        self.rejected = 0.
        if (self._tuned and
            ac >= self.adapt_lower_ac_rate and
            ac <= self.adapt_upper_ac_rate):
            return False
        use_mpi = comm is not None
        if use_mpi:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
        pa = pa.gather()
        # Only the root should run train the mixture
        if rank == 0:
            pa.resample()
            data = [pa.particles[i]['stochastics'][self.stochastic.__name__]
                    for i in xrange(pa.num_particles)]
            data = np.array(data)
            if data.ndim == 1:
                data = np.atleast_2d(data).T
            # Scale the data
            self._mean = data.mean(axis=0)
            self._std = data.std(axis=0)
            data = (data - self._mean) / self._std
            self._gmm = mixture.DPGMM(n_components=self.n_components,
                                      covariance_type=self.covariance_type,
                                      n_iter=self.n_iter,
                                      min_covar=1e-20)
            self.gmm.fit(data)
            if verbose >= 2:
                print self._mean
                print self._std
                Y_ = self.gmm.predict(data)
                for i, (mean, covar) in enumerate(zip(
                        self.gmm.means_, self.gmm._get_covars())):
                    if not np.any(Y_ == i):
                        continue
                    print '\n', mean, covar
        else:
            self._mean = None
            self._std = None
        self._gmm = comm.bcast(self._gmm)
        self._mean = comm.bcast(self._mean)
        self._std = comm.bcast(self._std)
        self.gmm.covars_ = self.gmm._get_covars()
        self._tuned = True
        return True
