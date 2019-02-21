"""

.. _step_methods:

============
Step Methods
============

In PySMC we define a few step methods for the Metropolis-Hastings algorithm
that extend the capabilities of PyMC.

Here is a list of what we offer:

"""


__all__ = ['RandomWalk', 'LognormalRandomWalk', 'DiscreteRandomWalk',
           'GaussianMixtureStep']


import pymc as pm
import numpy as np
from numpy.random import poisson as rpoisson
from sklearn import mixture


class RandomWalk(pm.Metropolis):

    _adapt_increase_factor = None

    _adapt_decrease_factor = None

    _adapt_upper_ac_rate = None

    _adapt_lower_ac_rate = None

    _min_adaptive_scale_factor = None

    _max_adaptive_scale_factor = None

    _adaptive_scale_factor = None

    _STATE_VARIABLES = None

    _accepted = None

    _rejected = None

    @property
    def STATE_VARIABLES(self):
        """
        Get the names of the variables required to store the state of the
        object.
        """
        return self._STATE_VARIABLES

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
                 adapt_increase_factor=1.1,
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
        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)
        #super(RandomWalk, self).__init__(stochastic, *args, **kwargs)
        self._adaptive_scale_factor = self.adaptive_scale_factor
        self._STATE_VARIABLES = ['_adapt_increase_factor',
                                 '_adapt_decrease_factor',
                                 '_adapt_upper_ac_rate',
                                 '_adapt_lower_ac_rate',
                                 '_min_adaptive_scale_factor',
                                 '_max_adaptive_scale_factor',
                                 '_adaptive_scale_factor',
                                 'adaptive_scale_factor',
                                 'proposal_sd',
                                 '_old_accepted',
                                 '_old_rejected'
                                ]
        self._old_accepted = 0.
        self._old_rejected = 0.
        self.proposal_sd = 1e-1
        self.adaptive_scale_factor = 1.

    def tune(self, pa=None, comm=None, divergence_threshold=1e10, verbose=0):
        ac = self.get_acceptance_rate(comm=comm)
        if ac == -1:
            return False
        self.reset_counters()
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
            print(('\n\t\tadaptive_scale_factor:', self.adaptive_scale_factor))
        return True

    @staticmethod
    def competence(s):
        """
        Tell PyMC that this step method is better than its random walk.
        """
        return 3

    def get_params(self, comm=None):
        """
        Get the state of the step method.
        """
        state = {}
        for var in self.STATE_VARIABLES:
            state[var] = getattr(self, var)
        if comm is not None:
            state['_old_accepted'] = comm.allreduce(state['_old_accepted'])
            state['_old_rejected'] = comm.allreduce(state['_old_rejected'])
            size = comm.Get_size()
            state['_old_accepted'] /= size
            state['_old_rejected'] /= size
        return state

    def set_params(self, state):
        """
        Set the state from a dictionary.
        """
        for var in list(state.keys()):
            setattr(self, var, state[var])
        self._old_accepted *= -1.
        self._old_rejected *= -1.

    def reset_counters(self):
        """
        Reset the counters that count accepted and rejected steps.
        """
        self._old_accepted = self.accepted
        self._old_rejected = self.rejected

    def get_acceptance_rate(self, comm=None):
        """
        Get the acceptance rate of the step method sm.
        """
        accepted = self.accepted - self._old_accepted
        rejected = self.rejected - self._old_rejected
        if comm is not None:
            accepted = comm.allreduce(accepted)
            rejected = comm.allreduce(rejected)
        if (accepted + rejected) == 0.:
            return -1
        return accepted / (accepted + rejected)


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
    @staticmethod
    def competence(s):
        """
        Tell PyMC that this step method is better than its random walk.
        """
        if isinstance(s, pm.Lognormal) or \
                isinstance(s, pm.Exponential):
            return 5

    def propose(self):
        """
        Propose a move.
        """
        tau = (self.adaptive_scale_factor * self.proposal_sd) ** 2
        self.stochastic.value = \
                pm.rlognormal(np.log(self.stochastic.value), tau)

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """
        tau = (self.adaptive_scale_factor * self.proposal_sd) ** 2
        cur_val = self.stochastic.value
        last_val = self.stochastic.last_value

        lp_for = pm.lognormal_like(cur_val, mu=np.log(last_val), tau=tau)
        lp_bak = pm.lognormal_like(last_val, mu=np.log(cur_val), tau=tau)

        if self.verbose > 1:
            print((self._id + ': Hastings factor %f' % (lp_bak - lp_for)))
        return lp_bak - lp_for


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

    # The mean when we do scale
    _mean = None

    # The std when we do scale
    _std = None

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
                 adapt_lower_ac_rate=0.5,
                 covariance_type='full',
                 n_components=50,
                 n_iter=1000,
                 *args, **kwargs):
        """Initialize the object."""
        RandomWalk.__init__(self,
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
        self._STATE_VARIABLES += ['_covariance_type',
                                  '_n_components',
                                  '_n_iter',
                                  '_tuned',
                                  '_gmm',
                                  '_mean',
                                  '_std']
        self._accepted = 0.
        self._rejected = 0.

    def propose(self):
        """
        Propose a move.
        """
        if not self._tuned:
            return super(GaussianMixtureStep, self).propose()
        x = self.gmm.sample()
        self.stochastic.value = x.flatten('F')

    def hastings_factor(self):
        """
        Compute the hastings factor.
        """

        if not self._tuned:
            return super(GaussianMixtureStep, self).hastings_factor()

        cur_val = np.atleast_2d(self.stochastic.value)
        last_val = np.atleast_2d(self.stochastic.last_value)

        lp_for = self._gmm.score(cur_val)[0]
        lp_bak = self._gmm.score(last_val)[0]

        if self.verbose > 1:
            print((self._id + ': Hastings factor %f' % (lp_bak - lp_for)))
        return lp_bak - lp_for

    def tune(self, pa=None, comm=None, divergence_threshold=1e10, verbose=0):
        """
        Tune the step...
        """
        if pa is None:
            raise RuntimeError('This step method works only in pysmc.')
        ac = self.get_acceptance_rate(comm=comm)
        if ac == -1:
            return False
        self.reset_counters()
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
                    for i in range(pa.num_particles)]
            data = np.array(data, dtype='float')
            if data.ndim == 1:
                data = np.atleast_2d(data).T
            self._gmm = mixture.DPGMM(n_components=self.n_components,
                                     covariance_type=self.covariance_type,
                                     n_iter=self.n_iter)
            self.gmm.fit(data)
            Y_ = self.gmm.predict(data)
            n_comp = 0
            for i in range(self.n_components):
                if np.any(Y_ == i):
                    n_comp += 1
            self._gmm = mixture.GMM(n_components=n_comp,
                                    covariance_type=self.covariance_type,
                                    n_iter=self.n_iter)
            self.gmm.fit(data)
            Y_ = self.gmm.predict(data)
            if verbose >= 2:
                for i, (mean, covar) in enumerate(zip(
                        self.gmm.means_, self.gmm._get_covars())):
                    if not np.any(Y_ == i):
                        continue
                    print(('\n', mean, covar))
        if use_mpi:
            self._gmm = comm.bcast(self._gmm)
        self.gmm.covars_ = self.gmm._get_covars()
        self._tuned = True
        return True


class DiscreteRandomWalk(RandomWalk):
    """
    This is a step method class that is good for discrete random variables.

    Good only for non-negative discrete random variables.

    **Base class:** :class:`pysmc.RandomWalk`
    """

    def __init__(self, stochastic,
                 prop_dist='poisson',
                 adapt_increase_factor=1.3,
                 adapt_decrease_factor=0.7,
                 adapt_upper_ac_rate=0.7,
                 adapt_lower_ac_rate=0.3,
                 min_adaptive_scale_factor=1e-32,
                 max_adaptive_scale_factor=1e99,
                 *args, **kwargs):
        """Initialize the object."""
        super(DiscreteRandomWalk, self).__init__(
                            stochastic,
                            adapt_increase_factor=adapt_increase_factor,
                            adapt_decrease_factor=adapt_decrease_factor,
                            adapt_upper_ac_rate=adapt_upper_ac_rate,
                            adapt_lower_ac_rate=adapt_lower_ac_rate,
                            min_adaptive_scale_factor=min_adaptive_scale_factor,
                            max_adaptive_scale_factor=max_adaptive_scale_factor,
                            *args, **kwargs)
        self.prop_dist = prop_dist
        self._STATE_VARIABLES.append('prop_dist')

    def propose(self):
        """
        Propose a move.
        """
        if self.prop_dist == 'poisson':
            k = self.stochastic.value.shape
            new_val = self.stochastic.value + rpoisson(
                    self.adaptive_scale_factor * self.proposal_sd) * (
                        -np.ones(k)) ** (np.random.random(k) > 0.5) 
            self.stochastic.value = np.abs(new_val)
        elif self.prop_dist == 'prior':
            self.stochastic.random()


# Assign methods to the registry of pymc
pm.StepMethodRegistry = []
for method in __all__:
    pm.StepMethodRegistry.append(eval(method))
