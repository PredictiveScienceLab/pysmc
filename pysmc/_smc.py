__all__ = ['SMC']


from . import MCMCWrapper
import pymc
import numpy as np
from scipy.optimize import brentq
import math
import itertools


class SMC(object):
    """
    Use Sequential Monte Carlo (SMC) to sample from a distribution.

    In order to use the class you have to supply a pymc.MCMC class.
    The class should at least one node with a parameter called gamma
    that can range from 0 to 1. This class will start sampling from
    gamma == 0 and gradually move to gamma == 1.
    """

    # The number of particles of this CPU
    _my_num_particles = None

    # The number of particles to be used
    _num_particles = None

    # The logarithm of the weights
    _log_w = None

    # The current effective sample size
    _ess = None

    # The number of MCMC steps for each gamma
    _num_mcmc = None

    # The MCMC proposal
    _proposal = None

    # The particles
    _particles = None

    # The acceptance rate observed by each particle
    _acceptance_rate = None

    # The thresshold of the effective sample size (percentage)
    _ess_threshold = None

    # The reduction of the effective sample size per gamma step
    _ess_reduction = None

    # Do you want to adaptively select the MCMC proposal step?
    _adapt_proposal_step = None

    # The amount of verbosity
    _verbose = None

    # The MPI class
    _mpi = None

    # The MPI communicator
    _comm = None

    # The rank of the CPU
    _rank = None

    # The size of the CPU pool
    _size = None

    # The monte carlo sampler
    _mcmc_sampler = None

    # The particles
    _particles = None

    # The underlying MCMC sampler
    _mcmc_sampler = None

    # The observed random variable
    _gamma_rv = None

    # The true name of the gamma parameter
    _gamma_name = None

    @property
    def my_num_particles(self):
        """Get the my number of particles."""
        return self._my_num_particles

    @property
    def num_particles(self):
        """Get the number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value):
        """Set the number of particles."""
        if not isinstance(value, int):
            raise TypeError('The number of particles must be an int.')
        if value <= 0:
            raise ValueError('The number of particles must be positive.')
        self._my_num_particles = value / self.size
        self._num_particles = self.my_num_particles * self.size
        # Allocate memory
        self._allocate_memory()

    @property
    def log_w(self):
        """Get the log of the weights."""
        return self._log_w

    @property
    def ess(self):
        """Get the current effective sample size."""
        return self._ess

    @property
    def num_mcmc(self):
        """Get the number of MCMC steps per gamma."""
        return self._num_mcmc

    @num_mcmc.setter
    def num_mcmc(self, value):
        """Set the number of MCMC steps per gamma."""
        if not isinstance(value, int):
            raise TypeError('The number of MCMC steps must be an integer.')
        if value <= 0:
            raise ValueError('The number of MCMC steps must be positive.')
        self._num_mcmc = value

    @property
    def ess_threshold(self):
        """Get the threshold of the effective sample size."""
        return self._ess_threshold

    @ess_threshold.setter
    def ess_threshold(self, value):
        """Set the threshold of the effective sample size.

        It must be a number in (0, 1) representing a percentage of the
        total number of particles. If the effective sample size falls
        below this value, then the particles are automatically
        resampled.
        """
        if not isinstance(value, float):
            raise TypeError('The ESS threshold must be a float.')
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS threshold must be in (0, 1).')
        self._ess_threshold = value

    @property
    def ess_reduction(self):
        """Get the reduction of the effective sample size per gamma step."""
        return self._ess_reduction

    @ess_reduction.setter
    def ess_reduction(self, value):
        """Set the reduction of the effective sample size per gamma step.

        It must be a number in (0, 1) representing the desired reduction
        of the effective sample size when we perform a step in gamma.
        The next gamma will be selected adaptively so that the prescribed
        reduction is achieved.
        """
        value = float(value)
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS reduction must be in (0, 1).')
        self._ess_reduction = value

    @property
    def adapt_proposal_step(self):
        """Get the adapt proposal step flag."""
        return self._adapt_proposal_step

    @adapt_proposal_step.setter
    def adapt_proposal_step(self, value):
        """Set the adapt proposal step flag.

        If the adapt proposal step is set to True, the step
        of the MCMC proposal is adaptively set so that it remains
        between self.lowest_allowed_acceptance_rate and
        self.highest_allowed_acceptance_rate.
        """
        value = bool(value)
        self._adapt_proposal_step = value

    @property
    def verbose(self):
        """Get the verbosity flag."""
        if self.rank == 0:
            return self._verbose
        else:
            return 0

    @verbose.setter
    def verbose(self, value):
        """Set the verbosity flag."""
        value = int(value)
        self._verbose = value

    @property
    def comm(self):
        """Get the MPI communicator."""
        return self._comm

    @property
    def mpi(self):
        """Get access to the MPI class."""
        return self._mpi

    @comm.setter
    def comm(self, value):
        """Set the MPI communicator."""
        self._comm = value
        if self.use_mpi:
            self._rank = self.comm.Get_rank()
            self._size = self.comm.Get_size()
        else:
            self._rank = 0
            self._size = 1

    @property
    def use_mpi(self):
        """Are we using MPI or not?"""
        return self.comm is not None

    @property
    def rank(self):
        """Get the rank of this CPU."""
        return self._rank

    @property
    def size(self):
        """Get the size of the CPU pool."""
        return self._size

    @property
    def mcmc_sampler(self):
        """Get the MCMC sampler."""
        return self._mcmc_sampler

    @mcmc_sampler.setter
    def mcmc_sampler(self, value):
        """Set the MCMC sampler."""
        assert isinstance(value, pymc.MCMC)
        self._mcmc_sampler = value
        self._update_gamma_rv()

    def _update_gamma_rv(self):
        """Update the variable that points to the observed rv."""
        for rv in self.mcmc_sampler.nodes:
            if rv.parents.has_key(self.gamma_name):
                self._gamma_rv = rv
                break

    @property
    def gamma_rv(self):
        """Get the observed random variable."""
        return self._gamma_rv

    @property
    def gamma(self):
        return self._gamma_rv.parents[self.gamma_name]

    @gamma.setter
    def gamma(self, value):
        self._gamma_rv.parents[self.gamma_name] = value

    @property
    def gamma_name(self):
        """Get the true name of the gamma parameter."""
        return self._gamma_name

    @property
    def particles(self):
        """Get the particles."""
        return self._particles

    def _logsumexp(self, log_x):
        """Perform the log-sum-exp of the weights."""
        my_max_exp = log_x.max()
        if self.use_mpi:
            max_exp = self.comm.allreduce(my_max_exp, op=self.mpi.MAX)
        else:
            max_exp = my_max_exp
        my_sum = np.exp(log_x - max_exp).sum()
        if self.use_mpi:
            all_sum = self.comm.allreduce(my_sum)
        else:
            all_sum = my_sum
        return math.log(all_sum) + max_exp

    def _normalize(self, log_w):
        """Normalize the weights."""
        c = self._logsumexp(log_w)
        return log_w - c

    def _get_ess_at(self, log_w):
        """Calculate the ESS at given the log weights.

        Precondition:
        The weights are assumed to be normalized.
        """
        log_w_all = log_w
        if self.use_mpi:
            log_w_all = np.ndarray(self.num_particles)
            self.comm.Gather([log_w, self.mpi.DOUBLE],
                [log_w_all, self.mpi.DOUBLE])
        if self.rank == 0:
            ess = 1. / math.fsum(np.exp(2. * log_w_all))
        else:
            ess = None
        if self.use_mpi:
            ess = self.comm.bcast(ess)
        return ess

    def _get_log_of_weight_factor_at(self, gamma):
        """Return the log of the weight factor when going to the new gamma."""
        tmp_prev = np.zeros(self.my_num_particles)
        tmp_new = np.zeros(self.my_num_particles)
        old_gamma = self.gamma
        for i in range(self.my_num_particles):
            self.mcmc_sampler.set_state(self.particles[i])
            tmp_prev[i] = self.mcmc_sampler.logp
            self.gamma = gamma
            tmp_new[i] = self.mcmc_sampler.logp
            self.gamma = old_gamma
        return tmp_new - tmp_prev

    def _get_unormalized_weights_at(self, gamma):
        """Return the unormalized weights at a given gamma."""
        return self.log_w + self._get_log_of_weight_factor_at(gamma)

    def _get_ess_given_gamma(self, gamma):
        """Calculate the ESS at a given gamma.

        Return:
        The ess and the normalized weights corresponding to that
        gamma.
        """
        log_w = self._get_unormalized_weights_at(gamma)
        log_w_normalized = self._normalize(log_w)
        return self._get_ess_at(log_w_normalized)

    def _resample(self):
        """Resample the particles.

        Precondition:
        The weights are assumed to be normalized.
        """
        if self.verbose:
            print 'Resampling...'
        idx_list = []
        log_w_all = np.ndarray(self.num_particles)
        if self.use_mpi:
            self.comm.Gather([self.log_w, self.mpi.DOUBLE],
                [log_w_all, self.mpi.DOUBLE])
        else:
            log_w_all = self.log_w
        if self.rank == 0:
            births = np.random.multinomial(self.num_particles,
                                           np.exp(log_w_all))
            for i in xrange(self.num_particles):
                idx_list += [i] * births[i]
        if self.rank == 0:
            idx = np.array(idx_list, 'i')
        else:
            idx = np.ndarray(self.num_particles, 'i')
        if self.use_mpi:
            self.comm.Bcast([idx, self.mpi.INT])
            self.comm.barrier()
            old_particles = self._particles
            old_evaluated_state = self._evaluated_state
            self._particles = []
            for i in xrange(self.num_particles):
                to_whom = i / self.my_num_particles
                from_whom = idx[i] / self.my_num_particles
                if from_whom == to_whom and to_whom == self.rank:
                    my_idx = idx[i] % self.my_num_particles
                    self._particles.append(old_particles[my_idx].copy())
                elif to_whom == self.rank:
                    self._particles.append(self.comm.recv(source=from_whom, tag=i))
                elif from_whom == self.rank:
                    my_idx = idx[i] % self.my_num_particles
                    self.comm.send(old_particles[my_idx], dest=to_whom, tag=i)
                self.comm.barrier()
        else:
            self._particles = [self._particles[i].copy() for i in idx]
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = self.num_particles
        if self.verbose:
            print 'Done!'

    def _allocate_memory(self):
        """Allocates memory.

        Precondition:
        num_particles have been set.
        """
        if self.verbose:
            print 'Allocating memory...'
        # Allocate and initialize the weights
        self._log_w = np.ones(self.my_num_particles) * (-math.log(self.num_particles))
        self._particles = [None for i in range(self.my_num_particles)]
        if self.verbose:
            print 'Done!'

    def _tune(self):
        """Tune the parameters of the proposals.."""
        if self.verbose > 2:
            print 'Tuning the MCMC parameters.'
        for sm in self.mcmc_sampler.step_methods:
            if self.verbose > 2:
                print 'Tuning step method: ', str(sm)
            if sm.tune(verbose=self.verbose):
                if self.verbose > 2:
                    print 'Success!'
            else:
                if self.verbose > 2:
                    print 'Failure!'

    def _find_next_gamma(self, gamma):
        """Find the next gamma.

        Parameters
        ----------
        gamma       :   float
                        The next gamma is between the current one and ``gamma``.

        Returns
        -------
        The next gamma.

        """
        if self.verbose > 2:
            print 'Finding next gamma.'
        # Define the function whoose root we are seeking
        def f(test_gamma, args):
            ess_test_gamma = args._get_ess_given_gamma(test_gamma)
            return ess_test_gamma - args.ess_reduction * args.ess
        if f(gamma, self) > 0:
            if self.verbose > 2:
                print 'We can move directly to the target gamma...'
            return gamma
        else:
            # Solve for the optimal gamma using the bisection algorithm
            next_gamma = brentq(f, self.gamma, gamma, self)
            if self.use_mpi:
                self.comm.barrier()
            if self.verbose > 2:
                print 'Success!'
            return next_gamma

    def __init__(self, mcmc_sampler=None,
                 num_particles=10, num_mcmc=10,
                 ess_threshold=0.67,
                 ess_reduction=0.90,
                 adapt_proposal_step=True,
                 verbose=0,
                 mpi=None,
                 comm=None,
                 gamma_name='gamma'):
        """Initialize the object.

        Caution:
        The likelihood and the prior MUST be set!

        Keyword Arguments:
        mcmc_sampler    ---     An mcmc_sampler object.
        num_particles   ---     The number of particles.
        num_mcmc        ---     The number of MCMC steps per gamma.
        ess_threshold   ---     The ESS threshold below which resampling
                                takes place.
        ess_reduction   ---     The ESS reduction that adaptively specifies
                                the next gamma.
        adapt_proposal_step     ---     Adapt or not the proposal step by
                                        monitoring the acceptance rate.
        verbose     ---     Be verbose or not.
        mpi         ---     set the mpi class.
        comm        ---     Set this to the MPI communicator (If you want to use
                            mpi).
        gamma_name  ---     The name you wish to use for gamma.

        Caution: The likelihood and the prior must be specified together!
        """
        assert isinstance(gamma_name, str)
        self._gamma_name = gamma_name
        self.mcmc_sampler = mcmc_sampler
        self._mpi = mpi
        if self.mpi is not None and comm is None:
            self.comm = self.mpi.COMM_WORLD
        elif comm is None:
            self.comm = None
        else:
            raise RunTimeError('To use MPI you have to specify '
                               + 'the mpi variable.')
        assert isinstance(mcmc_sampler, pymc.MCMC)
        self._mcmc_sampler = MCMCWrapper(mcmc_sampler)
        self.num_particles = num_particles
        self.num_mcmc = num_mcmc
        self.ess_threshold = ess_threshold
        self.ess_reduction = ess_reduction
        self.verbose = verbose
        self.adapt_proposal_step = adapt_proposal_step
        self._max_proposal_dt = 0.5

    def initialize(self, gamma=0., particles=None, num_mcmc_per_particle=1000):
        """
        Initialize SMC at a particular ``gamma``.
        
        The method has basically three ways of initializing the particles:
        1) If ``particles`` is not ``None``, then it is assumed to contain the
           particles at the corresponding value of ``gamma``.
        2) If ``particles`` is ``None`` and the MCMC sampler class has a method
           called ``draw_from_prior()`` that works, then it is called to
           initialize the particles.
        3) In any other case, MCMC sampling is used to initialize the particles.
           We are assuming that the MCMC sampler has already been tuned for
           that particular gamma and that a sufficient burning period has past.
           Then we record the current state as the first particle, we sample
           ``num_mcmc_per_particle`` times and record the second particle, and
           so on.
        
        Parameters
        ----------
        gamma                   :   float
                                    The initial ``gamma`` parameter. It must, of
                                    course, be within the right range of
                                    ``gamma``.
        particles               :   dict of MCMC states
                                    A dictionary of MCMC states representing
                                    the particles.
        num_mcmc_per_particle   :   int
                                    This parameter is ignored if ``particles``
                                    is not ``None``. If the only way to
                                    initialize the particles is to use MCMC,
                                    then this is the number of of mcmc samples
                                    we drop before getting an MCMC particle.
                                    
        """
        # Set gamma
        self.gamma = gamma
        # Set the weights and ESS
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = float(self.num_particles)
        if particles is not None:
            assert len(particles) == self.num_particles
            self._particles = particles
            # TODO: Fix if using mpi
            return
        self.particles[0] = self.mcmc_sampler.get_state()
        try:
            if self.verbose > 1:
                print 'Attempting to sampler from the prior...'
            for i in range(1, self.my_num_particles):
                self.mcmc_sampler.draw_from_prior()
                self.particles[i] = self.mcmc_sampler.get_state()
            if self.verbose > 1:
                print 'Success!'
        except AttributeError:
            if self.verbose > 1:
                print 'Failure!'
                print 'Doing MCMC to initialize the particles.'
            for i in range(1, self.my_num_particles):
                self.mcmc_sampler.sample(num_mcmc_per_particle)
                self.particles[i] = self.mcmc_sampler.get_state()
            if self.verbose > 1:
                print 'Success!'
    
    def move_to(self, gamma):
        """
        Move the current particle approximation to ``gamma``.
        
        Paremeters
        ----------
        gamma       :   float
                        The new ``gamma`` you wish to reach.
        
        Precondition
        ------------
        There is already a valid particle approximation. See
        ``SMC.initialize()`` for ways of doing this.
        
        """
        while self.gamma < gamma:
            if self.adapt_proposal_step:
                self._tune()
            new_gamma = self._find_next_gamma(gamma)         
            log_w = self._get_unormalized_weights_at(new_gamma)
            self._log_w = self._normalize(log_w)
            self._ess = self._get_ess_at(self.log_w)
            self.gamma = new_gamma
            if self.ess < self.ess_threshold * self.num_particles:
                self._resample()
            if self.verbose > 0:
                print 'Performing MCMC at gamma = ', self.gamma
            for i in range(self.my_num_particles):
                self.mcmc_sampler.set_state(self.particles[i])
                self.mcmc_sampler.sample(self.num_mcmc)
                self.particles[i] = self.mcmc_sampler.get_state()
            if self.verbose > 1:
                print 'Success!'

    def get_particle_approximation(self, name):
        """
        Get the particle approximation of the distribution.
        """
        w = np.exp(self.log_w)
        r = [self.particles[i]['stochastics'][name]
             for i in range(self.num_particles)]
        return w, r
