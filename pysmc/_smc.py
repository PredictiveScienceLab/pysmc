"""

.. _smc:

+++
SMC
+++

:class:`pysmc.SMC` is the class that makes everything happen.

Here is a complete reference of the public members:
"""


__all__ = ['SMC']


from . import MCMCWrapper
from . import get_var_from_particle_list
import pymc
import numpy as np
from scipy.optimize import brentq
import math
import itertools
import sys
import warnings


class SMC(object):
    """
    Use Sequential Monte Carlo (SMC) to sample from a distribution.

    :param mcmc_sampler:        An mcmc_sampler object.
    :type mcmc_sampler:         :class:`pymc.MCMC`
    :param num_particles:       The number of particles.
    :type num_particles:        int
    :param num_mcmc:            The number of MCMC steps per gamma.
    :type num_mcmc:             int
    :param ess_threshold:       The ESS threshold below which resampling
                                takes place.
    :type ess_threshold:        float
    :param ess_reduction:       The ESS reduction that adaptively specifies
                                the next ``gamma``.
    :type ess_reduction:        float
    :param adapt_proposal_step: Adapt or not the proposal step by
                                monitoring the acceptance rate.
    :type adapt_proposal_step:  bool
    :param verbose:             How much output to print (1, 2 and 3).
    :type verbose:              int
    :param gamma_name:          The name with which the ``gamma`` parameter is
                                refered to in your :mod:`pymc` model. The
                                default value is ``'gamma'``, but you can
                                change it to whatever you want.
    :type gamma_name:           str
    :param mpi:                 The MPI class (see :mod:`mpi4py` and
                                :ref:`mpi_example`). If ``None``, then no
                                parallelism is used.
    :param comm:                Set this to the MPI communicator. If ``None``,
                                then ``mpi.COMM_WORLD`` is used.
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

    # A database containing all the particles at all gammas
    _db = None

    # Count the total number of MCMC samples taken so far
    _total_num_mcmc = None

    @property
    def my_num_particles(self):
        """
        :getter:    The number of particles associated with each process.
        :type:      int

        .. note::

            If not using MPI, then it is the same as
            :meth:`pysmc.SMC.num_particles`. Otherwise is it is equal to
            ``num_particles / size``, where ``size`` is the total number of MPI
            processes.

        """
        return self._my_num_particles

    @property
    def num_particles(self):
        """
        :getter:    Get the number of particles.
        :setter:    Set the number of particles. All data in the instant of the
                    class will be lost if this is called.
        :type:      int
        :raises:    :exc:`exceptions.ValueError`
        """
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value):
        """Set the number of particles."""
        value = int(value)
        if value <= 0:
            raise ValueError('The number of particles must be positive.')
        self._my_num_particles = value / self.size
        self._num_particles = self.my_num_particles * self.size
        self._allocate_memory()

    @property
    def log_w(self):
        """
        :getter:    The logarithm of the weights of the particles.
        :type:      1D :class:`numpy.ndarray`
        """
        return self._log_w

    @property
    def ess(self):
        """
        :getter:    The current Effective Sample Size.
        :type:      float
        """
        return self._ess

    @property
    def num_mcmc(self):
        """
        :getter:    Get the number of MCMC steps per SMC step.
        :setter:    Set the number of MCMC steps per SMC step.
        :type:      int
        :raises:    :exc:`exceptions.ValueError`
        """
        return self._num_mcmc

    @num_mcmc.setter
    def num_mcmc(self, value):
        """Set the number of MCMC steps per gamma."""
        value = int(value)
        if value <= 0:
            raise ValueError('The number of MCMC steps must be positive.')
        self._num_mcmc = value

    @property
    def ess_threshold(self):
        """
        The threshold of the Effective Sample Size is a number between 0 and 1
        representing the percentage of the total number of particles. If the
        Effective Sample Size falls bellow ``ess_threshold * num_particles``,
        then the particles are automatically resampled.

        :getter:    Get the threshold of the Effective Sample Size.
        :setter:    Set the threshold of the Effective Sample Size.
        :type:      float
        :raises:    :exc:`exceptions.ValueError`
        """
        return self._ess_threshold

    @ess_threshold.setter
    def ess_threshold(self, value):
        """Set the threshold of the effective sample size."""
        value = float(value)
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS threshold must be in (0, 1).')
        self._ess_threshold = value

    @property
    def ess_reduction(self):
        """
        It is a number between 0 and 1 representing the desired
        percent reduction
        of the effective sample size when we perform a SMC step.
        The next ``gamma`` will be selected adaptively so that the prescribed
        reduction is achieved.

        :getter:    Get the reduction of the Effective Sample Size per SMC
                    step.
        :setter:    Set the reduction of the Effective Sample Size per SMC
                    step.
        :type:      float
        :raises:    :exc:`exceptions.ValueError`
        """
        return self._ess_reduction

    @ess_reduction.setter
    def ess_reduction(self, value):
        """Set the reduction of the effective sample size per SMC step."""
        value = float(value)
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS reduction must be in (0, 1).')
        self._ess_reduction = value

    @property
    def adapt_proposal_step(self):
        """
        If the ``adapt proposal step`` is set to ``True``, each of the step
        methods of the underlying :class:`pymc.MCMC` class are adaptively
        tuned by monitoring the acceptance rate.

        :getter:    Get the adapt flag.
        :setter:    Set the adapt flag.
        :type:      bool
        """
        return self._adapt_proposal_step

    @adapt_proposal_step.setter
    def adapt_proposal_step(self, value):
        """Set the adapt flag."""
        value = bool(value)
        self._adapt_proposal_step = value

    @property
    def verbose(self):
        """

        Specify the amount of output printed by the class. There are three
        levels:
            + 0:    Print nothing.
            + 1:    Print info from methods you call.
            + 2:    Print info from methods the methods you call call...
            + 3:    Guess what...

        :getter:    Get the verbosity flag.
        :setter:    Set the verbosity flag.
        :type:      int
        """
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
        """
        The MPI communicator.

        :getter:    Get the MPI communicator.
        :setter:    Set the MPI communicator.
        :type:      We do not check it.
        """
        return self._comm

    @property
    def mpi(self):
        """
        The MPI class.

        :getter:    Get the MPI class.
        :setter:    Set the MPI class.
        :type:      We do not check it.
        """
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
        """
        Check if MPI is being used.

        :returns:   ``True`` if MPI is used and ``False`` otherwise.
        :rtype:     bool
        """
        return self.comm is not None

    @property
    def rank(self):
        """
        The rank of the process that calls this method.

        :getter:    Get the rank of the process that calls this method.
        :type:      int
        """
        return self._rank

    @property
    def size(self):
        """
        The total number of MPI processes.

        :getter:    Get the total number of MPI processes.
        :type:      int
        """
        return self._size

    @property
    def mcmc_sampler(self):
        """
        The underlying :class:`pymc.MCMC` object.

        :getter: Get the underlying MCMC object.
        :setter: Set the underlying MCMC object.
        :raises: :exc:`exceptions.TypeError`
        :type: :class:`pymc.MCMC`
        """
        return self._mcmc_sampler

    @property
    def db(self):
        """
        The database containing info about all the particles we visited.

        :getter:    Get the database.
        :type:      dict
        """
        return self._db

    @mcmc_sampler.setter
    def mcmc_sampler(self, value):
        """Set the MCMC sampler."""
        if not isinstance(value, pymc.MCMC):
            raise TypeError('mcmc_sampler must be a pymc.MCMC object!')
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
        """
        The random variable of the :mod:`pymc` model that has a parameter named
        ``gamma_name``.

        :getter:    Get the value of the ``gamma_name`` parameter.
        :type:      :class:`pymc.Stochastic`
        """
        return self._gamma_rv

    @property
    def gamma(self):
        """
        The value of the ``gamma_name`` parameter.

        :getter:    Get the value of the ``gamma_name`` parameter.
        :setter:    Set the value of the ``gamma_name`` parameter.
        :type:      float
        """
        return self._gamma_rv.parents[self.gamma_name]

    @gamma.setter
    def gamma(self, value):
        self._gamma_rv.parents[self.gamma_name] = value

    @property
    def gamma_name(self):
        """
        The true name of the gamma parameter in the :mod:`pymc` model.

        :getter:    Get the name of the gamma parameter.
        :setter:    Set the name of the gamma parameter.
        :type:      str
        """
        return self._gamma_name

    @gamma_name.setter
    def gamma_name(self, value):
        """Set the true name of the gamma parameter."""
        self._gamma_name = gamma_name
        if self._mcmc_sampler is not None:
            self._update_gamma_rv()

    @property
    def particles(self):
        """
        The SMC particles.

        :getter:    Get the SMC particles.
        :setter:    Set the SMC particles.
        :type:      list of whatever objects your method supports.
        :raises:    :exc:`exceptions.ValueError`

        .. note::

            When using MPI and you must assign to the setter only the particles
            that pertain to the process that calls it. That is, you are
            responsible for scattering the particles from the root to everybody
            else. Hopefully, you will get an :exc:`exceptions.ValueError` if
            you get this wrong.

        """
        return self._particles

    @particles.setter
    def particles(self, value):
        """Set the SMC particles."""
        if not len(value) == self.my_num_particles:
            raise ValueError('The number of particles you specified is wrong.')
        self._particles = particles

    @property
    def weights(self):
        """
        The weights of the SMC particles.

        :getter:    Get the SMC weights.
        :setter:    Set the SMC weights.
        :type:      1D :class:`numpy.ndarray`

        .. note::

            When using MPI, the same concerns as in :attr:`pysmc.SMC.particles`
            hold here.

        """
        return np.exp(self.log_w)

    @weights.setter
    def weights(self, value):
        """Set the SMC weights."""
        value = np.array(value)
        self._log_w = self._normalize(np.log(value))

    @property
    def total_num_mcmc(self):
        """
        The total number of MCMC steps performed so far.
        This is zeroed, everytime you call :meth:`pysmc.SMC.initialize()`.

        :getter:    The total number of MCMC steps performed so far.
        :type:      int
        """
        return self._total_num_mcmc

    def _add_current_state_to_db(self):
        """Add the current state to the database."""
        if self.verbose > 1:
            print '\t-adding current state to database'
        if self._db is None:
            if self.verbose > 1:
                print '\t-database does not exist'
                print '\t-creating database'
            self._db = dict(gamma_name = self.gamma_name, data = {})
        if not self.db['gamma_name'] == self.gamma_name:
            warnings.warn(
            'Database \'gamma_name\' does not match self.gamma_name')
        if self.db['data'].has_key(self.gamma):
            warnings.warn(
            'Database already contains a record for %1.2f.' % self.gamma
            + 'It will be replaced!'
            )
        state = dict()
        state['weights'] = np.exp(self.log_w)
        state['particles'] = self.particles
        self.db['data'][self.gamma] = state

    def _check_if_gamma_is_in_db(self, gamma):
        """Check if the particular ``gamma`` is in the database.

        :raises:    :exc:`exceptions.RuntimeError`
        """
        if not self.db['data'].has_key(gamma):
            raise RuntimeError(
            'Database does not contain a record for %s = %1.2f.'
            % (self.gamma_name, gamma))

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

        Precondition
        ------------
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

        Returns
        -------
        The ess and the normalized weights corresponding to that
        gamma.
        """
        log_w = self._get_unormalized_weights_at(gamma)
        log_w_normalized = self._normalize(log_w)
        return self._get_ess_at(log_w_normalized)

    def _resample(self):
        """Resample the particles.

        Precondition
        ------------
        The weights are assumed to be normalized.
        """
        if self.verbose > 1:
            sys.stdout.write('- resampling: ')
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
                    self._particles.append(self.comm.recv(
                                                       source=from_whom, tag=i))
                elif from_whom == self.rank:
                    my_idx = idx[i] % self.my_num_particles
                    self.comm.send(old_particles[my_idx], dest=to_whom, tag=i)
                self.comm.barrier()
        else:
            self._particles = [self._particles[i].copy() for i in idx]
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = self.num_particles
        if self.verbose > 1:
            sys.stdout.write('SUCCESS\n')

    def _allocate_memory(self):
        """Allocate memory.

        Precondition
        ------------
        ``num_particles`` have been set.
        """
        # Allocate and initialize the weights
        self._log_w = (np.ones(self.my_num_particles)
                       * (-math.log(self.num_particles)))
        self._particles = [None for i in range(self.my_num_particles)]

    def _tune(self):
        """Tune the parameters of the proposals.."""
        # TODO: Make sure this actually works!
        if self.verbose > 1:
            print '- tuning the MCMC parameters:'
        for sm in self.mcmc_sampler.step_methods:
            if self.verbose > 1:
                sys.stdout.write('\t- tuning step method: %s' % str(sm))
            if sm.tune(verbose=self.verbose):
                if self.verbose > 1:
                    sys.stdout.write('\n\t\tSUCCESS\n')
            else:
                if self.verbose > 1:
                    sys.stdout.write('\n\t\tFAILURE\n')

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
        if self.verbose > 1:
            print '- finding next gamma.'
        # Define the function whoose root we are seeking
        def f(test_gamma, args):
            ess_test_gamma = args._get_ess_given_gamma(test_gamma)
            return ess_test_gamma - args.ess_reduction * args.ess
        if f(gamma, self) > 0:
            if self.verbose > 1:
                print '- \twe can move directly to the target gamma...'
            return gamma
        else:
            # Solve for the optimal gamma using the bisection algorithm
            next_gamma = brentq(f, self.gamma, gamma, self)
            if self.use_mpi:
                self.comm.barrier()
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
        """
        Initialize the object.

        See the doc of the class for the description.
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

    def initialize(self, gamma, particles=None, num_mcmc_per_particle=10):
        """
        Initialize SMC at a particular ``gamma``.

        The method has basically three ways of initializing the particles:

        + If ``particles`` is not ``None``, then it is assumed to contain the
           particles at the corresponding value of ``gamma``.
        + If ``particles`` is ``None`` and the MCMC sampler class has a method
           called ``draw_from_prior()`` that works, then it is called to
           initialize the particles.
        + In any other case, MCMC sampling is used to initialize the particles.
           We are assuming that the MCMC sampler has already been tuned for
           that particular gamma and that a sufficient burning period has past.
           Then we record the current state as the first particle, we sample
           ``num_mcmc_per_particle`` times and record the second particle, and
           so on.

        :param gamma:               The initial ``gamma`` parameter. It must, of
                                    course, be within the right range of
                                    ``gamma``.
        :type gamma:                float
        :param particles:           A dictionary of MCMC states representing
                                    the particles. When using MPI, we are
                                    assuming that each one of the CPU's has each
                                    own collection of particles.
        :type particles:            (see :attr:`pymc.SMC.particles for type)
        :param num_mcmc_per_particle:   This parameter is ignored if
                                        ``particles`` is not ``None``. If the
                                        only way to initialize the particles is
                                        to use MCMC, then this is the number of
                                        of mcmc samples we drop before getting
                                        a SMC particle.
        """
        if self.verbose > 0:
            print '------------------------'
            print 'START SMC Initialization'
            print '------------------------'
            print '- initializing at', self.gamma_name, ':', gamma
        # Zero out the MCMC step counter
        self._total_num_mcmc = 0
        # Set gamma
        self.gamma = gamma
        # Set the weights and ESS
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = float(self.num_particles)
        if particles is not None:
            sys.stdout.write('- attempting to initialize with particles: ')
            self.particles = particles
            sys.stdout.write('SUCCESS\n')
        else:
            self.particles[0] = self.mcmc_sampler.get_state()
            try:
                if self.verbose > 0:
                    sys.stdout.write('- initializing by sampling from the prior: ')
                for i in range(1, self.my_num_particles):
                    self.mcmc_sampler.draw_from_prior()
                    self.particles[i] = self.mcmc_sampler.get_state()
                if self.verbose > 0:
                    sys.stdout.write('SUCCESS\n')
            except AttributeError:
                if self.verbose > 0:
                    sys.stdout.write('FAILURE\n')
                    print '- initializing via MCMC'
                    total_samples = self.num_particles * num_mcmc_per_particle
                    print '- taking a total of', total_samples
                    print '- creating a particle every', num_mcmc_per_particle
                if self.verbose > 0:
                    pb = pymc.progressbar.progress_bar(self.num_particles *
                                                       num_mcmc_per_particle)
                for i in range(1, self.my_num_particles):
                    self.mcmc_sampler.sample(num_mcmc_per_particle)
                    self.particles[i] = self.mcmc_sampler.get_state()
                    self._total_num_mcmc += num_mcmc_per_particle
                    if self.verbose > 0:
                        pb.update((i + 2) * self.size * num_mcmc_per_particle)
        self._add_current_state_to_db()
        if self.verbose > 0:
            print '----------------------'
            print 'END SMC Initialization'
            print '----------------------'

    def move_to(self, gamma):
        """
        Move the current particle approximation to ``gamma``.

        :param gamma:   The new ``gamma`` you wish to reach.
        :type gamma:    float

        .. note::

            There must already be a valid particle approximation. See
            :meth:`pysmc.SMC.initialize()` for ways of doing this.

        """
        if self.verbose > 0:
            print '-----------------'
            print 'START SMC MOVE TO'
            print '-----------------'
            print 'initial ', self.gamma_name, ':', self.gamma
            print 'final', self.gamma_name, ':', gamma
            print 'ess reduction: ', self.ess_reduction
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
                print '- moving to', self.gamma_name, ':', self.gamma
                pb = pymc.progressbar.progress_bar(self.num_particles *
                                                   self.num_mcmc)
                print '- performing', self.num_mcmc, 'MCMC steps per particle'
            for i in range(self.my_num_particles):
                self.mcmc_sampler.set_state(self.particles[i])
                self.mcmc_sampler.sample(self.num_mcmc)
                self.particles[i] = self.mcmc_sampler.get_state()
                self._total_num_mcmc += self.num_mcmc
                if self.verbose > 0:
                    pb.update(i * self.size * self.num_mcmc)
            self._add_current_state_to_db()
            if self.verbose > 1:
                print '- acceptance rate for each step method:'
                for sm in self.mcmc_sampler.step_methods:
                    acc_rate = sm.accepted / (sm.accepted + sm.rejected)
                    print '\t-', str(sm), ':', acc_rate
        if self.verbose > 0:
            print '- total number of MCMC steps:', self.total_num_mcmc
            print '---------------'
            print 'END SMC MOVE TO'
            print '---------------'

    def get_particles_of(self, var_name, type_of_var='stochastics'):
        """
        Get the particles pertaining to variable ``name``.

        If the collected particles can be converted to a numpy array, then this
        what is returned. Otherwise, we return is as a list of whatever objects
        the particles are.

        :param var_name:    The name of the variable whose particles you want to
                            get.
        :type var_name:     str
        :param type_of_var: The type of variables you want to get. This can be
                            either 'stochastics' or 'deterministics' if you are
                            are using :mod:`pymc`. The default type is 'stochastics'.
                            However, I do not restrict its value, in case you
                            would like to define other types by extending
                            :mod:`pymc`.
        :type type_of_var:  str
        :returns:           The particles pertaining to variable ``name`` of
                            type ``type_of_var``.
        :rtype:             :class:`numpy.ndarray` if possible, otherwise a
                            list of whatever types your model has.

        .. note::

            The object must represent a valid particle approximation.

        .. warning::

            When in parallel, you will get the particles owned by the cpu that
            calls this method.

        """
        return get_var_from_particle_list(self.particles, var_name, type_of_var)

    def get_gammas_from_db(self):
        """
        Get the gammas we have visited so far from the databse.

        :returns:   The gammas we have visited so far doing SMC.
        :rtype:     1D :class:`numpy.ndarray`
        """
        gammas = self.db['data'].keys()
        gammas.sort()
        return np.array(gammas)

    def get_weights_from_db(self, gamma):
        """
        Get the weights of each one of the particle approximations
        constructed so far.

        :param gamma:   The gamma parameter characterizing the
                        approximation. Do not just put any value here.
                        Get the values from
                        :meth:`pysmc.SMC.get_gammas_from_db()`.
        :type gamma:    float
        :raises:    :exc:`exceptions.RuntimeError`
        """
        self._check_if_gamma_is_in_db(gamma)
        return self.db['data'][gamma]['weights']

    def get_particles_from_db(self, gamma, var_name,
                              type_of_var='stochastics'):
        """
        Get the particles pertaining to variable ``var_name`` at
        ``gamma`.

        :param gamma:       The gamma parameter characterizing the
                            approximation. Do not just put any value here.
                            Get the values from
                            :meth:`pysmc.SMC.get_gammas_from_db()`.
        :type gamma:        float
        :param var_name:    The name of the variable whose particles you want to
                            get.
        :type var_name:     str
        :param type_of_var: The type of variables you want to get. This can be
                            either 'stochastics' or 'deterministics' if you are
                            are using :mod:`pymc`. The default type is
                            'stochastics'.
                            However, I do not restrict its value, in case you
                            would like to define other types by extending
                            :mod:`pymc`.
        :type type_of_var:  str
        :returns:           The particles pertaining to variable ``name`` of
                            type ``type_of_var``.
        :rtype:             :class:`numpy.ndarray` if possible, otherwise a
                            list of whatever types your model has.
        :raises:    :exc:`exceptions.RuntimeError`
        """
        self._check_if_gamma_is_in_db(gamma)
        particle_list = self.db['data'][gamma]['particles']
        return get_var_from_particle_list(particle_list, var_name, type_of_var)
