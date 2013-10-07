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
from . import DistributedObject
from . import ParticleApproximation
from . import DataBase
import pymc
import numpy as np
from scipy.optimize import brentq
import math
import itertools
import sys
import warnings
import os


class SMC(DistributedObject):
    """
    Use Sequential Monte Carlo (SMC) to sample from a distribution.

    :param mcmc_sampler:        This is an essential part in initializing the
                                object. It can either be a ready to go
                                MCMC sampler or a module/class representing
                                a :mod:`pymc` model. In the latter case, the
                                MCMC sampler will be initialized automatically.
    :type mcmc_sampler:         :class:`pymc.MCMC`, :class:`pysmc.MCMCWrapper`
                                or a :mod:`pymc` model
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
    :param gamma_is_an_exponent: A flag that should be ``True`` if ``gamma``
                                 appears as an exponent in the probability
                                 density, e.g.,
                                 :math:`p(x | y) \\propto p(y | x)^{\\gamma} p(x)`.
                                 The default value is ``False``. However, if
                                 your model is of the right form **it pays off**
                                 to set it to ``True``. Then we can solve the
                                 problem of finding the next :math:``\gamma``
                                 in the sequence a lot faster.
    :type gamma_is_an_exponent:  bool
    :param db_filename:         The filename of a database for the object. If
                                the database exists and is a valid one, then
                                the object will be initialized at each last
                                state. If the parameter ``update_db`` is also
                                set, then the algorithm will dump the state of
                                each ``gamma`` it visits and commit it to the
                                data base. Otherwise, commits can be forced by
                                calling :meth:`pysmc.SMC.commit()`.
    :type db_filename:          str
    :param mpi:                 The MPI class (see :mod:`mpi4py` and
                                :ref:`mpi_example`). If ``None``, then no
                                parallelism is used.
    :param comm:                Set this to the MPI communicator. If ``None``,
                                then ``mpi.COMM_WORLD`` is used.
    """

    # The logarithm of the weights
    _log_w = None

    # The current effective sample size
    _ess = None

    # The number of MCMC steps for each gamma
    _num_mcmc = None

    # The thresshold of the effective sample size (percentage)
    _ess_threshold = None

    # The reduction of the effective sample size per gamma step
    _ess_reduction = None

    # Do you want to adaptively select the MCMC proposal step?
    _adapt_proposal_step = None

    # The amount of verbosity
    _verbose = None

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

    # Update the database or not
    _update_db = False

    # Count the total number of MCMC samples taken so far
    _total_num_mcmc = None

    # Does gamma appear as an exponent in the probability density?
    _gamma_is_an_exponent = None

    @property
    def gamma_is_an_exponent(self):
        """
        The Flag that determines if gamma is an exponent in the probability
        density.

        :getter:        Get the flag.
        :setter:        Set the flag.
        :type:          bool
        """
        return self._gamma_is_an_exponent

    @gamma_is_an_exponent.setter
    def gamma_is_an_exponent(self, value):
        """Set the value of the flag."""
        value = bool(value)
        self._gamma_is_an_exponent = value

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
        return len(self.particles)

    @property
    def num_particles(self):
        """
        :getter:    Get the number of particles.
        :type:      int
        """
        return self.my_num_particles * self.size

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
            raise ValueError('num_mcmc <= 0!')
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

    @property
    def update_db(self):
        """
        Update the database or not.

        :getter:    Get the ``update_db`` flag.
        :setter:    Set the ``update_db`` flag.
        :type:      bool
        """
        return self._update_db

    @update_db.setter
    def update_db(self, value):
        """
        Set the ``update_db`` flag.
        """
        value = bool(value)
        self._update_db = value

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

    def _set_gamma(self, value):
        """
        Set the value of gamma.
        """
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
        :type:      list of whatever objects your method supports.
        """
        return self._particles

    @property
    def total_num_mcmc(self):
        """
        The total number of MCMC steps performed so far.
        This is zeroed, everytime you call :meth:`pysmc.SMC.initialize()`.

        :getter:    The total number of MCMC steps performed so far.
        :type:      int
        """
        if self.use_mpi:
            return self.comm.allreduce(self._total_num_mcmc, op=self.mpi.SUM)
        else:
            return self._total_num_mcmc

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
        if self.gamma_is_an_exponent:
            return (gamma  - self.gamma)* self._loglike
        logp_new = self._get_logp_at_gamma(gamma)
        return logp_new - self._logp_prev

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
            num_particles = self.num_particles
            my_num_particles = self.my_num_particles
            old_particles = self._particles
            self._particles = []
            for i in xrange(num_particles):
                to_whom = i / my_num_particles
                from_whom = idx[i] / my_num_particles
                if from_whom == to_whom and to_whom == self.rank:
                    my_idx = idx[i] % my_num_particles
                    self._particles.append(old_particles[my_idx].copy())
                elif to_whom == self.rank:
                    self._particles.append(self.comm.recv(
                                                       source=from_whom, tag=i))
                elif from_whom == self.rank:
                    my_idx = idx[i] % my_num_particles
                    self.comm.send(old_particles[my_idx], dest=to_whom, tag=i)
                self.comm.barrier()
        else:
            self._particles = [self._particles[i].copy() for i in idx]
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = self.num_particles
        if self.verbose > 1:
            sys.stdout.write('SUCCESS\n')

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

    def _get_logp_of_particle(self, i):
        """
        Get the logp of a particle.
        """
        self.mcmc_sampler.set_state(self.particles[i])
        return self.mcmc_sampler.logp

    def _get_logp_of_particles(self):
        """
        Get the logp of all particles.
        """
        return np.array([self._get_logp_of_particle(i)
                         for i in xrange(self.my_num_particles)])

    def _get_logp_at_gamma(self, gamma):
        """
        Get the logp at gamma.
        """
        if gamma is not self.gamma:
            old_gamma = self.gamma
            self._set_gamma(gamma)
            logp = self._get_logp_of_particles()
            self._set_gamma(old_gamma)
        else:
            logp = self._get_logp_of_particles()
        return logp

    def _get_loglike(self, gamma0, gamma1):
        """
        Get the log likelihood assuming that gamma appears in the exponent.
        """
        logp0 = self._get_logp_at_gamma(gamma0)
        logp1 = self._get_logp_at_gamma(gamma1)
        return (logp1 - logp0) / (gamma1 - gamma0)

    def _find_next_gamma(self, gamma):
        """
        Find the next gamma.

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

        if self.gamma_is_an_exponent:
            self._loglike = self._get_loglike(self.gamma, gamma)
        else:
            self._logp_prev = self._get_logp_at_gamma(self.gamma)

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

    def _initialize_db(self, db_filename, update_db):
        """
        Initialize the database.
        """
        if db_filename is not None:
            db_filename = os.path.abspath(db_filename)
            if self.verbose > 0:
                print '- db: ' + db_filename
            if os.path.exists(db_filename):
                if self.verbose > 0:
                    print '- db exists'
                    print '- assuming this is a restart run'
                self._db = DataBase.load(db_filename)
                # Sanity check
                if not self.db.gamma_name == self.gamma_name:
                    raise RuntimeError(
                    '%s in db does not match %s in SMC' % (seld.db.gamma_name,
                                                           gamma_name))
                if self.verbose > 0:
                    print '- db:'
                    print '\t- num. of %s: %d' % (self.gamma_name,
                                                  self.db.num_gammas)
                    print '\t- first %s: %1.4f' % (self.gamma_name,
                                                  self.db.gammas[0])
                    print '\t- last %s: %1.4f' % (self.gamma_name,
                                                  self.db.gammas[-1])
                print '- initializing the object at the last state of db'
                self.initialize(self.db.gamma,
                        particle_approximation=self.db.particle_approximation)
            else:
                if self.verbose > 0:
                    print '- db does not exist'
                    print '- creating db file'
                self._db = DataBase(gamma_name=self.gamma_name,
                                    filename=db_filename)
            if self.verbose > 0:
                if update_db:
                    print '- commiting to the database at every step'
                else:
                    print '- manually commiting to the database'
        elif update_db:
            if self.verbose > 0:
                warnings.warn(
                '- update_db flag is on but no db_filename was specified\n'
              + '- setting the update_db flag to off')
            update_db = False
        self._update_db = update_db

    def _set_gamma_name(self, gamma_name):
        """
        Safely, set the gamma_name parameter.
        """
        if not isinstance(gamma_name, str):
            raise TypeError('The \'gamma_name\' parameter must be a str!')
        self._gamma_name = gamma_name

    def _set_mcmc_sampler(self, mcmc_sampler):
        """
        Safely, set the MCMC sampler.
        """
        if (not isinstance(mcmc_sampler, pymc.MCMC)
            and not isinstance(mcmc_sampler, MCMCWrapper)):
            # Try to make an MCMC sampler out of it (it will work if it is a
            # valid model).
            try:
                if self.rank == 0:
                    warnings.warn(
                    '- mcmc_sampler is not a pymc.MCMC.\n'
                  + '- attempting to make it one!')
                mcmc_sampler = pymc.MCMC(mcmc_sampler)
            except:
                raise RuntimeError(
            'The mcmc_sampler object could not be converted to a pymc.MCMC!')
        if not isinstance(mcmc_sampler, MCMCWrapper):
            mcmc_sampler = MCMCWrapper(mcmc_sampler)
        self._mcmc_sampler = mcmc_sampler
        self._update_gamma_rv()

    def _set_initial_particles(self, num_particles):
        """
        Safely, set the initial particles.
        """
        num_particles = int(num_particles)
        if num_particles <= 0:
            raise ValueError('num_particles <= 0!')
        my_num_particles = num_particles / self.size
        if my_num_particles * self.size < num_particles:
            warnings.warn(
            '- number of particles (%d) not supported on %d mpi processes' %
                (num_particles, self.size))
            num_particles = my_num_particles * self.size
            warnings.warn(
             '- changing the number of particles to %d' % num_particles)
        self._particles = [None for i in xrange(my_num_particles)]
        self._log_w = (np.ones(my_num_particles)
                       * (-math.log(num_particles)))

    def __init__(self, mcmc_sampler=None,
                 num_particles=10, num_mcmc=10,
                 ess_threshold=0.67,
                 ess_reduction=0.90,
                 adapt_proposal_step=True,
                 verbose=0,
                 mpi=None,
                 comm=None,
                 gamma_name='gamma',
                 db_filename=None,
                 update_db=False,
                 gamma_is_an_exponent=False):
        """
        Initialize the object.

        See the doc of the class for the description.
        """
        super(SMC, self).__init__(mpi=mpi, comm=comm)
        self._set_gamma_name(gamma_name)
        self._set_mcmc_sampler(mcmc_sampler)
        self._set_initial_particles(num_particles)
        self.num_mcmc = num_mcmc
        self.ess_threshold = ess_threshold
        self.ess_reduction = ess_reduction
        self.verbose = verbose
        self.adapt_proposal_step = adapt_proposal_step
        self.gamma_is_an_exponent = gamma_is_an_exponent
        self._initialize_db(db_filename, update_db)

    def initialize(self, gamma, particle_approximation=None,
                   num_mcmc_per_particle=10):
        """
        Initialize SMC at a particular ``gamma``.

        The method has basically three ways of initializing the particles:

        + If ``particles_approximation`` is not ``None``,
          then it is assumed to contain the
          particles at the corresponding value of ``gamma``.
        + If ``particles_approximation`` is ``None`` and the
          MCMC sampler class has a method
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
        :param particles_approximation: A dictionary of MCMC states representing
                                        the particles. When using MPI, we are
                                        assuming that each one of the CPU's
                                        has each own collection of particles.
        :type particles_approximation:  :class:`pysmc.ParticleApproximation`
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
        self._set_gamma(gamma)
        # Set the weights and ESS
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = float(self.num_particles)
        if particle_approximation is not None:
            if self.verbose > 0:
                print '- initializing with a particle approximation.'
            self._particles = particle_approximation.particles
            self._log_w = particle_approximation.log_w
            return
        else:
            self.particles[0] = self.mcmc_sampler.get_state()
            try:
                if self.verbose > 0:
                    sys.stdout.write(
                            '- initializing by sampling from the prior: ')
                for i in range(1, self.my_num_particles):
                    self.mcmc_sampler.draw_from_prior()
                    self.particles[i] = self.mcmc_sampler.get_state()
                if self.verbose > 0:
                    sys.stdout.write('SUCCESS\n')
            except AttributeError:
                if self.verbose > 0:
                    sys.stdout.write('FAILURE\n')
                    print '- initializing via MCMC'
                    if self.use_mpi:
                        total_samples = (self.my_num_particles
                                         * num_mcmc_per_particle)
                        print '- taking a total of', total_samples, 'samples per process'
                    else:
                        total_samples = (self.num_particles
                                         * num_mcmc_per_particle)
                        print '- taking a total of', total_samples, 'samples'
                    print '- creating a particle every', num_mcmc_per_particle
                if self.verbose > 0:
                    pb = pymc.progressbar.ProgressBar(self.num_particles *
                                                      num_mcmc_per_particle)
                # Only rank 0 keeps the first particle
                if self.rank == 0:
                    start_idx = 1
                else:
                    start_idx = 0
                for i in range(start_idx, self.my_num_particles):
                    self.mcmc_sampler.sample(num_mcmc_per_particle)
                    self.particles[i] = self.mcmc_sampler.get_state()
                    self._total_num_mcmc += num_mcmc_per_particle
                    if self.verbose > 0:
                        pb.animate((i + 2) * self.size * num_mcmc_per_particle)
                if self.verbose > 0:
                    print ''
        if self.update_db:
            self.db.add(self.gamma, self.get_particle_approximation())
            self.db.commit()
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
            self._set_gamma(new_gamma)
            if self.ess < self.ess_threshold * self.num_particles:
                self._resample()
            if self.verbose > 0:
                print '- moving to', self.gamma_name, ':', self.gamma
                pb = pymc.progressbar.ProgressBar(self.num_particles *
                                                   self.num_mcmc)
                print '- performing', self.num_mcmc, 'MCMC steps per particle'
            for i in range(self.my_num_particles):
                self.mcmc_sampler.set_state(self.particles[i])
                self.mcmc_sampler.sample(self.num_mcmc)
                self.particles[i] = self.mcmc_sampler.get_state()
                self._total_num_mcmc += self.num_mcmc
                if self.verbose > 0:
                    pb.animate(i * self.size * self.num_mcmc)
            if self.verbose > 0:
                print ''
            if self.update_db:
                self.db.add(self.gamma, self.get_particle_approximation())
                self.db.commit()
            if self.verbose > 1:
                print '- acceptance rate for each step method:'
                for sm in self.mcmc_sampler.step_methods:
                    acc_rate = sm.accepted / (sm.accepted + sm.rejected)
                    print '\t-', str(sm), ':', acc_rate
        total_num_mcmc = self.total_num_mcmc
        if self.verbose > 0:
            print '- total number of MCMC steps:', total_num_mcmc
            print '---------------'
            print 'END SMC MOVE TO'
            print '---------------'

    def get_particle_approximation(self):
        """
        Get a :class:`pysmc.ParticleApproximation` representing the current
        state of SMC.

        :returns:   A particle approximation of the current state.
        :rtype:     :class:`pysmc.ParticleApproximation`
        """
        return ParticleApproximation(log_w=self.log_w, particles=self.particles,
                                     mpi=self.mpi, comm=self.comm)

    def commit(self):
        """
        Commit the current state to the data base.
        """
        if not self.update_db:
            warnings.warn(
        '- requested a commit to the db but no db found\n'
        '- ignoring request')
            return
        self.db.add(self.gamma, self.get_particle_approximation())
        self.db.commit()
