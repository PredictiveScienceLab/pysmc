"""

.. _particle_approx:

+++++++++++++++++++++
ParticleApproximation
+++++++++++++++++++++

:class:`pysmc.ParticleApproximation` is a class that implements a
a particle approximation.

Here is the complete reference of the public members:

"""


__all__ = ['ParticleApproximation']


from . import DistributedObject
from . import get_var_from_particle_list
import numpy as np
import warnings


class ParticleApproximation(DistributedObject):

    """
    Initialize a particle approximation.

    If :math:`x` denotes collectively all the variables involved in the
    particle approximation, then this object represents :math:`p(x)` as
    discussed in the :ref:`tutorial`.

    :param log_w:       The logarithms of the weights of the particle
                        approximation.
    :type log_w:        1D :class:`numpy.ndarray`
    :param particles:   The particles.
    :type particles:    list of dict
    :param mpi:         Specify this if you are creating a distributed particle
                        approximation.
    :type mpi:          :class:`mpi4py.MPI`
    :param comm:        The MPI communicator.
    :type comm:         :class:`mpi4py.COMM`

    .. note::

        When creating a distributed object, the particles must already be
        scattered.

    """

    # The logarithms of the weights
    _log_w = None

    # The weights of the approximation
    _weights = None

    # The particles as returned from pysmc.SMC
    _particles = None

    # The mean of the approximation
    _mean = None

    # The variance of the approximation
    _variance = None


    @property
    def log_w(self):
        """
        The logarithms of the weights of the particle approximation.

        :getter:    Get the logarithms of the weights of the particle
                    approximation.
        :type:      1D :class:`numpy.ndarray`
        """
        return self._log_w

    @property
    def weights(self):
        """
        The weights of the particle approximation.

        :getter:    Get the weights of the particle approximation.
        :type:      1D :class:`numpy.ndarray`
        """
        return self._weights

    @property
    def particles(self):
        """
        The particles of the particle approximation.

        :getter:    Get the particles of the particle approximation.
        :type:      A list of whatever types the approximation has.
        """
        return self._particles

    @property
    def my_num_particles(self):
        """
        The number of particles owned by this process.

        :getter:    Get the number of particles owned by this process.
        :type:      int
        """
        return len(self.particles)

    @property
    def num_particles(self):
        """
        The number of particles.

        :getter:    Get the number of particles.
        :type:      int
        """
        return self.my_num_particles * self.size

    @property
    def mean(self):
        """
        The mean of the variables of all types of the particle approximation.

        The mean of a variable :math:`x` is computed as:

        .. math::
            m(x) := \\sum_{j=1}^N w^{(j)} x^{(j)}.
            :label: x_mean

        :getter:    Get the mean of the particles.
        :type:      dict
        """
        self.compute_all_means()
        return self._mean

    @property
    def variance(self):
        """
        The variance of all the variables of all types of the particle
        approximation.

        The variance of a variable :math:`x` is computed as:

        .. math::
            v(x) := \\sum_{j=1}^N w^{(j)} \\left(x^{(j)}\\right)^2 - m^2(x),
            :label: x_var

        where :math:`m(x)` is given in :eq:`x_mean`.

        :getter:    Get the variance of all particles.
        :type:      dict
        """
        self.compute_all_variances()
        return self._variance

    def _fix_particles_of_type_and_name(self, type_of_var, var_name):
        """
        Expose the particles themselves.
        """
        var_data = get_var_from_particle_list(self.particles, var_name,
                                              type_of_var=type_of_var)
        if (var_data.ndim == 2 and (var_data.shape[0] == 1
                                   or var_data.shape[1] == 0)):
            var_data = var_data.flatten()
        setattr(self, var_name, var_data)
        getattr(self, type_of_var)[var_name] = var_data
        self._mean[type_of_var][var_name] = None
        self._variance[type_of_var][var_name] = None

    def _fix_particles_of_type(self, type_of_var):
        """
        Expose the dictionary of the type of vars.
        """
        setattr(self, type_of_var, dict())
        self._mean[type_of_var] = {}
        self._variance[type_of_var] = {}
        for var_name in self.particles[0][type_of_var].keys():
            self._fix_particles_of_type_and_name(type_of_var, var_name)

    def _fix_particles(self):
        """
        Fix the local variables based on the particles stored locally.
        """
        self._mean = dict()
        self._variance = dict()
        for type_of_var in self.particles[0].keys():
            self._fix_particles_of_type(type_of_var)

    def __init__(self, log_w=None, particles=None, mpi=None, comm=None):
        """
        Initialize the particle approximation.

        See the doc of this class for further details.
        """
        super(ParticleApproximation, self).__init__(mpi=mpi, comm=comm)
        if particles is not None:
            self._particles = particles[:]
            if log_w is None:
                log_w = (np.ones(self.my_num_particles) *
                         (-math.log(self.num_particles)))
            self._log_w = log_w[:]
            self._weights = np.exp(log_w)
            self._fix_particles()

    def __getstate__(self):
        """
        Get the state of the object so that it can be stored.
        """
        state = dict()
        state['log_w'] = self.log_w
        state['particles'] = self.particles
        return state

    def __setstate__(self, state):
        """
        Set the state of the object.
        """
        self.__init__(log_w=state['log_w'],
                      particles=state['particles'])

    def _check_if_valid_type_of_var(self, type_of_var):
        """
        Check if ``type_of_var`` is a valid type of variable.
        """
        if not self.particles[0].has_key(type_of_var):
            raise RuntimeError(
        'The particles do not have a \'%s\' type of variables!' % type_of_var)

    def _check_if_valid_var(self, var_name, type_of_var):
        """
        Check if ``var_name`` of type ``type_of_var`` exists.
        """
        self._check_if_valid_type_of_var(type_of_var)
        if not self.particles[0][type_of_var].has_key(var_name):
            raise RuntimeError(
        'The particles do not have a \'%s\' variable of type \'%s\'!'
        % (var_name, type_of_var))

    def get_particle_approximation_of(self, func, var_name,
                                      type_of_var='stochastics',
                                      func_name='func'):
        """
        Returns the particle approximation of a function of ``var_name``
        variable of type ``type_of_var`` of the particle approximation.

        Let the variable and the function we are referring to be :math:`x` and
        :math:`f(x)`, respectively. Then, let :math:`y = f(x)` denote the
        induced random variable when we pass :math:`x` through the function.
        The method returns the following particle approximation to the
        probability density of :math:`y`:

        .. math::
            p(y) \\approx \\sum_{j=1}^N w^{(j)}
            \\delta\\left(y - f\\left(x^{(j)} \\right)\\right)

        :param func:        A function of the desired variable.
        :type func:         function
        :param var_name:    The name of the desired variable.
        :type var_name:     str
        :param type_of_var: The type of the variable.
        :type type_of_var:  str
        :param func_name:   A name for the function. The new variable will be
                            named ``func_name + '_' + var_name``.
        :type func_name:    str
        :returns:           A particle approximation representing the random
                            variable ``func(var_name)``.
        :rtype:             :class:`pysmc.ParticleApproximation`
        """
        self._check_if_valid_var(var_name, type_of_var)
        func_var_name = func_name + '_' + var_name
        weights = self.weights
        particles = [dict() for i in xrange(self.my_num_particles)]
        for i in xrange(self.my_num_particles):
            particles[i][type_of_var] = dict()
            func_part_i = func(self.particles[i][type_of_var][var_name])
            particles[i][type_of_var][func_var_name] = func_part_i
        return ParticleApproximation(weights=weights, particles=particles)

    def get_mean_of_func(self, func, var_name, type_of_var):
        """
        Get the mean of the ``func`` applied on ``var_name`` which is of type
        ``type_of_var``.

        Let the variable and the function we are referring to be :math:`x` and
        :math:`f(x)`, respectively. Then the method computes and returns:

        .. math::
            \sum_{j=1}^Nw^{(j)}f\left(x^{(j)}\\right).

        :param func:        A function of one variable.
        :type func:         function
        :param var_name:    The name of the variable.
        :type var_name:     str
        :param type_of_var: The type of the variable.
        :type type_of_var:  str
        :returns:           The mean of the random variable :math:`y = f(x)`.
        :rtype:             unknown
        """
        self._check_if_valid_var(var_name, type_of_var)
        res = 0.
        for i in xrange(self.my_num_particles):
            res += (self.weights[i] *
                    func(self.particles[i][type_of_var][var_name]))
        if self.use_mpi:
            res = self.comm.reduce(res, op=self.mpi.SUM)
        return res

    def compute_mean_of_var(self, var_name, type_of_var,
                             force_calculation=False):
        """
        Compute the mean of the particle approximation.

        :param var_name:    The name of the variable.
        :type var_name:     str
        :param type_of_var: The type of the variable.
        :type type_of_var:  str
        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        self._check_if_valid_var(var_name, type_of_var)
        if (self._mean[type_of_var][var_name] is not None
            and not force_calculation):
            return
        self._mean[type_of_var][var_name] = self.get_mean_of_func(
            lambda x: x, var_name, type_of_var)

    def compute_all_means_of_type(self, type_of_var,
                                   force_calculation=False):
        """
        Compute the means of every variable of a type ``type_of_var``.

        :param type_of_var: The type of the variable.
        :type type_of_var:  str
        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        for var_name in self.particles[0][type_of_var].keys():
            self.compute_mean_of_var(var_name, type_of_var,
                                      force_calculation=force_calculation)

    def compute_all_means(self, force_calculation=False):
        """
        Compute all the means associated with the particle approximation.

        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        for type_of_var in self.particles[0].keys():
            self.compute_all_means_of_type(type_of_var,
                                            force_calculation=force_calculation)

    def compute_variance_of_var(self, var_name, type_of_var,
                                 force_calculation=False):
        """
        Compute the variance of ``var_name``.

        :param var_name:    The name of the variable.
        :type var_name:     str
        :param type_of_var: The type of the variable.
        :type type_of_var:  str

        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        self._check_if_valid_var(var_name, type_of_var)
        if (self._variance[type_of_var][var_name] is not None
            and not force_calculation):
            return
        self._variance[type_of_var][var_name] = self.get_mean_of_func(
            lambda x: x ** 2, var_name, type_of_var)
        self.compute_mean_of_var(var_name, type_of_var,
                                  force_calculation=force_calculation)
        if self.rank == 0:
            self._variance[type_of_var][var_name] -= (
                self._mean[type_of_var][var_name] ** 2)

    def compute_all_variances_of_type(self, type_of_var,
                                       force_calculation=False):
        """
        Compute all variances of type ``type_of_var``.

        :param type_of_var: The type of the variable.
        :type type_of_var:  str
        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        for var_name in self.particles[0][type_of_var].keys():
            self.compute_variance_of_var(var_name, type_of_var,
                                          force_calculation=force_calculation)

    def compute_all_variances(self, force_calculation=False):
        """
        Compute all the variances.

        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        for type_of_var in self.particles[0].keys():
            self.compute_all_variances_of_type(type_of_var,
                                            force_calculation=force_calculation)

    def compute_all_statistics(self, force_calculation=False):
        """
        Compute all the statistics of the particle approximation.

        :param force_calculation:   Computes the statistics even if a previous
                                    calculation was already made.
        :type force_calculation:    bool
        """
        self.compute_all_means(force_calculation=force_calculation)
        self.compute_all_variances(force_calculation=force_calculation)

    def resample(self):
        """
        Resample the particles. After calling this, all particles will have
        the same weight.
        """
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
        self._weights = np.exp(self.log_w)
        self._fix_particles()

    def copy(self):
        """
        Copy the particle approximation.

        :returns:   A copy of the current particle approximation.
        :rtype:     :class:`pysmc.ParticleApproximation`
        """
        new_pa = ParticleApproximation(self.log_w, self.particles,
                                       mpi=self.mpi, comm=self.comm)
        new_pa.mean = deepcopy(self.mean)
        new_pa.variance = deepcopy(self.variance)
        return new_pa

    def allgather(self):
        """
        Get a particle approximation on every process.

        If we are not using MPI, it will simply return a copy of the object.

        :returns:       A fully functional particle approximation on a single
                        process.
        :rtype:         :class:`smc.ParticleApproximation`
        """
        if not self.use_mpi:
            return self.copy()
        log_w = np.ndarray(self.num_particles)
        #self.comm.Gather([self._log_w, self.mpi.DOUBLE],
        #                 [log_w, self.mpi.DOUBLE])
        log_w = np.hstack(self.comm.allgather(self._log_w))
        tmp = self.comm.allgather(self.particles)
        particles = [t[i] for t in tmp for i in range(len(t))]
        return ParticleApproximation(log_w=log_w, particles=particles)
