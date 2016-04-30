"""

.. _db:

++++++++
DataBaseConcept
++++++++

The class :class:`pysmc.DataBaseConcept` implements a simple database for dumping
SMC steps.
"""


__all__ = ['DataBaseConcept']


import os


class DataBaseConcept(object):
    
    """
    This class defines the interface of a valid database object
    for pysmc.

    The purpose of a database is to store and retrieve a sequence
    of particle approximations resulting by running the class
    :class:`pysmc.SMC` along with all the metadata required for
    restarting the algorithm.

    """
    
    def __init__(self):
        # Does nothing.
        pass

    def initialize(self, filename, gamma_name):
        raise NotImplementedError('Implement me.')

    def add(self, gamma, particle_approximation,
            step_method_params):
        """
        Add the ``particle_approximation`` corresponding to ``gamma`` to the
        database.

        :param gamma:                   The gamma parameter.
        :type gamma:                    any
        :param particle_approximation:  particle_approximation
        :type particle_approximation:   any
        """
        raise NotImplementedError('Implement me.')

    def commit(self):
        """
        Commit everything we have so far to the database.
        """
        pass

    @staticmethod
    def load(filename):
        """
        This is a static method. It loads a database from ``filename``.
        """
        raise NotImplementedError('Implement me.')

    @property
    def gamma_name(self):
        """
        Return the name of the gamma parameter.
        """
        raise NotImplementedError('Implement me.')

    @property
    def gammas(self):
        """
        The list of gammas we have visited.

        :getter:    Get the list of gammas we have visited.
        :type:      list
        """
        raise NotImplementedError('Implement me.')

    @property
    def num_gammas(self):
        """
        The number of gammas added to the database.

        :getter:    Get the number of gammas added to the data base.
        :type:      int
        """
        return len(self.gammas)

    def get_particle_approximation(self, i):
        """
        :getter:    The particle approximation associated with step i.
        """
        raise NotImplementedError('Implement me.')

    @property
    def particle_approximation(self):
        """
        The current particle approximation of the database.

        :getter:    Get the current particle approximation of the database.
        :type:      unknown
        """
        return self.get_particle_approximation(self.num_gammas - 1)

    def get_step_method_param(self, i):
        """
        :getter:    The step method parameters associated with step i.
        """
        raise NotImplementedError('Implement me')

    @property
    def step_method_param(self):
        """
        The current step method parameters in the database.

        :getter:    Get the current step method parameters from the database.
        :type:      unknown
        """
        return self.get_step_method_param(self.num_gammas - 1)

    def __str__(self):
        """
        A string representation of the object.
        """
        s = '-' * 80 + '\n'
        s += 'Database'.center(80) + '\n'
        s += 'gamma name: {0:s}\n'.format(self.gamma_name)
        s += 'number of steps so far: {0:d}\n'.format(self.num_gammas)
        s += '-' * 80
        return s
