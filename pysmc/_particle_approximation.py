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


from . import get_var_from_particle_list


class ParticleApproximation(object):

    """
    Represents a particle approximation.
    """

    # The weights of the approximation
    _weights = None

    # The particles as returned from pysmc.SMC
    _particles = None

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

    def _fix_particles_of_type_and_name(self, type_of_var, var_name):
        var_data = get_var_from_particle_list(self.particles, var_name,
                                              type_of_var=type_of_var)
        setattr(self, var_name, var_data)
        getattr(self, type_of_var)[var_name] = var_data

    def _fix_particles_of_type(self, type_of_var):
        setattr(self, type_of_var, dict())
        for var_name in self.particles[0][type_of_var].keys():
            self._fix_particles_of_type_and_name(type_of_var, var_name)

    def _fix_particles(self):
        for type_of_var in self.particles[0].keys():
            self._fix_particles_of_type(type_of_var)

    def __init__(self, weights, particles):
        """
        Initialize the particle approximation.

        See the doc of this class for further details.
        """
        self._weights = weights
        self._particles = particles
        self._fix_particles()
