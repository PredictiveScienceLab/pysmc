"""

.. _misc:

++++++++++++++++++++++
Miscellaneous routines
++++++++++++++++++++++

This contains methods that do not fit into the other sections of the
reference manual.

"""


__all__ = ['try_to_array', 'get_var_from_particle_list',
           'multinomial_resample', 'kde']


import numpy as np
from scipy.stats import gaussian_kde


def try_to_array(data):
    """
    Try to turn the data into a numpy array.

    :returns:   If possible, a :class:`numpy.ndarray` containing the
                data. Otherwise, it just returns the data.
    :rtype:     :class:`numpy.ndarray` or ``type(data)``
    """
    try:
        return np.array(data)
    except:
        return data


def get_var_from_particle_list(particle_list, var_name, type_of_var):
    """
    Get the particles pertaining to variable ``var_name`` of type
    ``type_of_var``.

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
    """
    data = [particle_list[i][type_of_var][var_name]
            for i in range(len(particle_list))]
    return try_to_array(data)


def multinomial_resample(p):
    """
    Sample the multinomial according to ``p``.

    :param p:   A numpy array of positive numbers that sum to one.
    :type p:    1D :class:`numpy.ndarray`
    :returns:   A set of indices sampled according to p.
    :rtype:     1D :class:`numpy.ndarray` of int
    """
    p = np.array(p)
    assert p.ndim == 1
    assert (p >= 0.).all()
    births = np.random.multinomial(p.shape[0], p)
    idx_list = []
    for i in xrange(p.shape[0]):
        idx_list += [i] * births[i]
    return np.array(idx_list, dtype='i')


def kde(particle_approximation, var_name):
    """
    Construct a kernel density approximation of the probability density
    of ``var_name`` of ``particle_approximation``.

    :param particle_approximation:  A particle approximation.
    :type particle_approximation:   :class:`pysmc.ParticleApproximation`
    :returns:                       A kernel density approximation.
    :rtype:                         :class:`scipy.stats.gaussian_kde`
    """
    x = getattr(particle_approximation, var_name)
    x = np.atleast_2d(x).T
    w = particle_approximation.weights
    idx = multinomial_resample(w)
    return gaussian_kde(x[idx, :].T)
