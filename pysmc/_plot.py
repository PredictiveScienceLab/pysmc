"""

.. _plot:

++++++++
Plotting
++++++++

"""


__all__ = ['hist']


import os


def hist(plt, particle_approximation, var_name, bins=10, normed=True):
    """
    Plot the histogram of variable of a particle approximation.

    :param plt:     A reference to :mod:`matplotlib.pyplot`.
    :param particle_approximation:  A particle approximation.
    :type particle_approximation:   :class:`pysmc.ParticleApproximation`
    :param var_name:    The name of the variable you want to plot.
    :type var_name:     str
    :param bins:        The number of bins you want to use.
    :type bins:         int
    :param normed:      ``True`` if you want the histogram to be normalized,
                        ``False`` otherwise.
    :type normed:       bool
    """
    x = getattr(particle_approximation, var_name)
    w = particle_approximation.weights
    plt.xlabel(var_name, fontsize=16)
    plt.ylabel('p(%s)' % var_name, fontsize=16)
    return plt.hist(x, weights=w, bins=bins, normed=normed)


def make_mp4_movie_from_db(plt, db, var_name, bins=10, normed=True,
                           prefix):
    """
    Make an mp4 movie from a database.
    """
    filename = prefix + os.path.sext

