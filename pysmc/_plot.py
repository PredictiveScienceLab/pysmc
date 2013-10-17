"""

.. _plot:

++++++++
Plotting
++++++++

"""


__all__ = ['hist', 'make_movie_from_db']


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import multinomial_resample
from . import kde


def hist(particle_approximation, var_name, normed=True):
    """
    Plot the histogram of variable of a particle approximation.

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
    if particle_approximation.use_mpi:
        comm = particle_approximation.comm
        x = np.hstack(comm.allgather(x))
        w = np.hstack(comm.allgather(w))
    bins = w.shape[0] / 10
    plt.xlabel(var_name, fontsize=16)
    plt.ylabel('p(%s)' % var_name, fontsize=16)
    return plt.hist(x, weights=w, bins=bins, normed=normed)


def make_movie_from_db(db, var_name):
    """
    Make a movie from a database.
    """
    k = kde(db.particle_approximations[0], var_name)
    x01 = k.dataset.min()
    x02 = k.dataset.max()
    x0 = np.linspace(x01, x02, 100)
    y0 = k(x0)
    k = kde(db.particle_approximations[-1], var_name)
    yl = k(x0)
    yl1 = yl.min()
    yl2 = yl.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False,
                         xlim=(x01, x02), ylim=(yl1, yl2))
    line, = ax.plot([], [], linewidth=2)
    particles, = ax.plot([], [], 'ro', markersize=5)
    gamma_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                         fontsize=16)
    ax.set_xlabel(var_name, fontsize=16)
    ax.set_ylabel('p(%s)' % var_name, fontsize=16)

    def init():
        line.set_data([], [])
        particles.set_data([], [])
        gamma_text.set_text('')
        return line, particles, gamma_text

    def animate(i):
        k = kde(db.particle_approximations[i], var_name)
        line.set_data(x0, k(x0))
        p = getattr(db.particle_approximations[i], var_name)
        particles.set_data(p, np.zeros(p.shape) + yl1 + 0.01 * (yl2 - yl1))
        gamma_text.set_text('%s = %1.4f' % (db.gamma_name, db.gammas[i]))
        return line, particles, gamma_text

    ani = animation.FuncAnimation(fig, animate, frames=db.num_gammas,
                                  interval=200, blit=True, init_func=init)
    return ani
