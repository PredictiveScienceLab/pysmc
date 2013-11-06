"""

.. _exa_rk:

+++++++++++++++++
Reaction Kinetics
+++++++++++++++++

We are trying to infer the forward and reverse rates of reaction
(:math:`k_1, k_2`) in a chemical system. The forward model is given by a set
of differential equations:

.. math::
    \\frac{du}{dt} = -k_1 u + k_2 v,
    \\frac{dv}{dt} = k_1 u - k_2 v.

The initial condition is fixed at :math:`u(0) = 1` and :math:`v(0) = 0`.
"""


__all__ = ['ReactionKineticsSolver']


import numpy as np
from scipy.integrate import ode


class ReactionKineticsSolver(object):

    """
    Implements the forward model.
    """

    # Observation times
    _t = None

    @property
    def t(self):
        """
        :getter:    Get the observation times.
        :type:      1D numpy.ndarray
        """
        return self._t

    def __init__(self, t=[2, 4, 5, 8, 10]):
        """
        Initialize the model.
        """
        self._t = np.array(t, dtype='float32')

    def __call__(self, k1, k2):
        """
        Evaluate the solver at ``k1`` and ``k2``.
        """
        def f(t, y, k1, k2):
            return [-k1 * y[0] + k2 * y[1], k1 * y[0] - k2 * y[1]]
        def jac(t, y, k1, k2):
            return [[-k1, k2], [k1, -k2]]
        r = ode(f).set_integrator('dopri5')
        r.set_initial_value([1, 0], 0).set_f_params(k1, k2)
        dt = 1e-2
        y = []
        for t in self.t:
            r.integrate(t)
            y.append(r.y)
        return np.vstack(y).flatten(order='F')


if __name__ == '__main__':
    """
    This main simply produces the obseved data for the inverse problem.
    """
    import cPickle as pickle
    re_solver = ReactionKineticsSolver()
    # The **real** reaction rates
    k1 = 2
    k2 = 4
    # The **true** response
    y = re_solver(k1, k2)
    # Add some noise to it
    noise = 0.1
    y_obs = y + noise * np.random.rand(*y.shape)
    # Save these to a file
    data = {}
    data['k1'] = k1
    data['k2'] = k2
    data['y'] = y
    data['noise'] = noise
    data['y_obs'] = y_obs
    with open('reaction_kinetics_data.pickle', 'wb') as fd:
        pickle.dump(data, fd, pickle.HIGHEST_PROTOCOL)
