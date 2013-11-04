"""
A simple mixture model to test the capabilities of SMC.

Author:
    Ilias Bilionis

Date:
    9/22/2013
"""


import pymc
import numpy as np
import math


def make_model():
    # The gamma parameter
    gamma = 1.

    @pymc.stochastic(dtype=float)
    def mixture(value=1., gamma=gamma, pi=[0.2, 0.8], mu=[-2., 3.],
                sigma=[0.01, 0.01]):
        """
        The log probability of a mixture of normal densities.

        :param value:       The point of evaluation.
        :type value :       float
        :param gamma:       The parameter characterizing the SMC one-parameter
                            family.
        :type gamma :       float
        :param pi   :       The weights of the components.
        :type pi    :       1D :class:`numpy.ndarray`
        :param mu   :       The mean of each component.
        :type mu    :       1D :class:`numpy.ndarray`
        :param sigma:       The standard deviation of each component.
        :type sigma :       1D :class:`numpy.ndarray`
        """
        # Make sure everything is a numpy array
        pi = np.array(pi)
        mu = np.array(mu)
        sigma = np.array(sigma)
        # The number of components in the mixture
        n = pi.shape[0]
        # pymc.normal_like requires the precision not the variance:
        tau = np.sqrt(1. / sigma ** 2)
        # The following looks a little bit awkward because of the need for
        # numerical stability:
        p = np.log(pi)
        p += np.array([pymc.normal_like(value, mu[i], tau[i])
                       for i in range(n)])
        p = math.fsum(np.exp(p))
        # logp should never be negative, but it can be zero...
        if p <= 0.:
            return -np.inf
        return gamma * math.log(p)

    return locals()


def eval_stochastic_variable(var, values):
    """
    Evaluate the logarithm of the probability of ``var`` at ``values``.

    :param var   :      The stochastic variable whose probability should be
                        evaluated.
    :type var    :      :class:`pymc.Stochastic`
    :param values:      The points of evalulation.
    :type values :      list or :class:`numpy.ndarray`
    :returns     :      The logarithm of the probabilities of the variable
                        at all ``values``.
    :rtype       :      1D :class:`numpy.ndarray`
    """
    n = len(values)
    y = np.zeros(n)
    for i in range(n):
        var.value = values[i]
        y[i] = math.exp(var.logp)
    return y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 3, 200)

    # Plot the original probability density
    m = make_model()
    y = eval_stochastic_variable(m['mixture'], x)
    plt.plot(x, y, linewidth=2)
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$p(x)$', fontsize=16)
    plt.savefig('../doc/source/images/simple_model_pdf.png')

    # Plot some members of the one-parameter SMC family of probability densities
    plt.clf()
    gammas = [1., 0.7, 0.5, 0.1, 0.05, 0.01]
    for gamma in gammas:
#        m['mixture'].parents['gamma'] = gamma
        m['gamma'] = gamma
        y = eval_stochastic_variable(m['mixture'], x)
        plt.plot(x, y, linewidth=2)
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$\pi_\gamma(x)$', fontsize=16)
    legend_labels = ['$\gamma = %1.2f$' % gamma for gamma in gammas]
    plt.legend(legend_labels, loc='upper left')
    plt.show()
#    plt.savefig('../doc/source/images/simple_model_pdf_family.png')
