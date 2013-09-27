.. _tutorial:

========
Tutorial
========

 
Before moving forward, make sure you understand:

    + What is probability? That's a big question... If you feel like it,
      I suggest you skim through `E. T. Jaynes`_'s book
      `Probability Theory: The Logic of Science`_.
    + What is `MCMC`_?
    + Read the tutorial of `PyMC`_.
      To the very least, you need to be able to construct probabilistic
      models using this package. For advanced applications, you need to be
      able to construct your own `MCMC step methods`_.
    + Of course, I have to assume some familiarity with the so called 
      Sequential Monte Carlo (SMC) or Particle Methods. Many resources can
      be found online at `Arnaud Doucet's collection`_ or in his book
      `Sequential Monte Carlo Methods in Practice`_. What exactly :mod:`pysmc`
      does is documented in :ref:`math`.


.. _what is smc:

-------------------------------
What is Sequential Monte Carlo?
-------------------------------

Sequential Monte Carlo (SMC) is a very efficient and effective way to sample
from complicated probability distributions known up to a normalizing constant.
The most important complicating factor is multi-modality. That is, probability
distributions that do not look at all like Gaussians.

The complete details can be found in :ref:`math`. However, let us give some
insights on what is really going on. Assume that we want so sample
from a probability distribution :math:`p(x)`, known up to a normalizing
constant:

.. math::

    p(x) \propto \pi(x).

Normally, we construct a variant of `MCMC`_ to sample from this
distribution. If :math:`p(x)` is multi-modal, it is certain that the Markov
Chain will get attracted to one of the modes. Theoretically, if the modes are
connected, it is guaranteed that they will all be visited
*as the number of MCMC steps goes to infinity*. However, depending on the
probability of the paths that connect the modes, the chain might
never escape during the finite number of MCMC steps that we can actually afford
to perform.

SMC attempts to alleviate this problem. The way it does it is similar to the
ideas found in `Simulated Annealing`_. The user defines a family of probability
densities:

.. math::
    p_{i}(x) \propto \pi_{i}(x),\;i=1,\dots,n,
    :label: smc_sequence

such that:

+ it is easy to sample from :math:`p_0(x)` (either directly or using MCMC),
+ the probability densities :math:`p_i(x)` and :math:`p_{i+1}(x)` are
  *similar*,
+ the last probability density of the sequence is the target, i.e.,
  :math:`p_n(x) = p(x)`.

There are many ways to define such a sequence. Usually, the exact sequence that
needs to be followed is obvious from the definition of the problem. An obvious
choice is:

.. math::
    p_i(x) \propto \pi^{\gamma_i}(x),\;i=1,\dots,n,
    :label: smc_sequence_power

where :math:`\gamma_0` is a non-negative number that makes :math:`p_i(x)` look
flat (e.g., if :math:`p(x)` has a compact support, you may choose 
:math:`\gamma_0=0` which makes :math:`p_0(x)` the uniform density. For
the general case a choice like :math:`\gamma_0=10^{-3}` would still do a good
job) and :math:`\gamma_n=1`. If :math:`n` is chosen sufficiently large and
:math:`gamma_i < \gamma_{i+1}` then indeed :math:`p_i(x)` and :math:`p_{i+1}(x)`
will look similar.

Now we are in a position to discuss what SMC does. We represent each one of the
probability densities :math:`p_i(x)`
:eq:`smc_sequence` with a *particle approximation*
:math:`\left\{\left(w^{(j)_i}, x^{(j)_i}\right)\right\}_{j=1}^N`, where:

+ :math:`N` is known as the *number of particles*,
+ :math:`w^{(j)}_i` is known as the *weight* of particle :math:`j`
  (normalized so that :math:`\sum_{j=1}^Nw^{(j)}_i=1`),
+ :math:`x^{(j)}_i` is known as the *particle* :math:`j`.

Typically we write:

.. math::
    p_i(x) \approx \sum_{j=1}^Nw^{(j)}_i\delta\left(x - x^{(j)}_i\right),
    :label: smc_approx

but what we really mean is that for any measurable
function of the state space :math:`f(x)` the following holds:

.. math::
    \lim_{N\rightarrow\infty}\sum_{j=1}^Nw_i^{(j)}f\left(x^{(j)}_i\right) = \
    \int f(x) p_i(x)dx,
    :label: smc_approx_def

almost surely.

So far so good. The only issue here is actually constructing a particle
approximation satisfying :eq:`smc_approx_def`. This is a little bit involved
and thus described in :ref:`math`. Here it suffices to say that it more or less
goes like this:

1. Start with :math:`i=0` (i.e., the easy to sample distribution).
2. Sample :math:`x_0^{(j)}` from :math:`p_0(x)` either directly (if possible) or
   using MCMC and set the weights equal to :math:`w_0^{(j)} = 1 / N`. Then
   :eq:`smc_approx_def` is satisfied for :math:`i=0`.
3. Compute the weights :math:`w_{i+1}(j)` and sample -using an appropriate MCMC
   kernel- the particles of the next step :math:`x_i^{(j+1)}` so that they
   corresponding particle approximation satisfies :eq:`smc_approx_def`.
4. Set :math:`i=i+1`.
5. If :math:`i=n` stop. Otherwise go to 3.

.. _what is in pysmc:

------------------------------------
What is implemented in :mod:`pysmc`?
------------------------------------

:mod:`pysmc` implements something a little bit more complicated than what is
described in :ref:`what is smc`. The full description can be found in
:ref:`math`. Basically, we assume that the user has defined a one-parameter
family of probability densities:

.. math::
    p_{\gamma}(x) \propto \pi_{\gamma}(x).
    :label: p_gamma

The code must be initialized with a particle approximation at a desired value
of :math:`\gamma=\gamma_0`. This can be done either manually by the user or
automatically by :mod:`pysmc` (e.g. by direct sampling or MCMC).
Having constructed an initial particle approximation, the code can be instructed
to move it to another :math:`\gamma=\gamma_1`. If the two probability densities
:math:`p_{\gamma_0}(x)` and :math:`p_{\gamma_1}(x)` are close, then the code
will jump directly into the construction of the particle approximation at
:math:`\gamma=\gamma_1`. If not, then it will adaptively construct a finite
sequence of :math:`\gamma`'s connecting :math:`\gamma_0` and :math:`\gamma_1`
and jump from one to the other. Therefore, the user only needs to specify:

+ the initial, easy-to-sample-from probability density,
+ the target density,
+ a one-parametric family of densities that connect the two.

We will see how this can be achieved through a bunch of examples.


.. _simple example:

----------------
A Simple Example
----------------

We will start with a probability density with two modes, namely a mixture of
two normal densities:

.. math::
    p(x) = \pi_1 \mathcal{N}\left(x | \mu_1, \sigma_1^2 \right) + \
           \pi_2 \mathcal{N}\left(x | \mu_2, \sigma_2^2 \right),
    :label: simple_model_pdf

where :math:`\mathcal{N}(x|\mu, \sigma^2)` denotes the probability density of a
normal random variable with mean :math:`\mu` and variance :math:`\sigma^2`.
:math:`\pi_i>0` is the weight given to the :math:`i`-th normal
(:math:`\pi_1 + \pi_2 = 1`) and :math:`\mu_i, \sigma_i^2` are the corresponding
mean and variance. We pick the following parameters:

+ :math:`\pi_1=0.2, \mu_1=-1, \sigma_1=0.01`,
+ :math:`\pi_2=0.8, \mu_2=2, \sigma_2=0.01`.

This probability density is shown in `Simple Example PDF Figure`_. It is obvious
that sampling this probability density using MCMC will be very problematic.

.. _Simple Example PDF Figure:
.. figure:: images/simple_model_pdf.png
    :align: center

    Plot of :eq:`simple_model_pdf` with
    :math:`\pi_1=0.2, \mu_1=-1, \sigma_1=0.01` and
    :math:`\pi_2=0.8, \mu_2=2, \sigma_2=0.01`.

.. _simple example pdf family:

++++++++++++++++++++++++++++++++++++++++++++++++++
Defining a family of probability densities for SMC
++++++++++++++++++++++++++++++++++++++++++++++++++

Remember that our goal is to sample :eq:`simple_model_pdf` using SMC. Towards
this goal we need to define a one-parameter family of probability densities
:eq:`p_gamma` starting from a simple one to our target. The simplest choice
is probably this:

.. math::
    \pi_{\gamma}(x) = p^{\gamma}(x).
    :label: simple_model_pdf_family

Notice that: 1) for :math:`\gamma=1` we obtain :math:`p_\gamma(x)` and 2) for
:math:`\gamma` small (say :math:`\gamma=10^{-2}`) we obtain a relatively flat
probability density. See `Simple Example Family of PDF's Figure`_.

.. _Simple Example Family of PDF's Figure:
.. figure:: images/simple_model_pdf_family.png
    :align: center

    Plot of :math:`\pi_\gamma(x)` of :eq:`simple_model_pdf_family` for
    various :math:`\gamma`'s.

.. _simple example model:

++++++++++++++++++++++++
Defining a `PyMC`_ model
++++++++++++++++++++++++

Since, this is our very first example we will use it as an opportunity to show
how `PyMC`_ can be used to define probabilistic models as well as MCMC sampling
algorithms. First of all let us mention that a `PyMC` model has to be packaged
either in a class or in a module. For the simple example we are considering, we
choose to use the module approach (see
:download:`examples/simple_model.py <../../examples/simple_model.py>`).
The model can be trivially defined using `PyMC` decorators. All we
have to do is define the logarithm of :math:`\pi_{\gamma}(x)`. We will call it
``mixture``. The contents of that module are:

.. code-block:: python
    :linenos:
    :emphasize-lines: 6,7

    import pymc
    import numpy as np
    import math

    @pymc.stochastic(dtype=float)
    def mixture(value=1., gamma=1., pi=[0.2, 0.8], mu=[-1., 2.],
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

This might look a little bit complicated but unfortunately one has to take care
of round-off errors when sump small numbers...
Notice that, we have defined pretty much every part of the mixture as an
independent variable. The essential variable that defines the family of
:eq:`simple_model_pdf_family` is ``gamma``. Well, you don't actually have to
call it ``gamma``, but we will talk about this later...

Let's import that module and see what we can do with it::

    >>> import simple_model as model
    >>> print model.mixture.parents
    {'mu': [-1.0, 2.0], 'pi': [0.2, 0.8], 'sigma': [0.01, 0.01], 'gamma': 1.0}

The final command shows you all the parents of the stochastic variable
``mixture``.
The stochastic variable mixture was assigned a value by default (see line 4
at the code block above). You can see the current value of the stochastic
variable at any time by doing::

    >>> print model.mixture.value
    1.0

If we started a `MCMC` chain at this point, this would be the initial value of
the chain. You can change it to anything you want by simply doing::

    >>> model.mixture.value = 0.5
    >>> print model.mixture.value
    0.5

To see the logarithm of the probability at the current state of the stochastic
variable, do::

    >>> print model.mixture.logp
    -111.11635344

Now, if you want to change, let's say, ``gamma`` to ``0.5`` all
you have to do is::

    >>> model.mixture.parents['gamma'] = 0.5
    >>> print model.mixture.gamma
    0.5

The logarithm of the probability should have changed also::

    >>> print model.mixture.logp
    -55.5581767201

.. _mcmc_attempt:

++++++++++++++++++++++++
Attempting to do `MCMC`_
++++++++++++++++++++++++

Let's load the model again and attempt to do `MCMC`_ using `PyMC`_'s
functionality::

    >>> import simple_model as model
    >>> import pymc
    >>> mcmc_sampler = pymc.MCMC(model)
    >>> mcmc_sampler.sample(1000000, thin=1000, burn=1000)

You should see a progress bar measuring the number of samples taken. It should
take about a minute to finish. We are actually doing :math:`10^6` `MCMC`_ steps,
we burn the first ``burn = 1000`` samples and we are looking at the chain
every ``thin = 1000`` samples (i.e., we are dropping everything in between). 
`PyMC`_ automatically picks a proposal (see `MCMC step methods`_) for you. For
this particular example it should have picked
:class:`pymc.step_methods.Metropolis` which corresponds to a simple random walk
proposal. There is no need to tune the parameters of the random walk since
`PyMC`_ is supposed to do that for you. In any case, it is possible to find the
right variance for the random walk, but you need to know exactly how far apart
the modes are...

You may look at the samples we've got by doing::

    >>> print mcmc_sampler.trace('mixture')[:]
    [ 1.9915846   1.93300521  2.09291872  2.05159841  2.06620882  1.88901709
      1.89521431  1.9631256   2.0363258   1.9756637   2.04818845  1.85036634
      1.98907666  1.82212356  1.97678175  1.99854311  1.92124829  2.02077581
      2.08536334  2.16664208  2.08328293  2.05378638  1.89437676  2.09555348
    ...

Now, let us plot the results::
    
    >>> import matplotlib.pyplot as plt
    >>> pymc.plot(mcmc_sampler)
    >>> plt.show()

The results are shown in `Simple Example MCMC Figure`_. Unless, you are
extremely lucky, you should have missed one of the modes...

.. _Simple Example MCMC Figure:
.. figure:: images/simple_model_mcmc.png
    :align: center

.. _E. T. Jaynes:
    E. T. Jaynes' http://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes>
.. _Probability Theory\: The Logic of Science: 
    http://omega.albany.edu:8008/JaynesBook.html
.. _MCMC:
    http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
.. _PyMC:
    http://pymc-devs.github.io/pymc/
.. _MCMC step methods:
    http://pymc-devs.github.io/pymc/extending.html#user-defined-step-methods
.. _Arnaud Doucet's collection:
    http://www.stats.ox.ac.uk/~doucet/smc_resources.html
.. _Sequential Monte Carlo Methods in Practice:
    http://books.google.com/books/about/Sequential_Monte_Carlo_Methods_in_Practi.html?id=BnWAcgAACAAJ
.. _Simulated Annealing:
    http://en.wikipedia.org/wiki/Simulated_annealing
