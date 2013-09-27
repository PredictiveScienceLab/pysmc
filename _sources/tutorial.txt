.. _tutorial:

========
Tutorial
========

 
Before moving forward, make sure you understand:

    + What is probability? That's a big question... If you feel like it,
      I suggest you skim through `E. T. Jaynes' \
      <http://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes>`_ book
      `Probability Theory: The Logic of Science \
      <http://omega.albany.edu:8008/JaynesBook.html>`_.
    + What is `MCMC \
      <http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_?
      You need to be familiar with the limitations of MCMC sampling.
    + Read the tutorial of `PyMC <http://pymc-devs.github.io/pymc/>`_.
      To the very least, you need to be able to construct probabilistic
      models using this package. For advanced applications, you need to be
      able to construct your own `step methods <http://pymc-devs.github.io/pymc/extending.html#user-defined-step-methods>`_
      .
    + Of course, I have to assume some familiarity with the so called 
      Sequential Monte Carlo (SMC) or Particle Methods. Many resources can
      be found online at `Arnaud Doucet's collection <http://www.stats.ox.ac.uk/~doucet/smc_resources.html>`_
      or at his book
      `Sequential Monte Carlo Methods in Practice <http://books.google.com/books/about/Sequential_Monte_Carlo_Methods_in_Practi.html?id=BnWAcgAACAAJ>`_.
      What exactly :mod:`pysmc` does is documented in :ref:`math`.


What is Sequential Monte Carlo?
-------------------------------

SMC is a very efficient and effective way to sample from complicated
probability distributions known up to a normalizing constant. The most
important complicating factor is multimodality. That is, probability
distributions that do not look at all like Gaussians.
