"""
.. module:: pysmc
    :synopsis: The main PySMC module.

.. moduleauthor:: Ilias Bilionis <ebilionis@mcs.anl.gov>


.. _reference:
=========
Reference
=========

The goal of :mod:`pysmc` is to implement Sequential Monte Carlo (SMC)
techniques on top of the Monte Carlo (MC) package
`PyMC <http://pymc-devs.github.io/pymc/>`_. The manual assumes that the user
is already familiar with the way PyMC works. You are advised to read their
tutorial before going on. A nice place to start with :mod:`pysmc` is
our :ref:`tutorial`.


.. _classes:
-------
Classes
-------
Here is complete reference of all the classes included in :mod:`pysmc`.

.. automodule:: pysmc._mcmc_wrapper
.. autoclass:: pysmc.MCMCWrapper
    :members:

.. automodule:: pysmc._smc
.. autoclass:: pysmc.SMC
    :members:

"""


__all__ = ['MCMCWrapper', 'SMC']


from ._mcmc_wrapper import *
from ._smc import *
