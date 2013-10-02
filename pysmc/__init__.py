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

.. automodule:: pysmc._particle_approximation
.. autoclass:: pysmc.ParticleApproximation
    :members:

.. automodule:: pysmc._smc
.. autoclass:: pysmc.SMC
    :members:

.. automodule:: pysmc._db
.. autoclass:: pysmc.DataBase
    :members:

.. _methods:

-------
Methods
-------

.. automodule:: pysmc._plot
.. autofunction:: pysmc.hist

.. automodule:: pysmc._misc
.. autofunction:: pysmc.try_to_array
.. autofunction:: pysmc.hist
.. autofunction:: pysmc.make_movie_from_db
.. autofunction:: pysmc.multinomial_resample
.. autofunction:: pysmc.kde

"""


__all__ = ['MCMCWrapper', 'SMC', 'ParticleApproximation', 'try_to_array',
           'get_var_from_particle_list', 'DataBase',
           'hist', 'make_movie_from_db', 'multinomial_resample', 'kde']


from ._misc import *
from ._mcmc_wrapper import *
from ._particle_approximation import *
from ._db import *
from ._smc import *
from ._plot import *
