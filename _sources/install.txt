.. _install:

============
Installation
============

We describe the necessary steps required to succesfully install
:mod:`pysmc`. The most important prior step is to satisfy the dependencies
of the code.


.. _depend:

-----------
Dependecies
-----------

There are two different categories of packages on which we rely. The 
:ref:`required` ones have to be there no matter what. The :ref:`optional`
ones can be skipped but without them you will loose some of the
functionality of :mod:`pysmc` or/and you will not be able to run all the
examples in the :mod:`tutorial`.

.. _required:

Required
++++++++
The following packages are required:

    + `Numpy <www.numpy.org>`_ for linear algebra.
    + `SciPy <www.scipy.org>`_ for some root finding methods.
    + `PyMC <http://pymc-devs.github.io/pymc/>`_ for general probabilistic
      model definition and MCMC sampling. We suggest that you install the
      `latest PyMC version from GitHub <https://github.com/pymc-devs/pymc>`_
      .


.. _optional:

Optional
++++++++
The following packages are optional but highly recommended:

    + `MPI4PY <mpi4py.scipy.org>`_ to enable the parallel capabilities of
      :mod:`pysmc`.
    + `matplotlib <matplotlib.org>`_ for plotting.
    + `FiPy <http://www.ctcms.nist.gov/fipy/>`_ in order to run the
      :ref:`diffusion_example`.

.. _final_steps:

Final Steps
+++++++++++
As soon as you are done installing the packages above, you can fetch
:mod:`pysmc` from `GitHub <https://github.com/ebilionis/pysmc>`_ by::

    git clone https://github.com/ebilionis/pysmc.git

Then, all you have to do is enter the ``pysmc`` directory that was created
and run::

    python setup.py install

If you want to put the code in an non default location, simply do::

    python setup.py install --prefix=/path/to/your/directory

If you do the latter, make sure you update your ``PYTHONPATH`` variable::

    export PYTHONPATH=/path/to/your/directory/lib/python2.7/site-packages:$PYTHONPATH
