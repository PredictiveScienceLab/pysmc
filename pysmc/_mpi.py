"""
.. _mpi::

+++++++++++++++++++++++++++++++
MPI related classes and methods
+++++++++++++++++++++++++++++++
"""


__docformat__ = 'reStructuredText'


__all__ = ['DistributedObject']


class DistributedObject(object):

    """
    This is a class that represents an object that is (potentially)
    distributed in parallel.
    """

    # The mpi class
    _mpi = None

    # The communicator
    _comm = None

    # The rank
    _rank = None

    # The size
    _size = None

    @property
    def mpi(self):
        """
        The MPI class.

        :getter:    Get the MPI class.
        :type:      :class:`mpi4py.MPI`
        """
        return self._mpi

    @property
    def comm(self):
        """
        The MPI communicator.

        :getter:    Get the MPI communicator.
        :type:      :class:`mpi4py.COMM`
        """
        return self._comm

    @property
    def size(self):
        """
        The size of the MPI pool.

        :getter:    Get the size of MPI pool.
        :type:      int
        """
        return self._size

    @property
    def rank(self):
        """
        The rank of this process.

        :getter:    Get the rank of this process.
        :type:      int
        """
        return self._rank

    @property
    def use_mpi(self):
        """
        Check if MPI is being used.

        :returns:   ``True`` if MPI is used and ``False`` otherwise.
        :rtype:     bool
        """
        return self.comm is not None

    def __init__(self, mpi=None, comm=None):
        """
        Initialize the object.

        See docstring for full details.
        """
        self._mpi = mpi
        if self.mpi is not None and comm is None:
            comm = self.mpi.COMM_WORLD
        self._comm = comm
        if self.use_mpi:
            self._rank = self.comm.Get_rank()
            self._size = self.comm.Get_size()
        else:
            self._rank = 0
            self._size = 1
