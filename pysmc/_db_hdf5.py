""" 

.. _db_hdf5:

++++++++
HDF5 Database
++++++++

The class :class:`pysmc.HDF5DataBase` implements a simple database for
dumping SMC steps to HDF5 files. It uses `pytables <http://www.pytables.org>`_.
"""


__all__ = ['HDF5DataBase']


from . import DataBaseConcept
from . import ParticleApproximation


import tables as tb
import numpy as np


class HDF5DataBase(DataBaseConcept):

    """
    A database using HDF5.
    """

    def __init__(self):
        super(HDF5DataBase, self).__init__()

    @property
    def smc_state(self):
        return self.fd.get_node_attr('/', 'smc_state')

    @property
    def gamma_name(self):
        return self.smc_state['gamma_name']

    @property
    def gammas(self):
        return self.fd.root.gammas[:]

    @property
    def num_gammas(self):
        return self.fd.root.gammas.shape[0]

    def _get_group_of_step(self, i):
        """
        Return the group of step ``i``.
        """
        return self.fd.get_node('/steps/s' + str(i))

    def get_particle_approximation(self, i):
        pag = self._get_group_of_step(i).pa
        log_w = pag.log_w[:]
        num_particles = log_w.shape[0]
        particles = []
        for i in xrange(num_particles):
            s = {}
            for k in self.fd.list_nodes(pag.stochastics):
                s[k.name] = np.array(self.fd.get_node(k)[i])
            d = {}
            for k in self.fd.list_nodes(pag.deterministics):
                d[k.name] = np.array(self.fd.get_node(k)[i])
            p = {'stochastics': s,
                 'deterministics': d}
            particles.append(p)
        pa = ParticleApproximation(log_w=log_w, particles=particles)
        return pa

    def get_step_method_param(self, i):
        pag = self._get_group_of_step(i)
        smp = self.fd.get_node_attr(pag, 'step_func_params')
        return smp

    @staticmethod
    def load(filename):
        obj = HDF5DataBase()
        obj.fd = tb.open_file(filename, mode='a')
        return obj

    def initialize(self, filename, smc):
        self.fd = tb.open_file(filename, mode='w')
        self.fd.set_node_attr('/', 'smc_state',
                                   smc.__getstate__())
        self.fd.create_earray('/', 'gammas', atom=tb.Float64Atom(), shape=(0,))
        self.fd.create_earray('/', 'log_Zs', atom=tb.Float64Atom(), shape=(0,))
        self.fd.create_group('/', 'steps')
        self.fd.flush()

    def add(self, gamma, pa, smp, log_Z):
        self.fd.root.gammas.append([gamma])
        self.fd.root.log_Zs.append([log_Z])
        i = self.fd.root.gammas.shape[0] - 1
        sg = self.fd.create_group('/steps', 's' + str(i))
        pag = self.fd.create_group(sg, 'pa')
        self.fd.create_carray(pag, 'log_w', obj=pa.log_w)
        pa0 = pa.particles[0] 
        for k1 in pa0.keys():
            kpag = self.fd.create_group(pag, k1)
            v = pa0[k1]
            for k2 in v.keys():
                self.fd.create_carray(kpag, k2, obj=getattr(pa, k2))
        self.fd.set_node_attr(sg, 'step_func_params', smp)
        self.fd.flush()
