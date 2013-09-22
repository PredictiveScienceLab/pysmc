"""A class that solves the following diffusion problem:

    dphi/dt = div(grad(phi)) + S(x,t),

    where S(x, t) is the source term and it is modelled as:

        S(x, t) = s / (2 * PI * h ** 2) * exp( - ||xc - x||^2/(2 * h ** 2), for
        t in [0, tau]
        S(x, t) = 0, for t > tau.

Author:
    Ilias Bilionis

Date:
    1/26/2013
"""

from fipy import Grid2D
from fipy import CellVariable
from fipy import TransientTerm
from fipy import DiffusionTerm
from fipy import ExplicitDiffusionTerm
from fipy import numerix
import numpy as np
import math


class Diffusion(object):
    """Solve the Diffusion problem."""

    # Number of cells
    _num_cells = None

    # The discretization step
    _dx = None

    # The mesh
    _mesh = None

    # The solution vector
    _phi = None

    # The source term
    _source = None

    # The equation
    _eq = None

    # The time step
    _dt = None

    # The maximum solution time
    _max_time = None

    # The indices of the variables you want to observe
    _idx = None

    # The sensor times
    _T = None

    @property
    def num_cells(self):
        """Get the number of cells."""
        return self._num_cells

    @property
    def dx(self):
        """Get the discretization steps."""
        return self._dx

    @property
    def mesh(self):
        """Get the mesh."""
        return self._mesh

    @property
    def phi(self):
        """Get the solution."""
        return self._phi

    @property
    def source(self):
        """Get the source term."""
        return self._source

    @property
    def eq(self):
        """Get the equation."""
        return self._eq

    @property
    def dt(self):
        """Get the timestep."""
        return self._dt

    @property
    def idx(self):
        """Get the indices of the observed variables."""
        return self._idx

    @property
    def max_time(self):
        """Get the maximum solution time."""
        return self._max_time

    @property
    def T(self):
        """Get the sensor measurement times."""
        return self._T

    def __init__(self, X=None, T=None, time_steps=5, max_time=0.4, num_cells=25, L=1.):
        """Initialize the objet.

        Keyword Arguments:
            X           ---     The sensor locations.
            T           ---     The sensor measurment times.
            time_steps  ---     How many timesteps do you want to measure.
            max_time    ---     The maximum solution time.
            num_cells   ---     The number of cells per dimension.
            L           ---     The size of the computational domain.
        """
        assert isinstance(num_cells, int)
        self._num_cells = num_cells
        assert isinstance(L, float) and L > 0.
        self._dx = L / self.num_cells
        self._mesh = Grid2D(dx=self.dx, dy=self.dx, nx=self.num_cells,
                ny=self.num_cells)
        self._phi = CellVariable(name='solution variable', mesh=self.mesh)
        self._source = CellVariable(name='source term', mesh=self.mesh,
                hasOld=True)
        self._eqX = TransientTerm() == ExplicitDiffusionTerm(coeff=1.) + self.source
        self._eqI = TransientTerm() == DiffusionTerm(coeff=1.) + self.source
        self._eq = self._eqX + self._eqI
        assert isinstance(max_time, float) and max_time > 0.
        self._max_time = max_time
        #self.max_time / time_steps #.
        if X is None:
            idx = range(self.num_cells ** 2)
        else:
            idx = []
            x1, x2 = self.mesh.cellCenters
            for x in X:
                dist = (x1 - x[0]) ** 2 + (x2 - x[1]) ** 2
                idx.append(np.argmin(dist))
        self._idx = idx
        if T is None:
            T = np.linspace(0, self.max_time, time_steps)[1:]
        self._max_time = T[-1]
        self._T = T
        self._dt = self.T[0] / time_steps
        self.num_input = 5
        self.num_output = len(self.T) * len(self.idx)

    def __call__(self, x):
        """Evaluate the solver."""
        self.phi.value = 0.
        x_center = x[:2]
        tau = x[2]
        h = x[3]
        s = x[4]
        x, y = self.mesh.cellCenters
        self.source.value = 0.5 * s / (math.pi * h ** 2) * numerix.exp(-0.5 *
                ((x_center[0] - x) ** 2 + (x_center[1] - y) ** 2)
                / (2. * h ** 2))
        y_all = []
        t = 0.
        self._times = []
        next_time = 0
        while t < self.max_time:
            t += self.dt
            add = False
            if next_time < len(self.T) and t >= self.T[next_time]:
                t = self.T[next_time]
                next_time += 1
                add = True
            if t >= tau:
                self.source.value = 0.
            self.eq.solve(var=self.phi, dt=self.dt)
            if add:
                self._times.append(t)
                y_all.append([self.phi.value[i] for i in self.idx])
        y = np.hstack(y_all)
        y = y.reshape((self.T.shape[0], len(self.idx))).flatten(order='F')
        return y


class DiffusionSourceLocationOnly(Diffusion):
    """A Diffusion solver that allows only the source location to vary."""

    # The source signal
    _s = None

    # The source spread
    _h = None

    # The source time
    _tau = None

    @property
    def s(self):
        """Get the source signal."""
        return self._s

    @property
    def h(self):
        """Get the source spread."""
        return self._h

    @property
    def tau(self):
        """Get the source time."""
        return self._tau

    def __init__(self, s=2., h=0.05, tau=0.3, X=None, T=None, time_steps=5,
            max_time=0.4, num_cells=25, L=1.):
        """Initialize the object.

        Keyword Arguments:
            s   ---     The source signal.
            h   ---     The source spread.
            tau ---     The source time.
        The rest are as in Diffusion.
        """
        assert isinstance(s, float) and s > 0.
        self._s = s
        assert isinstance(h, float) and h > 0.
        self._h = h
        assert isinstance(tau, float) and tau > 0.
        self._tau = tau
        super(DiffusionSourceLocationOnly, self).__init__(X=X, T=T,
                time_steps=time_steps, max_time=max_time, num_cells=num_cells,
                L=L)
        self.num_input = 2

    def __call__(self, x):
        """Evaluate the object."""
        x = np.array(x)
        x = np.hstack([x, [self.tau, self.h, self.s]])
        return super(DiffusionSourceLocationOnly, self).__call__(x)
