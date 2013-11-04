import pymc
import numpy as np
from diffusion_solver import DiffusionSourceLocationOnly as Solver

def make_model():
    # Construct the prior term
    location = pymc.Uniform('location', lower=[0,0], upper=[1,1])
    # The locations of the sensors
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    # The output of the model
    solver = Solver(X=X)
    @pymc.deterministic(plot=False)
    def model_output(value=None, loc=location):
        return solver(loc)
    # The hyper-parameters of the noise
    alpha = pymc.Exponential('alpha', beta=1.)
    beta = pymc.Exponential('beta', beta=1.)
    tau = pymc.Gamma('tau', alpha=alpha, beta=beta)
    # Load the observed data
    data = np.loadtxt('observed_data')
    # The observations at the sensor locations
    @pymc.stochastic(dtype=float, observed=True)
    def sensors(value=data, mu=model_output, tau=tau, gamma=1.):
        """The value of the response at the sensors."""
        return gamma * pymc.normal_like(value, mu=mu, tau=tau)
    return locals()
