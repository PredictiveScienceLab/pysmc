"""
The definition of the posterior of the reaction kinetics problem.
"""


import pymc
from reaction_kinetics_solver import *


def make_model():
    import cPickle as pickle
    with open('reaction_kinetics_data.pickle', 'rb') as fd:
        data = pickle.load(fd)
    y_obs = data['y_obs']
    # The priors for the reaction rates:
    k1 = pymc.Lognormal('k1', mu=2, tau=1./(10. ** 2), value=5.)
    k2 = pymc.Lognormal('k2', mu=4, tau=1./(10. ** 2), value=5.)
    # The noise term
    #sigma = pymc.Uninformative('sigma', value=1.)
    sigma = pymc.Exponential('sigma', beta=1.)
    # The forward model
    re_solver = ReactionKineticsSolver()
    @pymc.deterministic
    def model_output(value=None, k1=k1, k2=k2):
        return re_solver(k1, k2)
    # The likelihood term
    @pymc.stochastic(observed=True)
    def output(value=y_obs, mod_out=model_output, sigma=sigma, gamma=1.):
        return gamma * pymc.normal_like(y_obs, mu=mod_out, tau=1/sigma ** 2)
    return locals()


if __name__ == '__main__':
    # Generate the observations
    import cPickle as pickle
    re_solver = ReactionKineticsSolver()
    k1 = 2 
    k2 = 4
    data = {}
    data['y_obs'] = re_solver(k1, k2)
    with open('reaction_kinetics_data.pickle', 'wb') as fd:
        pickle.dump(data, fd)
