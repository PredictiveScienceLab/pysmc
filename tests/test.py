"""
A first test to see how we can make this work...

Author:
    Ilias Bilionis

Date:
    9/21/2013
"""


import pymc



if __name__ == '__main__':
    from pymc.examples import disaster_model
    m = make_smc_model(disaster_model)
    print m.nodes
    print m.switchpoint.logp + m.early_mean.logp + m.late_mean.logp
    print m.logp
    g = pymc.graph.dag(m)
    g.write_png('disaster_model.png')