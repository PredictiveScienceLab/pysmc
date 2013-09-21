"""
Initialize the pysmc module.

Author:
    Ilias Bilionis

Date:
    9/20/2013
"""


__all__ = ['SMCStochastic', 'make_smc_model']


class SMCStochastic(pymc.Stochastic):
    """
    A wrapper for a stochastic class used in SMC.
    """

    # The gamma parameter
    _gamma = None

    # The underlying stochastic variable
    _rv = None

    @property
    def gamma(self):
        return self.parents['gamma']

    @gamma.setter
    def gamma(self, value):
        self.parents['gamma'] = float(value)

    @property
    def rv(self):
        return self._rv

    def _smc_logp(self, **args):
        return (args['gamma'] - 1.) * self.rv.get_logp()

    def __init__(self, rv):
        """
        Initialize the object.
        """
        self._rv = rv
        super(SMCStochastic, self).__init__(logp=self._smc_logp,
                                            doc=self.__doc__,
                                            name=str(rv) + '_wrapper',
                                            parents={'gamma' : 0, 'rv' : rv},
                                            value=rv.value,
                                            dtype=rv.dtype,
                                            rseed=rv.rseed,
                                            observed=rv.observed,
                                            plot=rv.plot,
                                            verbose=rv.verbose)


def make_smc_model(module):
    model = pymc.Model(module)
    for rv in model.nodes:
        if isinstance(rv, pymc.Stochastic) and rv.observed:
            rv_smc = SMCStochastic(rv)
    module.rv_smc = rv_smc
    return pymc.Model(module)