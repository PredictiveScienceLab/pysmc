"""
A wrapper for pymc.MCMC that is suitable for SMC.

Author:
    Ilias Bilionis

Date:
    9/22/2013
"""


__all__ = ['MCMCWrapper']


from pymc import MCMC
import warnings


class MCMCWrapper(object):

    """
    A wrapper class for pymc.MCMC.

    We need it because we do not want to overcomplicate the code in
    pysmc.SMC and we do not want to touch the code in pymc.

    However, this class uses extremely detailed knowledge of the
    internals of pymc.
    """

    # The underlying pymc.MCMC object.
    _mcmc_sampler = None

    @property
    def mcmc_sampler(self):
        return self._mcmc_sampler

    @mcmc_sampler.setter
    def mcmc_sampler(self, value):
        assert isinstance(value, MCMC)
        self._mcmc_sampler = value

    @property
    def nodes(self):
        """Expose the nodes of the MCMC sampler."""
        return self.mcmc_sampler.nodes

    @property
    def stochastics(self):
        """Expose the stochastic variables of the MCMC sampler."""
        return self.mcmc_sampler.stochastics

    @property
    def deterministics(self):
        """Expose the deterministic variables of the MCMC sampler."""
        return self.mcmc_sampler.deterministics

    @property
    def db(self):
        """Expose the database of the MCMC sampler."""
        return self.mcmc_sampler.db

    @property
    def logp(self):
        """Expose the log of the probability of the MCMC sampler."""
        return self.mcmc_sampler.logp

    @property
    def step_methods(self):
        """Expose the step methods of the MCMC sampler."""
        return self.mcmc_sampler.step_methods

    def __init__(self, mcmc_sampler=None):
        """
        Initialize the object.

        Parameters
        ----------
        mcmc_sampler    :   pymc.MCMC
                            The underlying MCMC sampler. If ``Non``,
                            then it **must** be specified before using
                            an object of this class.
        """
        self.mcmc_sampler = mcmc_sampler

    def get_state(self):
        """
        Get a dictionary describing the state of the sampler.

        Keep in mind that we do not copy the internal parameters of the
        sampler (i.e., the step methods, their parameters, etc.). We
        only copy the values of all the Stochastics and all the
        Deterministics. On contrast pymc.MCMC.get_state() copies the
        internal parameters and the Stochastics. It does not copy the
        Deterministics. The deterministics are needed because in most
        of our examples they are going to be very expensive to
        revaluate.

        Returns
        -------
        A dictionary ``state`` containing the current state of MCMC.
        The keys of the dictionary are as follows:
            - state['stochastics']: A dictionary keeping the values of
              all stochastic variables.
            - state['deterministics']: A dictionary keeping the values
              of all deterministic variables.
        """
        state = dict(stochastics={}, deterministics={})

        # The state of each stochastic parameter
        for s in self.stochastics:
            state['stochastics'][s.__name__] = s.value.copy()

        # The state of each deterministic
        for d in self.deterministics:
            state['deterministics'][d.__name__] = d.value.copy()

        return state

    def set_state(self, state):
        """
        Set the state of the sampler.

        Parameters
        ----------
        state       :   dict
                        A dictionary describing the state of the
                        sampler. Preferably returned by ``get_state()``.
        """
        # Restore stochastic parameters state
        stoch_state = state.get('stochastics', {})
        for sm in self.stochastics:
            try:
                sm.value = stoch_state[sm.__name__]
            except:
                warnings.warn(
        'Failed to restore state of stochastic %s from %s backend' %
                    (sm.__name__, self.db.__name__))

        # Restore the deterministics
        det_state = state.get('deterministics', {})
        for dm in self.deterministics:
            try:
                dm._value.force_cache(det_state[dm.__name__])
            except:
                warnings.warn(
        'Failed to restore state of deterministic %s from %s backend' %
                    (dm.__name__, self.db.__name__))

    def sample(self, iter, burn=0, thin=None, tune_interval=1,
               tune_throughout=False, save_interval=None,
               burn_till_tuned=False, stop_tuning_after=0,
               verbose=0, progress_bar=False):
        """
        Sample ``iter`` times from the model.

        This is basically exactly the same call as
        ``pymc.MCMC.sample()`` but with defaults that are more suitable
        for SMC. For example, you typically do not want to allow pymc
        to tune the parameters of the step methods. SMC should do this.
        This is because the transition kernels need to retain their
        invariance properties. They can't have parameters that change
        on the fly.

        Parameters
        ----------
        iter            :   int
                            The number of iterations to be run.
        burn            :   int
                            The number of samples to be burned before we
                            start storing things to the database. There
                            is no point in SMC to have this equal to
                            anything else than 0.
        thin            :   int
                            Store to the database every ``thin``
                            samples. If ``None`` then we just store the
                            last sample. That is the method will set
                            ``thin = iter``.
        tune_interval   :   int
                            The tuning interval. The default is not to
                            tune anything. Do not change it.
        tune_throughout :   bool
                            Tune during all the samples we take. The
                            default is ``False``. Do not change it.
        burn_till_tunned:   bool
                            Burn samples until the parameters get tuned.
                            The default is no. Do not change it.
        stop_tuning_after:  int
                            Stop tuning after a certain number of
                            iterations. No point in setting this.
        verbose         :   int
                            How much verbosity you like.
        progress_bar    :   bool
                            Show the progress bar or not.
        """
        if thin is None:
            thin = iter
        self.mcmc_sampler.sample(iter, burn=burn, thin=thin,
                                 tune_interval=tune_interval,
                                 tune_throughout=tune_throughout,
                                 save_interval=save_interval,
                                 burn_till_tuned=burn_till_tuned,
                                 stop_tuning_after=stop_tuning_after,
                                 verbose=verbose,
                                 progress_bar=progress_bar)

    def draw_from_prior(self):
        """Expose the corresponding function of pymc.MCMC."""
        try:
            self.mcmc_sampler.draw_from_prior()
        except:
            pass
