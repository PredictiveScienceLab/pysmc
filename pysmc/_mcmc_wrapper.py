"""

.. _mcmc_wrapper:

+++++++++++
MCMCWrapper
+++++++++++

:class:`pysmc.MCMCWrapper`
provides exactly the same functionality as :class:`pymc.MCMC`.
The only new thing is that it has the ability to get and set the current
state of MCMC from a dictionary describing the state. Basically, you should
not construct this class your self, :class:`pymc.SMC` will do it
automatically.

.. note:: It does not inherit from :class:`pymc.MCMC`. It simply stores
          a reference to a :class:`pymc.MCMC` object internally.

Here is a complete reference of the public members:
"""


__all__ = ['MCMCWrapper']


from pymc import MCMC
import warnings


class MCMCWrapper(object):

    """
    This is a wrapper class for :class:`pymc.MCMC`.

    :param mcmc_sampler:    The underlying MCMC sampler. If ``None``,
                            then it **must** be specified before using
                            an object of this class.
    :type mcmc_sampler:     :class:`pymc.MCMC`
    """

    # The underlying pymc.MCMC object.
    _mcmc_sampler = None

    @property
    def mcmc_sampler(self):
        """
        The underlying :class:`pymc.MCMC` object.

        :getter: Get the underlying MCMC object.
        :setter: Set the underlying MCMC object.
        :raises: :exc:`exceptions.TypeError`
        :type: :class:`pymc.MCMC`
        """
        return self._mcmc_sampler

    @mcmc_sampler.setter
    def mcmc_sampler(self, value):
        if not isinstance(value, MCMC):
            raise TypeError('You must provide a pymc.MCMC object!')
        value.assign_step_methods()
        self._mcmc_sampler = value

    @property
    def nodes(self):
        """The nodes of the model."""
        return self.mcmc_sampler.nodes

    @property
    def stochastics(self):
        """The stochastic variables of the model."""
        return self.mcmc_sampler.stochastics

    @property
    def deterministics(self):
        """The deterministic variables of the model."""
        return self.mcmc_sampler.deterministics

    @property
    def db(self):
        """The database of the MCMC sampler."""
        return self.mcmc_sampler.db

    @property
    def logp(self):
        """
        The log of the probability of the current state of the MCMC sampler.
        """
        return self.mcmc_sampler.logp

    @property
    def step_methods(self):
        """The step methods of the MCMC sampler."""
        return self.mcmc_sampler.step_methods

    def __init__(self, mcmc_sampler=None):
        """See doc of class."""
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

        :returns:   A dictionary ``state`` containing the current state of
                    MCMC. The keys of the dictionary are as follows:

                    - ``state['stochastics']``: A dictionary keeping the values of
                      all stochastic variables.
                    - ``state['deterministics']``: A dictionary keeping the values
                      of all deterministic variables.

        :rtype:     :class:`dict`

        """
        state = dict(stochastics={}, deterministics={})

        # The state of each stochastic parameter
        for s in self.stochastics:
            state['stochastics'][s.__name__] = s.value

        # The state of each deterministic
        for d in self.deterministics:
            state['deterministics'][d.__name__] = d.value

        return state

    def set_state(self, state):
        """
        Set the state of the sampler.

        :parameter state:   A dictionary describing the state of the
                            sampler. Look at
                            :meth:`pysmc.MCMCWrapper.get_state()` for the
                            appropriate format.
        :type state: :class:`dict`
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
        :meth:`pymc.MCMC.sample()` but with defaults that are more suitable
        for :class:`pysmc.SMC`. For example, you typically do not want to allow
        :meth:`pymc.MCMC.sample()`
        to tune the parameters of the step methods.
        It is :class:`SMC` that should do this.
        This is because the transition kernels need to retain their
        invariance properties. They can't have parameters that change
        on the fly.

        :param iter:            The number of iterations to be run.
        :type iter:             int
        :param burn:            The number of samples to be burned before we
                                start storing things to the database. There
                                is no point in :class:`pysmc.SMC` to have this
                                equal to anything else than 0.
        :type burn:             int
        :param thin:            Store to the database every ``thin``
                                samples. If ``None`` then we just store the
                                last sample. That is the method will set
                                ``thin = iter``.
        :type thin:             int
        :param tune_interval:   The tuning interval. The default is not to
                                tune anything. Do not change it.
        :type tune_interval:    int
        :param tune_throughout: Tune during all the samples we take. The
                                default is ``False``. Do not change it.
        :type tune_throughout:  bool
        :param burn_till_tuned: Burn samples until the parameters get tuned.
                                The default is no. Do not change it.
        :type burn_till_tuned:  bool
        :param stop_tuning_after:   Stop tuning after a certain number of
                                    iterations. No point in setting this.
        :type stop_tuning_after:    int
        :param verbose:         How much verbosity you like.
        :type verbose:          int
        :param progress_bar:    Show the progress bar or not.
        :type progress_bar:     bool
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
        """
        Draw from the prior of the model.

        :raises: :exc:`exceptions.AttributeError` if the action is not
                 possible.
        """
        self.mcmc_sampler.draw_from_prior()
