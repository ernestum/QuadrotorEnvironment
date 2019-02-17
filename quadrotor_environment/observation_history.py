import warnings

import numpy as np


class ObservationHistory:
    """
    The ObservationHistory keeps track of past state and action observations and view to the most recent history.

    It has support for observing multiple past states and actions. For the states as well for the actions it supports
    a differential view. We then do not observe past states or actions directly but rather their difference to
    each other. The most recent state and action is observed directly (the difference would be zero). Except for the
    case when the number of observed actions is 1 and differential. Then we just observe the most recent propeller
    acceleration.
    This corresponds to observing the first derivative. This is why the differences are being divided by the step size
    (dt). Since we often deal with very small step sizes, this would result in rather large numbers for the differential
    view. Therefore the history is rescaled by some factor.
    """
    __slots__ = "num_observed_past_states", "num_observed_past_actions", "differential_state_observation_history", \
                "state_history", "action_history", "dt_history", "differential_action_observation_history", \
                "history_scale"

    def __init__(self, num_observed_past_states: int = 1,
                 num_observed_past_actions: int = 0,
                 differential_state_observation_history: bool = False,
                 differential_action_observation_history: bool = False,
                 history_scale: float = .1):
        """
        Constructs a new ObservationHistory with the given parameters.

        Note:
            Some parameter configurations make no sense. When you want a differential view on your history, it must be
            longer than one entry. Warnings will be issued if you try those configurations but no exception is thrown.

        :param num_observed_past_states: The number of past states to be observed. Must be at least 1.
        :param num_observed_past_actions: The number of past actions to be observed. Can be 0 if the actions should not
        be observed.
        :param differential_state_observation_history: Flag to enable a differential view on the state history.
        :param differential_action_observation_history: Flag do enable a differential view on the action history.
        :param history_scale: The factor by which the differential history is multiplied to push the overall scale of
        those observations closer to the scale of the absolute state.
        """
        assert num_observed_past_states > 0
        assert num_observed_past_actions >= 0
        assert history_scale != 0
        if differential_state_observation_history and num_observed_past_states <= 1:
            warnings.warn("A differential state observation history makes no sense with a history length of 1")

        self.num_observed_past_states = num_observed_past_states
        self.num_observed_past_actions = num_observed_past_actions
        self.differential_state_observation_history = differential_state_observation_history
        self.differential_action_observation_history = differential_action_observation_history
        self.history_scale = history_scale

        self.state_history = []
        self.action_history = []
        self.dt_history = []

    def get_observation_with_history(self, state_observation: np.ndarray, relative_propeller_speed: np.ndarray, dt: float):
        """
        Updates the observation history with the provided state observation and action observation given that `dt` time
        has passed since we last observed something. Also computes a view on the most recent history based on the
        configuration of the history.

        :param state_observation: The vector describing the state observation.
        :param relative_propeller_speed: The vector describing the relative propeller speed.
        :param dt: The amount of time that has passed since we last observed something.
        :return: 1D-Vector containing the combined state and action observations with histories.
        """
        self.state_history.append(state_observation)
        self.action_history.append(relative_propeller_speed)
        self.dt_history.append(dt)

        if self.num_observed_past_actions > 0:
            if self.differential_action_observation_history and self.num_observed_past_actions == 1:
                observed_actions = ((np.asarray(self.action_history[-1]) - np.asarray(self.action_history[-2])) / self.dt_history[-1]) * self.history_scale
            elif self.differential_action_observation_history and self.num_observed_past_actions > 1:
                observed_actions = np.asarray(self.action_history[-self.num_observed_past_actions:])
                recent_dts = np.asarray(self.dt_history[-self.num_observed_past_actions + 1:])
                diff_hist = ((observed_actions[1:] - observed_actions[:-1]).T / recent_dts) * self.history_scale
                observed_actions = np.concatenate([diff_hist.ravel(), relative_propeller_speed]).ravel()
            else:
                observed_actions = np.asarray(self.action_history[-self.num_observed_past_actions:]).ravel()
        else:
            observed_actions = []

        observations = np.asarray(self.state_history[-self.num_observed_past_states:])
        if self.differential_state_observation_history and self.num_observed_past_states > 1:
            recent_dts = np.asarray(self.dt_history[-self.num_observed_past_states + 1:])
            diff_hist = ((observations[1:] - observations[:-1]).T / recent_dts) * self.history_scale
            observations = np.concatenate([diff_hist.ravel(), state_observation])
        observations = observations.ravel()

        return np.concatenate([observations, observed_actions])

    def reset(self, initial_observation, relative_propeller_speed, initial_dt):
        """
        Resets the histories based on the given initial observation, action and dt. The first couple of history entries
        are filled with those initial values to ensure that our view on the history always has the same size.

        :param initial_observation: The initial state observation to assume.
        :param relative_propeller_speed: The initial relative propeller speed observation to assume.
        :param initial_dt: The initial time difference to assume.
        """
        self.state_history = [initial_observation] * self.num_observed_past_states
        self.action_history = [relative_propeller_speed] * self.num_observed_past_actions
        self.dt_history = [initial_dt] * max(self.num_observed_past_actions, self.num_observed_past_states)

    def prune_history(self):
        """
        Removes some old entries from the history that are not needed any more for computing the view any more.
        This method is useful when using the ObservationHistory for very long observations. Then we need to prune from
        time to time to avoid filling up our memory.
        """
        prune_point = max(self.num_observed_past_actions, self.num_observed_past_states) + 1
        self.state_history = self.state_history[-prune_point:]
        self.action_history = self.action_history[-prune_point:]
        self.dt_history = self.dt_history[-prune_point:]
