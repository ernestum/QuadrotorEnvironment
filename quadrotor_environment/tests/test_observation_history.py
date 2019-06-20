import numpy as np

import pytest

from quadrotor_environment.observation_history import ObservationHistory


@pytest.mark.parametrize("num_observed_past_states", [1, 2, 3])
@pytest.mark.parametrize("num_observed_past_actions", [0, 1, 2, 3])
@pytest.mark.parametrize("differential_state_observation_history", [True, False])
@pytest.mark.parametrize("differential_action_observation_history", [True, False])
def test_observation_history_for_crashes(num_observed_past_states, num_observed_past_actions,
                                         differential_state_observation_history,
                                         differential_action_observation_history):
    obs_hist = ObservationHistory(num_observed_past_states=num_observed_past_states,
                                  num_observed_past_actions=num_observed_past_actions,
                                  differential_state_observation_history=differential_state_observation_history,
                                  differential_action_observation_history=differential_action_observation_history)
    obs_hist.reset(np.random.uniform(0, 1, 10), np.random.uniform(0, 1, 4), np.random.uniform(0.009, 0.011))

    for i in range(10):
        if i == 5:
            obs_hist.prune_history()
        obs_hist.get_observation_with_history(np.random.uniform(0, 1, 10), np.random.uniform(0, 1, 4), np.random.uniform(0.009, 0.011))


def test_observation_history_simple():
    obs_hist = ObservationHistory(5, num_observed_past_actions=0)
    obs_hist.reset(['a'], [0], 1)

    assert all(obs_hist.get_observation_with_history(['a'], [0], 1) == ['a'] * 5)

    assert all(obs_hist.get_observation_with_history(['b'], [0], 1) == ['a', 'a', 'a', 'a', 'b'])

    assert all(obs_hist.get_observation_with_history(['c'], [0], 1) == ['a', 'a', 'a', 'b', 'c'])

    obs_hist.reset(['d', 'c'], [0], 1)

    assert all(obs_hist.get_observation_with_history(['d', 'c'], [0], 1) == ['d', 'c'] * 5)


@pytest.mark.parametrize("num_observed_past_states", [1, 2, 3])
@pytest.mark.parametrize("num_observed_past_actions", [0, 1, 2])
@pytest.mark.parametrize("differential_state_observation_history", [True, False])
@pytest.mark.parametrize("differential_action_observation_history", [True, False])
def test_observation_history(num_observed_past_states, num_observed_past_actions,
                                         differential_state_observation_history,
                                         differential_action_observation_history):
    obs_hist = ObservationHistory(num_observed_past_states=num_observed_past_states,
                                  num_observed_past_actions=num_observed_past_actions,
                                  differential_state_observation_history=differential_state_observation_history,
                                  differential_action_observation_history=differential_action_observation_history)
    observations = [np.random.uniform(-100, 100, 4) for _ in range(5)]
    actions = [np.random.uniform(-100, 100, 4) for _ in range(5)]
    dts = [1, 2, 3, 4, 5]

    obs_hist.reset(observations[0], actions[0], dts[0])

    obs_with_hist = obs_hist.get_observation_with_history(observations[0], actions[0], dts[0])




