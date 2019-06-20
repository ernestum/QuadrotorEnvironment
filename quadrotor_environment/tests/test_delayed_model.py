import numpy as np

import pytest
from quadrotor_environment.model_delay_wrapper import ModelDelayWrapper


class LinearTestModel():
    def next_state(self, x, action, h):
        x += action * h
        return x


class DiscreteTestModel:
    def next_state(self, x, action, h):
        x += action * int(h)
        return x


@pytest.mark.parametrize("obs_delay", [0, 1, 2])
@pytest.mark.parametrize("action_delay", [0, 1, 2])
def test_DiscreteModel(obs_delay, action_delay):
    model = DiscreteTestModel()

    delayer = ModelDelayWrapper(model, controller_jitter=0, controller_period=1,
                                observation_delay=obs_delay, observation_jitter=0,
                                action_delay=action_delay, action_jitter=0)
    obs0, _, _ = delayer.reset('x', '')

    assert obs0 == ''

    assert delayer.compute_past_state(0) == ''
    assert delayer.compute_past_state(1) == 'x'
    assert delayer.compute_past_state(2) == 'xx'

    actions = 'b______'
    expected_observations = "x" * (obs_delay + action_delay) + "b" + "_" * (len(actions) - obs_delay - action_delay - 1)
    expected_states = "x" * action_delay + "b" + "_" * (len(actions) - action_delay - 1)
    actual_observations = ""
    actual_states = ""

    for action in actions:
        actual_observations, _, _ = delayer.step(action)
        actual_states += delayer.compute_past_state(delayer.time)[-1]

    assert actual_observations == expected_observations
    assert actual_states == expected_states


def test_zero_delay_sanity_check_linear():
    actions = [1, 2, 0, -1, -2]

    model = LinearTestModel()

    delayer = ModelDelayWrapper(model, controller_period=1)
    delayer_observations = [delayer.reset(0, 0)[0]]

    raw_state = 0
    raw_observations = [raw_state]
    for action in actions:
        delayer_observation, dt, obs_age = delayer.step(action)
        assert np.isclose(dt, 1)
        delayer_observations.append(delayer_observation)

        raw_state = model.next_state(raw_state, action, 1)
        raw_observations.append(raw_state)

    assert all(np.isclose(raw_observations, delayer_observations))


@pytest.mark.parametrize("obs_delay", [0, 1, 2])
@pytest.mark.parametrize("action_delay", [0, 1, 2])
def test_LinearModel(obs_delay, action_delay):
    model = LinearTestModel()

    delayer = ModelDelayWrapper(model, controller_jitter=0, controller_period=1,
                                observation_delay=obs_delay, observation_jitter=0,
                                action_delay=action_delay, action_jitter=0)
    obs0, _, _ = delayer.reset(0, 0)

    assert obs0 == 0

    assert delayer.compute_past_state(0) == 0
    assert delayer.compute_past_state(1) == 0
    assert delayer.compute_past_state(2) == 0

    actions = [1, 2, 0, -1, -2]
    undelayed_states = [1, 3, 3, 2, 0]
    expected_states = [0] * action_delay + undelayed_states[:len(actions) - action_delay]
    actual_states = []
    expected_observations = [0] * (obs_delay) + expected_states[:len(actions) - obs_delay]
    actual_observations = []

    for action in actions:
        actual_observations.append(delayer.step(action)[0])
        actual_states.append(delayer.compute_past_state(delayer.time))

    assert all(np.isclose(expected_observations, actual_observations))
    assert all(np.isclose(expected_states, actual_states))

@pytest.mark.parametrize("obs_delay", [1, 2])
@pytest.mark.parametrize("action_delay", [1, 2])
@pytest.mark.parametrize("control_jitter", [0, 0.1, 1.])
@pytest.mark.parametrize("observation_jitter", [0, 0.1, 1.])
@pytest.mark.parametrize("action_jitter", [0, 0.1, 1.])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_LinearModelWithJitter(obs_delay, action_delay, control_jitter, observation_jitter, action_jitter, seed):
    np.random.seed(seed)
    model = LinearTestModel()

    delayer = ModelDelayWrapper(model, controller_jitter=control_jitter, controller_period=1,
                                observation_delay=obs_delay, observation_jitter=observation_jitter,
                                action_delay=action_delay, action_jitter=action_jitter)
    obs0, _, _ = delayer.reset(0, 0)
    assert obs0 == 0

    actions = [1, 2, 0, -1, -2]

    for action in actions:
        delayer.step(action)


