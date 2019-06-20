import numpy as np

import pytest

from quadrotor_environment.observation_mapping import ToObservationMap
from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment


@pytest.mark.parametrize("local_observations", [True, False])
@pytest.mark.parametrize("position_scale", np.linspace(0.01, 10, 5))
@pytest.mark.parametrize("velocity_scale", np.linspace(0.01, 10, 5))
@pytest.mark.parametrize("angular_velocity_scale", np.linspace(0.01, 10, 5))
@pytest.mark.parametrize("propeller_speed_scale", np.linspace(0.01, 10, 5))
@pytest.mark.parametrize("dt_scale", np.linspace(0.01, 10, 5))
@pytest.mark.parametrize("observation_age_scale", np.linspace(0.01, 10, 5))
def test_observation_scaling(local_observations, position_scale, velocity_scale, angular_velocity_scale,
                             propeller_speed_scale, dt_scale, observation_age_scale):
    m = ToObservationMap(local_observations=local_observations, huber_scaling=False, position_scale=position_scale,
                         velocity_scale=velocity_scale, angular_velocity_scale=angular_velocity_scale,
                         propeller_speed_scale=propeller_speed_scale, dt_scale=dt_scale,
                         observation_age_scale=observation_age_scale, observe_rotation=False,
                         rotation_observation_mode="rotmat_global")
    env = QuadrotorEnvironment()
    for _ in range(20):
        s = env.random_state()
        dt = np.random.uniform(0, 1)
        obs_age = np.random.uniform(0, 1)
        observation = m(s, dt, obs_age)

        if local_observations:
            assert np.isclose(np.linalg.norm(observation[0:3]), np.linalg.norm(s.position * position_scale))
            assert np.isclose(np.linalg.norm(observation[3:6]), np.linalg.norm(s.velocity * velocity_scale))
        else:
            assert np.all(np.isclose(observation[0:3], s.position * position_scale))
            assert np.all(np.isclose(observation[3:6], s.velocity * velocity_scale))

        assert np.all(np.isclose(observation[6:9], s.angular_velocity * angular_velocity_scale))
        assert np.all(np.isclose(observation[9:13], s.propeller_speed * propeller_speed_scale))
        assert np.isclose(observation[13], dt * dt_scale)
        assert np.isclose(observation[14], obs_age * observation_age_scale)

@pytest.mark.parametrize("local_observations", [True, False])
@pytest.mark.parametrize("position_scale", [0, 1])
@pytest.mark.parametrize("velocity_scale", [0, 1])
@pytest.mark.parametrize("angular_velocity_scale", [0, 1])
@pytest.mark.parametrize("propeller_speed_scale", [0, 1])
@pytest.mark.parametrize("dt_scale", [0, 1])
@pytest.mark.parametrize("observation_age_scale", [0, 1])
@pytest.mark.parametrize("observe_rotation", [True, False])
def test_observation_shape(local_observations, position_scale, velocity_scale, angular_velocity_scale,
                             propeller_speed_scale, dt_scale, observation_age_scale, observe_rotation):
    m = ToObservationMap(local_observations=local_observations, huber_scaling=False, position_scale=position_scale,
                         velocity_scale=velocity_scale, angular_velocity_scale=angular_velocity_scale,
                         propeller_speed_scale=propeller_speed_scale, dt_scale=dt_scale,
                         observation_age_scale=observation_age_scale, observe_rotation=observe_rotation,
                         rotation_observation_mode="rotmat_global")
    env = QuadrotorEnvironment()

    expected_size = 0
    expected_size += 3 if position_scale != 0 else 0
    expected_size += 3 if velocity_scale != 0 else 0
    expected_size += 3 if angular_velocity_scale != 0 else 0
    expected_size += 4 if propeller_speed_scale != 0 else 0
    expected_size += 1 if dt_scale != 0 else 0
    expected_size += 1 if observation_age_scale != 0 else 0
    expected_size += 9 if observe_rotation != 0 else 0

    observation = m(env.random_state(), np.random.uniform(0, 1), np.random.uniform(0, 1))

    assert observation.shape == (expected_size, )


def test_observation_offsets():
    pass #TODO
