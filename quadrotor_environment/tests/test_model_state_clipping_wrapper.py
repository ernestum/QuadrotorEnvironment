import numpy as np

from quadrotor_environment.model_state_clipping_wrapper import StateClippingWrapper
from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
from quadrotor_environment.quadrotor_model import QuadrotorModel


def test_state_clipping():
    env = QuadrotorEnvironment(initial_state_parameters=dict(
        box_side_length=100, velocity=10, angular_velocity=50))
    wrapper = StateClippingWrapper(env.simulation_model,
                                   position_bounds=(-10, 10),
                                   velocity_bounds=(-5, 5),
                                   angular_velocity_bounds=(-20, 20))
    state = env.random_state()
    for _ in range(100):
        for s in range(100):
            state = wrapper.next_state(state, np.random.uniform(0, env.simulation_model.hovering_thrust*2, 4), 0.01)
            assert np.all(np.abs(state.angular_velocity) <= 20)
            assert np.all(np.abs(state.position) <= 10)
            assert np.all(np.abs(state.velocity) <= 5)


