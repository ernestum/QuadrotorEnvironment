import numpy as np
from quadrotor_environment.quadrotor_model import SysState, QuadrotorModel


class StateClippingWrapper:
    """
    A wrapper around the QuadrotorModel that clips its state.

    It supports clipping the position, velocity and angular velocity.
    """
    __slots__ = "quadrotor_model", "position_bounds", "velocity_bounds", "angular_velocity_bounds"

    def __init__(self, quadrotor_model: QuadrotorModel,
                 position_bounds=(-np.inf, np.inf),
                 velocity_bounds=(-5, 5), #TODO: choose infinity bounds by default
                 angular_velocity_bounds=(-20, 20)):

        def tuplify_bounds(b):
            if not isinstance(b, (list, tuple)):
                return (-b, b)
            return b
        position_bounds = tuplify_bounds(position_bounds)
        velocity_bounds = tuplify_bounds(velocity_bounds)
        angular_velocity_bounds = tuplify_bounds(angular_velocity_bounds)
        assert position_bounds[0] < position_bounds[1]
        assert velocity_bounds[0] < velocity_bounds[1]
        assert angular_velocity_bounds[0] < angular_velocity_bounds[1]
        self.quadrotor_model = quadrotor_model
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.angular_velocity_bounds = angular_velocity_bounds

    def next_state(self, x: SysState, u, h):
        """
        Computes the next state according to the model function and clips the state components to the specified bounds.

        :param x: The current state.
        :param u: The current action to apply.
        :param h: The amount of time to apply the action.
        :return: The next state after clipping.
        """
        x = self.quadrotor_model.next_state(x, u, h)
        x.position = np.clip(x.position, *self.position_bounds)
        x.velocity = np.clip(x.velocity, *self.velocity_bounds)
        x.angular_velocity = np.clip(x.angular_velocity, *self.angular_velocity_bounds)
        return x
