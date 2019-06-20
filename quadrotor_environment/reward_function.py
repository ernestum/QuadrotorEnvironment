import warnings

import numpy as np
from numpy.linalg import norm

from quadrotor_environment.quadrotor_model import SysState
from quadrotor_environment.utils import inverted_huber_loss, quat2eulzyx


class RewardFunction:
    """
    Parameterizable reward function for a quadrotor.

    The reward is a weighted sum over the state components. Since we want to keep most state components small in order
    to hover around the origin or to maintain an attitude, the NEGATIVE weighted sum is used. Most state components have
    individual weights for their horizontal and the vertical part since flying side to side is inherently different than
    flying up an down.
    """
    __slots__ = 'compute_reward', 'simple_reward_terms', 'propeller_speed_reward_terms', 'rotation_reward_terms'

    def __init__(self,
                 reward_scale: float =0.00076,
                 
                 position: float = 0,
                 velocity: float = 0,
                 angular_velocity: float = 0,

                 position_h: float = 9.5,
                 position_v: float = 2.2,
                 velocity_h: float = 0.9,
                 velocity_v: float = 1.5,
                 rotation_h: float = 2,
                 rotation_v: float = 2,
                 angular_velocity_h: float = 2,
                 angular_velocity_v: float = 2,
                 propeller_speed: float = 0,
                 propeller_speed_deviation: float = 1,
                 propeller_acceleration: float = 1,
                 inverted_huber_reward_scaling: bool = True):
        """
        Creates a new reward function with the given state component weights.

        Note:
            To compute the reward components for the rotation component, euler angles (e_x, e_y, e_z) are used, which
            are applied in the order ZYX to ensure that the yaw is valid in the global reference frame.

        :param reward_scale: Global scale by which all other rewards are multiplied.
        Some algorithms like this one tuned.
        :param position_h: weight for the horizontal position component, which is `norm(x, y)`.
        :param position_v: weight for the vertical position component, which is `abs(z)`.
        :param velocity_h: weight for the horizontal velocity which is `norm(v_x, v_y)`.
        :param velocity_v: weight for the vertical velocity, which is `abs(v_z)`.
        :param rotation_h: weight for the horizontal rotation (without yaw), which is `norm(e_x, e_y)`.
        :param rotation_v: weight for the vertical rotation around the yaw axis, which is `abs(e_z)`.
        :param angular_velocity_h: weight for the horizontal angular velocity without heading, which is
        `norm(va_x, va_y)`.
        :param angular_velocity_v: weight for the vertical angular velocity (around yaw axis), which is `abs(va_z)`.
        :param propeller_speed: weight for the propeller speed (relative) which is `norm(relative_propeller_speed)`.
        :param propeller_speed_deviation: weight for the squared deviation from the mean propeller speed of the
        individual propeller speeds. This punishes strong steering.
        :param propeller_acceleration: Squared sum of all propeller accelerations. This punishes rapid changes of the
        propeller speeds which hurts the motors and motor controllers.
        :param inverted_huber_reward_scaling: Flag that enables inverted huber reward scaling to some of the reward
        components. This includes the position, velocity, angular velocity and rotation.
        """
        assert reward_scale != 0

        if (not np.isclose(position_h, 0) or not np.isclose(position_v, 0)) and not np.isclose(position, 0):
            warnings.warn("horizontal or vertical position reward is used in together with combined position reward!")
            
        if (not np.isclose(velocity_h, 0) or not np.isclose(velocity_v, 0)) and not np.isclose(velocity, 0):
            warnings.warn("horizontal or vertical velocity reward is used in together with combined velocity reward!")
            
        if (not np.isclose(angular_velocity_h, 0) or not np.isclose(angular_velocity_v, 0)) and not \
                np.isclose(angular_velocity, 0):
            warnings.warn("horizontal or vertical angular_velocity reward is used in together with combined angular_"
                          "velocity reward!")

        self.simple_reward_terms = simple_reward_terms = []
        if not np.isclose(position, 0):
            simple_reward_terms.append(lambda state: norm(state.position) * position)
        if not np.isclose(position_h, 0):
            simple_reward_terms.append(lambda state: norm(state.position[:2]) * position_h)
        if not np.isclose(position_v, 0):
            simple_reward_terms.append(lambda state: np.abs(state.position[2]) * position_v)

        if not np.isclose(velocity, 0):
            simple_reward_terms.append(lambda state: norm(state.velocity) * velocity)
        if not np.isclose(velocity_h, 0):
            simple_reward_terms.append(lambda state: norm(state.velocity[:2]) * velocity_h)
        if not np.isclose(velocity_v, 0):
            simple_reward_terms.append(lambda state: np.abs(state.velocity[2]) * velocity_v)

        if not np.isclose(angular_velocity, 0):
            simple_reward_terms.append(lambda state: norm(state.angular_velocity) * angular_velocity)
        if not np.isclose(angular_velocity_h, 0):
            simple_reward_terms.append(lambda state: norm(state.angular_velocity[:2]) * angular_velocity_h)
        if not np.isclose(angular_velocity_v, 0):
            simple_reward_terms.append(lambda state: np.abs(state.angular_velocity[2]) * angular_velocity_v)

        self.propeller_speed_reward_terms = propeller_speed_reward_terms = []
        if not np.isclose(propeller_speed, 0):
            propeller_speed_reward_terms.append(lambda relative_propeller_speed, relative_propeller_acceleration:
                                                propeller_speed * norm(relative_propeller_speed))

        if not np.isclose(propeller_speed_deviation, 0):
            propeller_speed_reward_terms.append(lambda relative_propeller_speed, relative_propeller_acceleration:
                                                propeller_speed_deviation * np.sum((relative_propeller_speed - np.mean(relative_propeller_speed)) ** 2))

        if not np.isclose(propeller_acceleration, 0):
            propeller_speed_reward_terms.append(lambda relative_propeller_speed, relative_propeller_acceleration:
                                                propeller_acceleration * np.sum(relative_propeller_acceleration**2))

        self.rotation_reward_terms = rotation_reward_terms = []
        if not np.isclose(rotation_h, 0):
            rotation_reward_terms.append(lambda eul_zyx: rotation_h * np.linalg.norm(eul_zyx[1:2]))

        if not np.isclose(rotation_v, 0):
            rotation_reward_terms.append(lambda eul_zyx: rotation_v * np.abs(eul_zyx[0]))

        if inverted_huber_reward_scaling:
            def compute_reward(state, relative_propeller_speed, relative_propeller_acceleration):
                reward = np.sum(inverted_huber_loss(term(state)) for term in simple_reward_terms)
                if len(rotation_reward_terms) > 0:
                    eul_zyx = quat2eulzyx(state.rotation)
                    reward += sum(inverted_huber_loss(term(eul_zyx)) for term in rotation_reward_terms)
                reward += np.sum(term(relative_propeller_speed, relative_propeller_acceleration) for term
                                 in propeller_speed_reward_terms)
                return -reward_scale * reward
        else:
            def compute_reward(state, relative_propeller_speed, relative_propeller_acceleration):
                reward = np.sum(term(state) for term in simple_reward_terms)
                if len(rotation_reward_terms) > 0:
                    eul_zyx = quat2eulzyx(state.rotation)
                    reward += np.sum(term(eul_zyx) for term in rotation_reward_terms)
                reward += np.sum(term(relative_propeller_speed, relative_propeller_acceleration) for term
                                 in propeller_speed_reward_terms)
                return -reward_scale * reward

        self.compute_reward = compute_reward

    def __call__(self, state: SysState, relative_propeller_speed: np.ndarray,
                 relative_propeller_acceleration: np.ndarray) -> float:
        return self.compute_reward(state, relative_propeller_speed, relative_propeller_acceleration)
