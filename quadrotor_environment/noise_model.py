from typing import Union, Dict, Sequence

import numpy as np
import quaternion

from quadrotor_environment.quadrotor_model import SysState


def _parse_noise_spec(spec):
    if isinstance(spec, dict):
        return spec['mean'], spec['std']
    if np.asarray(spec).ndim == 0:
        return 0, spec
    else:
        return spec


class NoisedStateMap:
    """
    A mapping function that maps a state to a state with gaussian noise added.
    """
    __slots__ = "noised_position", "noised_velolicty", "noised_angular_velocity", "noised_propeller_speed", "noised_rotation"

    def __init__(self, noise_scale=1,
                 position_noise: Union[None, Dict, Sequence] = None,
                 velocity_noise: Union[None, Dict, Sequence] = None,
                 rotation_noise: Union[None, Dict, Sequence] = None,
                 angular_velocity_noise: Union[None, Dict, Sequence] = None,
                 propeller_speed_noise: Union[None, Dict, Sequence] = None):
        """
        Constructs a new noise state map with the given parameters.

        Note:
            The gaussian noises can be specified in a number of ways
            1. `std`
            2. `(mean, std)`
            3. `dict(mean=your_mean, std=your_std)`
            The mean and standard deviation can be either scalar or match the dimension as the state part they are
            applied to.

        :param noise_scale: Global scale multiplied to all noises. Default is 1.
        :param position_noise: The noise to apply to the position.
        :param velocity_noise: The noise to apply to the velocity.
        :param rotation_noise: The noise to apply to the rotation.
        :param angular_velocity_noise: The noise to apply to the angular velocity.
        :param propeller_speed_noise: The noise to apply to the propeller speed.
        """
        if position_noise is not None:
            position_mean, position_std = _parse_noise_spec(position_noise)
            self.noised_position = lambda state: state.position + \
                                                 np.random.randn(3) * position_std * noise_scale + position_mean
        else:
            self.noised_position = lambda state: state.position

        if velocity_noise is not None:
            velocity_mean, velocity_std = _parse_noise_spec(velocity_noise)
            self.noised_velolicty = lambda state: state.velocity + \
                                                  np.random.randn(3) * velocity_std * noise_scale + velocity_mean
        else:
            self.noised_velolicty = lambda state: state.velocity

        if angular_velocity_noise is not None:
            angular_velocity_mean, angular_velocity_std = _parse_noise_spec(angular_velocity_noise)
            self.noised_angular_velocity = lambda state: state.angular_velocity + \
                                                         np.random.randn(3) * angular_velocity_std * noise_scale + \
                                                         angular_velocity_mean
        else:
            self.noised_angular_velocity = lambda state: state.angular_velocity

        if propeller_speed_noise is not None:
            propeller_speed_mean, propeller_speed_std = _parse_noise_spec(propeller_speed_noise)
            self.noised_propeller_speed = lambda state: state.propeller_speed + \
                                                        np.random.randn(4) * propeller_speed_std * noise_scale + \
                                                        propeller_speed_mean
        else:
            self.noised_propeller_speed = lambda state: state.propeller_speed

        if rotation_noise is not None:
            rotation_mean, rotation_std = _parse_noise_spec(rotation_noise)
            self.noised_rotation = lambda state: np.quaternion(*(state.rotation.components +
                                                                 np.random.randn(4) * rotation_std * noise_scale +
                                                                 rotation_mean))
        else:
            self.noised_rotation = lambda state: state.rotation

    def __call__(self, state: SysState):
        """
        Apply the specified noise to the state.

        :param state: The 'clean' state without any noise.
        :return: A copy of the state with noise added.
        """
        nosied_state = SysState(self.noised_position(state),
                                self.noised_velolicty(state),
                                self.noised_rotation(state),
                                self.noised_angular_velocity(state),
                                self.noised_propeller_speed(state))
        return nosied_state
