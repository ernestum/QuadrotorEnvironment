import numpy as np
import quaternion

from quadrotor_environment.quadrotor_model import SysState
from quadrotor_environment.utils import inverted_huber_loss, quat2eul, quat2eulzyx, eulNASA2rotmat, rotate_vector


class ToObservationMap:
    """
    The ToObservationMap computes observations from states based on scaling and transforming the state.

    It supports transformation to an ego perspective and different ways to observe the rotation.
    It also supports offsets on all state elements which can be set through the attributes
        * position_offset
        * velocity_offset
        * angular_velocity_offset
    """
    __slots__ = "position_offset", "velocity_offset", "angular_velocity_offset", "observed_position", \
                "observed_velocity", "observed_rotation", "observed_angular_velocity", "observed_propeller_speed", \
                "observed_dt", "observed_observation_age"

    def __init__(self, local_observations: bool = True, huber_scaling: bool = False, position_scale: float = 0.5,
                 velocity_scale: float = 0.5, angular_velocity_scale: float = 0.15, propeller_speed_scale: float = 0,
                 dt_scale: float = 0, observation_age_scale: float = 0, observe_rotation: bool = True,
                 rotation_observation_mode="rotmat_global"):
        """
        Creates a new ToObservationMap with the given settings.

        If any of the scaling factors is zero, the corresponding state component is not included in the final state
        vector.

        :param local_observations: This flag enables if the position and velocity should be transformed to the
        quadrotors local coordinate system.
        :param huber_scaling: This flag enables the scaling by the inverted huber loss function of all state components
        (except for rotation).
        :param position_scale: The factor by which the position should be scaled before observing.
        :param velocity_scale: The factor by which the velocity should be scaled before observing.
        :param angular_velocity_scale: The factor by which the angular velocity should be scaled before observing.
        :param propeller_speed_scale: The factor by which the propeller speed should be scaled before observing.
        :param dt_scale: The factor by which the time difference should be scaled before observing.
        :param observation_age_scale: The factor by which the observation age should be scaled before observing.
        :param observe_rotation: Flag indicating if the rotation should be observed or not.
        :param rotation_observation_mode: The transformation to be applied to the rotation quaternion before observing.
         Can be any of 'rotmat_global', 'rotmat_local', 'rotmat_local_partial', 'euler_global', 'euler_local',
         'euler_local_partial', 'direct'
        """

        self.position_offset = np.zeros(3)
        self.velocity_offset = np.zeros(3)
        self.angular_velocity_offset = np.zeros(3)

        # NOTE: we handle a lot with lambdas here so we can make all the case distinctions once in advance to avoid
        # doing them over and over again during evaluation. This gives us significant speedups.

        # Extract observations from state object, subtract offsets and transform to local coordinate system
        if local_observations:
            observed_position_ = lambda state: \
                rotate_vector(state.rotation, (state.position - self.position_offset) * position_scale)
            observed_velocity_ = lambda state: \
                rotate_vector(state.rotation, (state.velocity - self.velocity_offset) * velocity_scale)
        else:
            observed_position_ = lambda state: (state.position - self.position_offset) * position_scale
            observed_velocity_ = lambda state: (state.velocity - self.angular_velocity_offset) * velocity_scale

        observed_rotation_ = lambda state: state.rotation
        observed_rotation_rate_ = lambda state: \
            (state.angular_velocity - self.angular_velocity_offset) * angular_velocity_scale
        observed_propeller_speed_ = lambda state: state.propeller_speed * propeller_speed_scale
        observed_dt = lambda dt: [dt * dt_scale]
        observed_observation_age = lambda obs_age: [obs_age * observation_age_scale]

        # Apply rotation transformation
        if rotation_observation_mode == 'rotmat_global':
            observed_rotation = lambda state: quaternion.as_rotation_matrix(observed_rotation_(state)).ravel()
        elif rotation_observation_mode == 'rotmat_local':
            observed_rotation = lambda state: eulNASA2rotmat(quat2eulzyx(observed_rotation_(state)))
        elif rotation_observation_mode == 'rotmat_local_partial':
            observed_rotation = lambda state: eulNASA2rotmat(quat2eulzyx(observed_rotation_(state)) * [0, 1, 1])
        elif rotation_observation_mode == 'euler_global':
            observed_rotation = lambda state: quat2eul(observed_rotation_(state))
        elif rotation_observation_mode == 'euler_local':
            observed_rotation = lambda state: quat2eulzyx(observed_rotation_(state))
        elif rotation_observation_mode == 'euler_local_partial':
            observed_rotation = lambda state: quat2eulzyx(observed_rotation_(state))[1:]
        elif rotation_observation_mode == 'direct':
            observed_rotation = lambda state: observed_rotation_(state).components

        # Apply inverted huber scaling
        if huber_scaling:
            observed_position = lambda state: list(map(inverted_huber_loss, observed_position_(state)))
            observed_velocity = lambda state: list(map(inverted_huber_loss, observed_velocity_(state)))
            observed_angular_velocity = lambda state: list(map(inverted_huber_loss, observed_rotation_rate_(state)))
            observed_propeller_speed = lambda state: list(map(inverted_huber_loss, observed_propeller_speed_(state)))
        else:
            observed_position = observed_position_
            observed_velocity = observed_velocity_
            observed_angular_velocity = observed_rotation_rate_
            observed_propeller_speed = observed_propeller_speed_

        # Mask out zero-scaled elements (not observed elements)
        if np.isclose(position_scale, 0):
            observed_position = lambda state: []

        if np.isclose(velocity_scale, 0):
            observed_velocity = lambda state: []

        if not observe_rotation:
            observed_rotation = lambda state: []

        if np.isclose(angular_velocity_scale, 0):
            observed_angular_velocity = lambda state: []

        if np.isclose(propeller_speed_scale, 0):
            observed_propeller_speed = lambda state: []

        if np.isclose(dt_scale, 0):
            observed_dt = lambda state: []

        if np.isclose(observation_age_scale, 0):
            observed_observation_age = lambda state: []

        self.observed_position = observed_position
        self.observed_velocity = observed_velocity
        self.observed_rotation = observed_rotation
        self.observed_angular_velocity = observed_angular_velocity
        self.observed_propeller_speed = observed_propeller_speed
        self.observed_dt = observed_dt
        self.observed_observation_age = observed_observation_age

    def __call__(self, state: SysState, dt: float = None, observation_age: float = None) -> np.ndarray:
        """
        Computes the observation vector based on the observed state, the dt and the age of the observation.

        :param state: The observed state to be transformed.
        :param dt: The duration of the most recent time step.
        :param observation_age: The age of the observation.
        :return: A flat vector with the observed state.
        """
        return np.concatenate([self.observed_position(state),
                               self.observed_velocity(state),
                               self.observed_rotation(state),
                               self.observed_angular_velocity(state),
                               self.observed_propeller_speed(state),
                               self.observed_dt(dt),
                               self.observed_observation_age(observation_age)])
