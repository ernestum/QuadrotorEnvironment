import numpy as np
import quaternion

from quadrotor_environment.quadrotor_model import QuadrotorModel
from quadrotor_environment.utils import quat2eul


class AttitudePDController:
    """
    PD controller to control a quadrotors attitude.
    """

    def __init__(self, simulation_model: QuadrotorModel, p=1.5, d=0.4, yaw_scale=1):
        """
        Construct a new AttitudePDController to fly the specified quadrotor model with the specified gains.

        :param simulation_model: The quadrotor model used to simulate the quadrotor we want to fly. It is needed to
        transfrom rotational moments to motor speeds.
        :param p: The p gain of the PD controller.
        :param d: The d gain of the PD controller.
        :param yaw_scale: The yaw gains are multiplied by this value.
        """
        self.simulation_model = simulation_model
        self.kp = np.asarray([1, 1, yaw_scale]) * p
        self.kd = np.asarray([1, 1, yaw_scale]) * d

    def __call__(self, state):
        """
        Computes motor speeds based on a given state. Note that this includes the hovering thrust.

        :param state: The state for which to compute the motor speeds.
        :return: The computed motor speeds.
        """
        euler_angles = quat2eul(state.rotation.components)
        angle_action = -euler_angles * self.kp - state.angular_velocity * self.kd
        return self.simulation_model.moments_to_propeller_speeds(angle_action, 1)
