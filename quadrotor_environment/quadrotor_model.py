from typing import Sequence, Union

import numpy as np
import quaternion

from quadrotor_environment.utils import rotate_vector


class SysState:
    """
    Holds the system state of the QuadrotorModel.
    """
    __slots__ = "position", "velocity", "rotation", "angular_velocity", "propeller_speed"

    def __init__(self, position: np.ndarray, velocity: np.ndarray, rotation: np.quaternion, rotation_rate: np.ndarray,
                 prop_rate: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.rotation = rotation
        self.angular_velocity = rotation_rate
        self.propeller_speed = prop_rate

    def isnan(self):
        """
        Checks if any of the state components is nan.
        :return: True if any of the state components is nan. False otherwise.
        """
        return np.any(np.concatenate([np.isnan(self.position), np.isnan(self.velocity), [np.isnan(self.rotation)],
                                      np.isnan(self.angular_velocity), np.isnan(self.propeller_speed)]))


class QuadrotorModel:
    """
    The QuadrotorModel simulates a flying quadrotor.

    Note that this is an explicit model that does not rely on euler integration or RK-4 integration. Therefore rather
    large steps can be taken without any inaccuracies.
    """
    __slots__ = 'I', 'ip', 'lp', 'b', 'kp', 'rhoa', 'qs', 'M', 'M_inv', 'mass', 'gravity', 'hovering_thrust', 'params'

    def __init__(self, g0: float = 9.80665,
                       m: float = 1.43805,
                       I: Sequence[float] = (0.01467, 0.01441, 0.02441),
                       lp: float = 0.16,
                       kp: Union[float, Sequence[float]] = 1.018841e-5,
                       ip: float = 6.32180911740998e-09,
                       b: float = 4.82e-8,
                 ):
        """
        Constructs a new QuadrotorModel with the given simulation parameters.

        :param g0: Gravity
        :param m: Mass of quadrotor (kg)
        :param I: Eigenvalues of tensor of moments of inertia
        :param lp: Half side length of the square described by the propellers (m)
        :param kp: Thrust coefficient of propellers.
        :param ip: Inertia coefficient of propellers.
        :param b: Drag coefficient of propellers.
        """
        kp = np.asarray(kp)

        self.gravity = g0
        self.mass = m
        self.I = np.asarray(I)
        self.M = np.array([lp * kp * np.array([-1, -1, +1, +1]),
                           lp * kp * np.array([+1, -1, -1, +1]),
                            b *      np.array([-1, +1, -1, +1]),
                           kp *      np.array([-1, -1, -1, -1])])
        self.M_inv = np.linalg.inv(self.M)
        self.ip = ip
        self.lp = lp
        self.kp = kp
        self.b = b
        self.hovering_thrust = np.sqrt((m * g0 / kp) / 4)

    def next_state(self, x: SysState, u: Sequence[float], h: float):
        """
        Compute the future state x' when applying the controls u in state x for the duration of h seconds.

        :param x: The initial state.
        :param u: The controls to apply.
        :param h: The duration (s) for which to apply the controls.
        :return: The next state x'.
        """
        # Compute the next propeller rate
        next_prop_rate = x.propeller_speed - (self.ip ** h - 1) * (u - x.propeller_speed)

        # Compute moments and forces from motor speeds in body frame
        TF = np.dot(self.M, next_prop_rate**2)

        # Compute acceleration in inertial
        a = rotate_vector(x.rotation.conjugate(), np.array([0, 0, TF[3] / self.mass])) + np.array([0, 0, self.gravity])

        # Compute next position in inertial
        next_pos = x.position + x.velocity * h + a * h * h / 2

        # Compute next velocity in inertial
        next_vel = x.velocity + a * h

        # Compute next rotation in body frame
        omhalfh = (x.angular_velocity * h + 1.0 / self.I * TF[0:3] * h * h / 2.0) / 2.0
        next_rotation = x.rotation * np.quaternion(0, omhalfh[0], omhalfh[1], omhalfh[2]).exp()
        next_rotation = next_rotation.normalized()

        # Compute next rotation rate in body frame
        next_rotation_rate = x.angular_velocity + 1.0 / self.I * TF[0:3] * h

        x = SysState(next_pos, next_vel, next_rotation, next_rotation_rate, next_prop_rate)
        return x

    def moments_to_propeller_speeds(self, angular_moments, relative_thrust):
        """
        Computes motor speeds based on the given angular moments and the relative thrust.

        :param angular_moments: The desired angular moments to apply to the quadrotor.
        :param relative_thrust: The desired relative thrust to apply. The thrust is relative to the hovering thrust,
        0 being no thrust and 1 hovering thrust.
        :return:
        """
        t = -self.gravity * self.mass * relative_thrust
        squared_speeds = np.dot(self.M_inv, np.concatenate([angular_moments, [t]]))
        squared_speeds -= np.min([0, np.min(squared_speeds)])
        return np.sqrt(squared_speeds)
