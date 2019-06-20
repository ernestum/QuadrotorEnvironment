import numpy as np


class PropellerSpeedMapping:
    """
    The PropellerSpeedMapping maps network outputs from the range [-1, 1] to propeller speeds that can be sent to the
    quadrotor.

    It constrains the absolute speed as well as the acceleration.
    """
    __slots__ = "previous_relative_speed", "speed_fn", "hovering_speed", "relative_max_propeller_acceleration"

    def __init__(self, hovering_speed: float, relative_min_speed: float = 0, relative_max_speed: float = 3,
                 relative_max_propeller_acceleration: float = np.infty,
                 acceleration_constrain_mode: str = 'clip', output_control_mode: str = 'direct'):
        """
        Constructs a new PropellerSpeedMapping with the given parameters.

        :param hovering_speed: The absolute speed per propeller, that is needed to make the quadrotor hover.
        :param relative_min_speed: The minimum speed relative to the hovering speed. The resulting absolute minimum
        speed will be `hovering_speed * relative_min_speed`.
        :param relative_max_speed: The maximum speed relative to the hovering speed. The resulting absolute maximum
        speed will be `hovering_speed * relative_max_speed`.
        :param relative_max_propeller_acceleration: The maximum relative acceleration of the propellers.
        :param acceleration_constrain_mode: Specifies in what way the accelerations are constrained.
        Can be either 'tanh' oder 'clip'.
        :param output_control_mode: Chooses how the propeller speed is controlled. Either by mapping the actions to
        propeller speeds or by mapping the actions to propeller accelerations. This can be either 'direct' or
        'acceleration'.
        """
        assert relative_min_speed < relative_max_speed
        self.previous_relative_speed = None
        self.hovering_speed = hovering_speed
        self.relative_max_propeller_acceleration = relative_max_propeller_acceleration

        def clip_relative_speed(s):
            return np.clip(s, relative_min_speed, relative_max_speed)

        if acceleration_constrain_mode == 'clip':
            def clip_relative_acceleration(acceleration):
                return np.clip(acceleration, -relative_max_propeller_acceleration, relative_max_propeller_acceleration)
        elif acceleration_constrain_mode == 'tanh':
            def clip_relative_acceleration(acceleration):
                return np.tanh(acceleration / relative_max_propeller_acceleration) * relative_max_propeller_acceleration
        else:
            raise ValueError("acceleration_constrain_mode must be either 'clip' or 'tanh'. Not '{}'".format(
                acceleration_constrain_mode))

        if output_control_mode == 'direct':
            def action_to_speed(a, dt):
                relative_target_speed = (a + 1) / 2 * (relative_max_speed - relative_min_speed) + relative_min_speed
                relative_target_acceleration = (relative_target_speed - self.previous_relative_speed) / dt

                acceleration_clipped_relative_target_speed = self.previous_relative_speed + clip_relative_acceleration(relative_target_acceleration) * dt
                actual_relative_acceleration = (acceleration_clipped_relative_target_speed - self.previous_relative_speed) / dt
                self.previous_relative_speed = acceleration_clipped_relative_target_speed
                return self.previous_relative_speed * hovering_speed, self.previous_relative_speed, actual_relative_acceleration

            self.speed_fn = action_to_speed
        elif output_control_mode == 'thrust':
            if relative_min_speed < 0:
                raise ValueError("negative thrust is not possible! Relative min speed must be above 0 when using thrust control")
            hovering_thrust = hovering_speed ** 2
            def action_to_speed(a, dt):
                relative_target_thrust = (a + 1) / 2 * (relative_max_speed - relative_min_speed) + relative_min_speed
                relative_target_acceleration = (relative_target_thrust - self.previous_relative_speed) / dt

                acceleration_clipped_relative_target_thrust = self.previous_relative_speed + clip_relative_acceleration(relative_target_acceleration) * dt
                actual_relative_acceleration = (acceleration_clipped_relative_target_thrust - self.previous_relative_speed) / dt
                self.previous_relative_speed = acceleration_clipped_relative_target_thrust
                return np.sqrt(self.previous_relative_speed * hovering_thrust), self.previous_relative_speed, actual_relative_acceleration

            self.speed_fn = action_to_speed

        elif output_control_mode == 'acceleration':
            if np.isinf(relative_max_propeller_acceleration):
                raise ValueError("The relative maximum acceleration must be finite when the output_control_mode is set "
                                 "to 'acceleration'!")

            def action_to_speed(a, dt):
                relative_target_acceleration = a * relative_max_propeller_acceleration

                relative_target_speed = self.previous_relative_speed + relative_target_acceleration * dt
                new_relative_speed = clip_relative_speed(relative_target_speed)
                relative_acceleration = (new_relative_speed - self.previous_relative_speed) / dt
                self.previous_relative_speed = new_relative_speed
                return self.previous_relative_speed * hovering_speed, self.previous_relative_speed, relative_acceleration

            self.speed_fn = action_to_speed
        else:
            raise ValueError("output_control_mode must be either 'direct' or 'acceleration'. Not '{}'".format(
                output_control_mode))

    def __call__(self, action: np.ndarray, dt: float):
        """
        Computes propeller speeds based on neural network 'action' outputs.

        :param action: The output of the neural network, assumed to be in [-1, 1]
        :param dt: The time that has passed since the last propeller speed was set.
        :return: A triple consisting of the propeller speed to be sent to the quadrotor, the propeller speed relative to
         the hovering speed and the relative propeller acceleration.
        """
        return self.speed_fn(action, dt)

    def reset(self, initial_thrust: np.ndarray):
        """
        Resets the PropellerSpeedMapping to the given initial thrust.
        :param initial_thrust: The initial thrust to assume.
        :return: The assumed relative previous speed.
        """
        # TODO: add warning here if initial thrust is infeasible
        self.previous_relative_speed = initial_thrust / self.hovering_speed
        return self.previous_relative_speed
