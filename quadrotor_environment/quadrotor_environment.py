from typing import Dict, Union

import numpy as np
from gym import Env
from gym.spaces import Box

# from quadrotor_rl_code.quadrotor_environment.attitude_pd_controller import AttitudePDController
from quadrotor_environment.attitude_pd_controller import AttitudePDController
from quadrotor_environment.model_delay_wrapper import ModelDelayWrapper
from quadrotor_environment.model_state_clipping_wrapper import StateClippingWrapper
from quadrotor_environment.noise_model import NoisedStateMap
from quadrotor_environment.observation_history import ObservationHistory
from quadrotor_environment.observation_mapping import ToObservationMap
from quadrotor_environment.propeller_speed_mapping import PropellerSpeedMapping
from quadrotor_environment.reward_function import RewardFunction
from .quadrotor_model import SysState, QuadrotorModel
from .utils import sample_random_rotation

try:
    from zerocm import ZCM
    from zcom import timestamped_vector_double
except Exception:
    pass


class QuadrotorEnvironment(Env):
    """
    The QuadrotorEnvironment is a Gym environment that simulates a quadrotor. The simulation has support for
        * action and observation delays with jitter
        * observation noise, and
        * state clipping.
    The environment is highly customizable including
        * the (randomized) initial state of the quadrotor
        * the state to observation transformation, which can include a history of past observations
        * the action to propeller speed mapping including direct and differential drive, absolute speed clipping and
          propeller acceleration clipping.
        * the reward function
    Its design is modular. E.g. you can use the action to propeller speed mapping or the state to observation
    transformation when running the learned policies on real hardware without taking the environment apart.
    """

    __slots__ = "max_time", "enable_rendering", "simulation_model", "delayed_model", "noised_state", \
                "state_to_obersvation", "observation_history", "reward_base", "action_to_propeller_speed", "attitude_pd", \
                "initial_state_parameters", "zcm", "reward_fn", "action_space", "observation_space"

    def __str__(self):
        return "Quadrotor Environment"

    def __init__(self, reward_function_parameters: Dict = dict(), quadrotor_model_parameters: Dict = dict(),
                 delay_parameters: Dict = dict(), observation_mapping_parameters: Dict = dict(),
                 noise_parameters: Dict = dict(), observation_history_parameters: Dict = dict(),
                 state_clipping_parameters: Dict = dict(),
                 observe_prop_rate: bool = False,
                 reward_base: str = 'current_state',
                 propeller_speed_mapping_parameters: Dict = dict(),
                 pd_support_parameters: Union[None, Dict] = None,
                 initial_state_parameters: Dict = dict(box_side_length=2, velocity=1, angular_velocity=1),
                 max_time: float = 10, controller_period: float = 0.01, enable_rendering: bool = False,
                 quadrotor_model_parameters_for_preprocessing=None):
        """
        Constructs a new QuadrotorEnvironment with the given parameters.

        :param reward_function_parameters: The parameters to be passed to the reward function object.
        See the documentation for the RewardFunction class for details.
        :param delay_parameters: The parameters to be passed to the model delay wrapper. Defaults to no delays or
        jitters. See the documentation for the ModelDelayWrapper class for details.
        :param observation_mapping_parameters: The parameters to be passed to the state to observation mapping object.
        Defaults to observe position, rotation, velocity and angular velocity with some balanced scaling factors. The
        rotation is observed as a (flattened) rotation matrix by default. See the documentation for the ToObservationMap
         class for details.
        :param noise_parameters: The parameters to be passed to the state noiser object. It applies noise to the system
        state before it is observed. By default there is no noise. See the documentation for the NoisedStateMap for
        details.
        :param observation_history_parameters: Parameters to be passed to the ObservationHistory class. It provides a 
        view on past state observations and actions. By default just the current state is observed and no past actions.
        See the ObservationHistory documentation for details.
        :param state_clipping_parameters: Parameters to be passed to the StateClippingWrapper class. During runtime it
        constraints the state within defined bounds. See the StateClippingWrapper documentation for details.
        :param observe_prop_rate: Flag that enables observing the current propeller rate TODO: this one should go away
        :param reward_base: The reward is calculated based on a system state. What system state (observed state,
        actual state, state with or without the observed noise etc.) is used to compute the reward is determined by this
        parameter. Can beone of 'current_state', 'next_state', 'observed_state', 'observed_state_with_noise'.
        :param propeller_speed_mapping_parameters: The parameters passed to the PropellerSpeedMapping class.
        By default, the propeller speed is controlled directly, clipped to up to three times the hovering speed without
        any constraints on the propeller acceleration.
        :param pd_support_parameters: Gains for an attitude proportinal, differential (PD) controller whose output is
        added to all actions. If None, there is no PD controller. See the AttitudePDController for details on the
        parameters.
        :param initial_state_parameters: Controls the distribution from which the initial state is drawn upon reset.
        Must be a dict with the members
            * box_side_length: for the size of the box around the origin in which the quadrotor will be placed
            * velocity: for the maximum initial velocity in x, y and z direction.
            * angular_velocity: for the maximum angular velocity around all three rotation axes.
        :param max_time: The maximum time after which an episode in this environment is done.
        :param controller_period: The duration (in seconds simulated time) between two calls to the policy.
        :param enable_rendering: A flag controlling if the environment should be rendered or not.
        :param quadrotor_model_parameters_for_preprocessing: Some of the preprocessing steps such as the observation
        computation and the PropellerSpeedMapping use some parameters of the simulation such as the hovering thrust.
        If we want to base those preprocessing steps on different simulation parameters than the actual simulation, this
        parameter can be set. If None the original simulation parameters are being used.
        """
        assert reward_base in ['observed_state', 'observed_state_with_noise', 'current_state', 'next_state']
        if quadrotor_model_parameters_for_preprocessing is None:
            quadrotor_model_parameters_for_preprocessing = quadrotor_model_parameters
        quadrotor_model_for_preprocessing = QuadrotorModel(**quadrotor_model_parameters_for_preprocessing)
        self.max_time = max_time
        self.enable_rendering = enable_rendering

        self.simulation_model = QuadrotorModel(**quadrotor_model_parameters)
        self.delayed_model = ModelDelayWrapper(StateClippingWrapper(self.simulation_model, **state_clipping_parameters),
                                               controller_period=controller_period, **delay_parameters)
        self.noised_state = NoisedStateMap(**noise_parameters)
        if observe_prop_rate:
            self.state_to_observation = ToObservationMap(
                propeller_speed_scale=0.1 / quadrotor_model_for_preprocessing.hovering_thrust,
                **observation_mapping_parameters)
        else:
            self.state_to_observation = ToObservationMap(**observation_mapping_parameters)
        self.observation_history = ObservationHistory(**observation_history_parameters)
        self.reward_base = reward_base
        self.action_to_propeller_speed = PropellerSpeedMapping(quadrotor_model_for_preprocessing.hovering_thrust, **propeller_speed_mapping_parameters)
        if pd_support_parameters is not None:
            self.attitude_pd = AttitudePDController(quadrotor_model_for_preprocessing, **pd_support_parameters)
        else:
            self.attitude_pd = None

        self.initial_state_parameters = initial_state_parameters
        self.zcm = None

        self.reward_fn = RewardFunction(**reward_function_parameters)

        self.action_space = Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = Box(-np.infty, np.infty, self.reset().shape, dtype=np.float32)

    def step(self, action):
        """
        Computes the next observation, reward, done flag and info as the Environment interface of Gym requires.

        :param action: The action to be transformed to propeller speeds.
        :return: Tuple consisting of observation, reward, done, {}
        """
        # Compute rotor speeds
        propeller_speeds, relative_propeller_speed, relative_propeller_acceleration = self.action_to_propeller_speed(
            action, self.observation_history.dt_history[-1])  # This is a hacky way to access the most recent dt

        if self.attitude_pd is not None:
            current_state = self.delayed_model.compute_current_state()
            propeller_speeds += self.attitude_pd(current_state)
            propeller_speeds = np.clip(propeller_speeds, 0, self.action_to_propeller_speed.hovering_speed*3)

        if self.reward_base == 'current_state':
            reward = self.reward_fn(self.delayed_model.compute_current_state(), relative_propeller_speed,
                                    relative_propeller_acceleration)

        # Compute next state
        observed_state, dt, observation_age = self.delayed_model.step(propeller_speeds)
        if self.reward_base == 'observed_state':
            reward = self.reward_fn(observed_state, relative_propeller_speed, relative_propeller_acceleration)

        # Compute Observation
        observed_state_with_noise = self.noised_state(observed_state)

        if self.reward_base == 'observed_state_with_noise':
            reward = self.reward_fn(observed_state_with_noise, relative_propeller_speed,
                                    relative_propeller_acceleration)

        observation = self.observation_history.get_observation_with_history(
            self.state_to_observation(observed_state_with_noise, dt, observation_age),
            relative_propeller_speed, dt)

        if self.reward_base == 'next_state':
            reward = self.reward_fn(self.delayed_model.compute_current_state(), relative_propeller_speed,
                                    relative_propeller_acceleration)

        if self.enable_rendering:
            self.render(reward, self.delayed_model.compute_current_state(), observed_state_with_noise, propeller_speeds)

        return observation, reward, self.delayed_model.time >= self.max_time, {}

    def random_state(self):
        """
        Computes a random state based on the initial state parameters.
        The initial propeller speed is always the hovering speed.

        :return: A new system state.
        """
        radius = self.initial_state_parameters['box_side_length']
        velocity = self.initial_state_parameters['velocity']
        angular_velocity = self.initial_state_parameters['angular_velocity']
        return SysState(np.random.uniform(-radius, radius, 3),
                        np.random.uniform(-velocity, velocity, 3),
                        sample_random_rotation(),
                        np.random.uniform(-angular_velocity, angular_velocity, 3),
                        np.ones(4) * self.simulation_model.hovering_thrust)

    def reset(self, initial_state: Union[SysState, None] = None,
              initial_propeller_speed: Union[np.ndarray, None] = None):
        """
        Resets the environment to either a random initial state and hovering speed for the propellers or the provided
        initial state and propeller speed.

        :param initial_state: The initial state. If None (default) a random state is generated.
        :param initial_propeller_speed:  The initial propeller speed. If None (default) the hovering speed is used.
        :return: The initial observation.
        """
        if initial_propeller_speed is None:
            initial_propeller_speed = np.ones(4) * self.simulation_model.hovering_thrust
        if initial_state is None:
            initial_state = self.random_state()
        if initial_state.propeller_speed is None:
            initial_state.propeller_speed = initial_propeller_speed
        observed_state, initial_dt, initial_obs_age = self.delayed_model.reset(initial_propeller_speed, initial_state)

        initial_observation = self.state_to_observation(observed_state, initial_dt, initial_obs_age)

        self.observation_history.reset(initial_observation,
                                       initial_propeller_speed * 0.1 / self.simulation_model.hovering_thrust,
                                       initial_dt)
        initial_relative_propeller_speed = self.action_to_propeller_speed.reset(initial_propeller_speed)

        if self.enable_rendering:
            self.render(0, initial_state, initial_state, initial_propeller_speed)

        return self.observation_history.get_observation_with_history(initial_observation,
                                                                     initial_relative_propeller_speed, initial_dt)

    def on_target_msg(self, channel, msg):
        # print("new target:", msg.values)
        assert len(msg.values) == 3
        self.state_to_observation.position_offset = msg.values

    def render(self, reward, state, observed_state, thrust):

        if self.zcm is None:
            self.zcm = ZCM("")
            if self.zcm.good():
                self.zcm.subscribe("target", timestamped_vector_double, self.on_target_msg)
                self.zcm.start()

        if self.zcm.good():
            msg = timestamped_vector_double()
            msg.ts = int(self.delayed_model.time * 1e9)

            def publish(channel, data):
                msg.values = data
                msg.len = len(data)
                self.zcm.publish(channel, msg)

            publish("quadrotor_viz/quat_pose", np.concatenate([state.position, state.rotation.components]))
            publish("quadrotor_viz/quat_pose_noised", np.concatenate([observed_state.position, observed_state.rotation.components]))
            publish("quadrotor_viz/velocity", state.velocity)
            publish("quadrotor_viz/rotation_rate", state.angular_velocity)
            publish("quadrotor_viz/rotatorspeed", state.propeller_speed)
            publish("quadrotor_viz/control", thrust)
            publish("quadrotor_viz/reward", [reward])

    @property
    def time(self):
        return self.delayed_model.time

    def get_current_state(self) -> SysState:
        return self.delayed_model.compute_current_state()

