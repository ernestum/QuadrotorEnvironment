from typing import Union, Sequence, Dict

import numpy as np

from gym import Env

from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
from quadrotor_environment.quadrotor_model import SysState


def deep_map(map_fn, data: Union[dict, list]):
    if isinstance(data, dict):
        mapped_data = dict()
        for key in data.keys():
            mapped, recurse = map_fn(data[key])
            if isinstance(mapped, (dict, list, tuple)) and recurse:
                mapped = deep_map(map_fn, mapped)
            mapped_data[key] = mapped
        return mapped_data
    if isinstance(data, (list, tuple)):
        mapped_data = list()
        for element in data:
            mapped, recurse = map_fn(element)
            if isinstance(mapped, (dict, list)) and recurse:
                mapped = deep_map(map_fn, mapped)
            mapped_data.append(mapped)
        return mapped_data


class SPRWrappedQuadrotorEnvironment(Env):
    """
    This is a wrapper around the QuadrotorEnvironment which can randomize some of its parameters
    (SPR = simulation parameter randomization).

    It takes the same parameters as the QuadrotorEnvironment except that it is allowed to exchange individual float
    numbers with triples of the format `('randomize', low, high)`. In this case that value is drawn uniformly from the
    interval [low, high] on each reset of the environment.
    This can help to train policies that are robust to changes in the simulation environment.
    """

    __slots__ = "configuration_space", "unwrapped_environment", "observation_space", "action_space"

    def __init__(self, **configuration_space):
        """
        Constructs a new simulation parameter randomizer for the quadrotor environment.

        :param configuration_space: The configuration that is to be passed to the quadrotor environment, possibly with
        some `('randomize', low, high)` triples in it (see class documentation).
        """
        self.configuration_space = configuration_space
        self.unwrapped_environment = None
        self.reset()
        self.observation_space = self.unwrapped_environment.observation_space
        self.action_space = self.unwrapped_environment.action_space

    def step(self, action):
        """
        Steps the wrapped environment with the action.

        :param action: The action to apply to the wrapped environment.
        :return: The return value of the wrapped environment.
        """
        return self.unwrapped_environment.step(action)

    def reset(self, initial_state: Union[SysState, None] = None,
              initial_propeller_speed: Union[np.ndarray, None] = None, spr_seed: int = None):
        """
        Randomizes the quadrotor configuration, creates a new QuadrotorEnvironment with the randomized configuration
        and resets it.
        :return: The return value of the environment reset.
        """
        random = np.random.RandomState(spr_seed)
        def sample_map_fn(d):
            if isinstance(d, (list, tuple)) and len(d) == 3 and d[0] == 'randomize':
                low = np.asarray(d[1])
                high = np.asarray(d[2])
                return random.rand() * (high - low) + low, False
            else:
                return d, True

        new_config = deep_map(sample_map_fn, self.configuration_space)
        self.unwrapped_environment = QuadrotorEnvironment(**new_config)
        return self.unwrapped_environment.reset(initial_state, initial_propeller_speed)

    def render(self, **kwargs):
        """
        Forwards to the wrapped environment.
        :param kwargs: kwargs to forward.
        :return: Return value of the wrapped environments render method.
        """
        return self.unwrapped_environment.render(**kwargs)

    def get_current_state(self):
        """
        Returns the current state of the wrapped environment.

        :return: The current state of the wrapped environment.
        """
        return self.unwrapped_environment.get_current_state()

    @property
    def time(self):
        """
        :return: The current time of the wrapped environment.
        """
        return self.unwrapped_environment.time

    @property
    def max_time(self):
        """
        :return: The current maximum time of the wrapped environment.
        """
        return self.unwrapped_environment.max_time

    @property
    def unwrapped(self):
        """
        :return: The wrapped environment.
        """
        return self.unwrapped_environment




