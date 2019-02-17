import numpy as np

from gym import Env

from quadrotor_rl_code.quadrotor_environment.quadrotor_environment import QuadrotorEnvironment


def sample_map_fn(d):
    if isinstance(d, (list, tuple)) and len(d) == 3 and d[0] == 'randomize':
        low = np.asarray(d[1])
        high = np.asarray(d[2])
        return np.random.rand() * (high - low) + low, False
    else:
        return d, True


def mean_map_fn(d):
    if isinstance(d, (list, tuple))  and len(d) == 3 and d[0] == 'randomize':
        low = np.asarray(d[1])
        high = np.asarray(d[2])
        return (high - low)/2, False
    else:
        return d, True


def deep_map(map_fn, data: dict):
    mapped_data = dict()
    for key in data.keys():
        mapped, recurse = map_fn(data[key])
        if isinstance(mapped, dict) and recurse:
            mapped = deep_map(map_fn, mapped)
        mapped_data[key] = mapped
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

    def reset(self):
        """
        Randomizes the quadrotor configuration, creates a new QuadrotorEnvironment with the randomized configuration
        and resets it.
        :return: The return value of the environment reset.
        """
        new_config = deep_map(sample_map_fn, self.configuration_space)
        self.unwrapped_environment = QuadrotorEnvironment(**new_config)
        return self.unwrapped_environment.reset()

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
        return self.unwrapped_environment.compute_current_state()

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




