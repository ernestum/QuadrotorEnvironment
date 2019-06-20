from unittest import TestCase

import numpy as np

from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
from quadrotor_environment.quadrotor_model import SysState


class TestQuadrotorEnvironment(TestCase):

    def setUp(self):
        self.offsets = [
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
            [10., 10., 10.],
        ]

        self.dts = [0.0025, 0.05, 0.1]

        self.hover_actions = np.ones((300, 5)) * 0.5

        self.hover_then_flip_actions = np.array(self.hover_actions)
        self.hover_then_flip_actions[150:, (1, 2)] = 0.6
        self.hover_then_flip_actions[150:, (3, 4)] = 0.4

        self.random_uniform_actions = np.random.rand(300, 5)
        self.random_normal_actions = np.clip(np.random.randn(300, 5)/2 + 0.5, 0, 1)

        self.action_sequences = [
            self.hover_actions,
            self.hover_then_flip_actions,
            self.random_uniform_actions,
            self.random_normal_actions
        ]

    # def test_position_independence(self):
    #     """
    #     Tests if two environments that are equals except for an offset on the initial position behave the same.
    #     """
    #     # self.skipTest("too expensive now")
    #     reward_handler = HoveringReward()
    #
    #     def dotest(dt, offset, sqrt_scaling, actions):
    #         # Construct environments
    #         env1 = QuadrotorEnvironment(reward_handler, initial_pos_range=0., initial_vel_range=0.,
    #                                     initial_rot_range=0., initial_rotrate_range=0., quadrotor_mass_range=1.,
    #                                     quadrotor_size_range=1., gravity_range=9.81, dt_range=dt, pos_bounds=None,
    #                                     sqrt_scaling_on_controls=sqrt_scaling)
    #         env2 = QuadrotorEnvironment(reward_handler, initial_pos_range=0., initial_vel_range=0.,
    #                                     initial_rot_range=0., initial_rotrate_range=0., quadrotor_mass_range=1.,
    #                                     quadrotor_size_range=1., gravity_range=9.81, dt_range=dt, pos_bounds=None,
    #                                     sqrt_scaling_on_controls=sqrt_scaling)
    #
    #         observation1, done1 = env1.reset(), False
    #         observation2, done2 = env2.reset(), False
    #
    #         self.assertAlmostEquals(env1.get_hovering_thrust(), env2.get_hovering_thrust())
    #         hovering_thrust = env1.get_hovering_thrust()
    #         control_mode = dict(
    #             type='direct',  # can also be 'joystick'
    #             hovering_thrust=hovering_thrust,
    #             max_thrust_control=hovering_thrust * 0.05,
    #             max_steering=hovering_thrust * 0.05,
    #             thrust_bias=0.1,
    #             steering_bias=.1
    #         )
    #         env1.control_mode = env2.control_mode = control_mode
    #
    #         env2.simulation_model.x.pos += offset
    #
    #         observation_offset = env2.observation_from_state(env2.simulation_model.x)[:3] - env1.observation_from_state(env1.simulation_model.x)[:3]
    #
    #         observations = []
    #         for action in actions:
    #             observation1, done1, reward1 = env1.execute(action)
    #             observation2, done2, reward2 = env2.execute(action)
    #             observation2[:3] += observation_offset
    #
    #             self.assertTrue(all(np.isclose(env1.simulation_model.x.pos, env2.simulation_model.x.pos - offset)))
    #             self.assertTrue(all(np.isclose(observation1[3:], observation2[3:])))
    #             self.assertEqual(done1, done2)
    #
    #             observations.append(observation1)
    #
    #     for dt in self.dts:
    #         for offset in self.offsets:
    #             for sqrt_scaling in [True, False]:
    #                 for actions in self.action_sequences:
    #                     dotest(dt, offset, sqrt_scaling, actions)

    def test_hovering_thrust(self):
        """
        Tests if the hovering thrust actually makes the quadrotor hover
        """
        initial_state = SysState(np.zeros(3), np.zeros(3), np.quaternion(1, 0, 0, 0), np.zeros(3), None)
        env = QuadrotorEnvironment()
        initial_observation = env.reset(initial_state)
        for _ in range(1000):
            observation, done, reward, _ = env.step(np.ones(4) * 2/3 - 1)
            self.assertTrue(all(np.isclose(observation, initial_observation)))

