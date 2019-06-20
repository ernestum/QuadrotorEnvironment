import time

import numpy as np
import quaternion
from hyperopt import fmin, tpe, Trials
import matplotlib.pyplot as plt

from hyperopt.hp import uniform
from hyperopt.plotting import main_plot_vars, main_plot_history

from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
from quadrotor_environment.quadrotor_model import SysState


def eval_params(params: dict):
    render = params.pop("render", False)
    num_epochs = params.pop("num_epochs", 30)
    env = QuadrotorEnvironment(pd_support_parameters=params,
                               enable_rendering=render,
                               delay_parameters=dict(action_delay=0.0),
                               reward_function_parameters=dict(
                                   reward_scale=1, position=0, velocity=0, angular_velocity=0,
                                   position_h=0, position_v=0, velocity_h=0, velocity_v=0, rotation_h=1,
                                   rotation_v=1, angular_velocity_h=.1, angular_velocity_v=0,
                                   propeller_speed_deviation=0, propeller_acceleration=0,
                                   inverted_huber_reward_scaling=True),
                               max_time=3 if render else 1)

    def mk_init_state():
        return SysState(np.zeros(3),
                        np.zeros(3),
                        np.quaternion(*([1, 0, 0, 0] + np.random.uniform(-0.1, 1, 4))).normalized(),
                        np.zeros(3) + [0, 0, 0.], np.ones(4) * env.simulation_model.hovering_thrust)

    # env.reset(initial_state=mk_init_state())

    reward_sum = 0
    for _ in range(num_epochs):
        done = False
        env.reset()
        while not done:
            _, reward, done, _ = env.step(np.zeros(4) - 1)
            reward_sum += reward
            if render:
                time.sleep(0.01)

    print(reward_sum / num_epochs)
    return -reward_sum / num_epochs


if __name__ == '__main__':
    while True:
        eval_params(dict(d=.6,
                         p=2,
                         yaw_scale=0.2,
                         render=True))

    space = dict(p=uniform("p", 0, 10), d=uniform("d", 0, 2), yaw_scale=uniform("yaw_scale", 0, 2))
    trials = Trials()
    best = fmin(eval_params, space, tpe.suggest, 50)
    print(best)
    while True:
        best['render'] = True
        eval_params(best)
