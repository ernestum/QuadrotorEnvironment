import matplotlib.pyplot as plt

from quadrotor_environment.model_delay_wrapper import ModelDelayWrapper
from quadrotor_environment.tests.test_delayed_model import LinearTestModel


def make_trajectory(delayer):
    obs0, _, _ = delayer.reset(0, 0)
    assert obs0 == 0

    actions = [1, 2, 0, -1, -2, 0, 0, 0]
    observations = []
    times = []
    for action in actions:
        times.append(delayer.time)
        observations.append(delayer.step(action))
    return times, observations


def get_title(obs_delay, action_delay):
    if not obs_delay and not action_delay:
        return "No Delays"
    elif obs_delay and action_delay:
        return "Observation Delay and Action Delay"
    elif obs_delay:
        return "Observation Delay"
    elif action_delay:
        return "Action Delay"
    else:
        assert False


def get_color(obs_jitter, action_jitter):
    if not obs_jitter and not action_jitter:
        return "black"
    elif obs_jitter and action_jitter:
        return "purple"
    elif obs_jitter:
        return "red"
    elif action_jitter:
        return "blue"
    else:
        assert False


def main():
    model = LinearTestModel()
    control_jitter = 0
    for obs_delay in [0, 1]:
        for action_delay in [0, 1]:
            # figure = plt.figure()
            plt.title(get_title(obs_delay, action_delay))
            plt.xlabel("Time")
            plt.ylabel("State")
            for action_jitter in [0, 0.2]:
                for observation_jitter in [0, 0.2]:
                    color = get_color(observation_jitter, action_jitter)
                    delayer = ModelDelayWrapper(model, controller_jitter=control_jitter, controller_period=1,
                                                observation_delay=obs_delay, observation_jitter=observation_jitter,
                                                action_delay=action_delay, action_jitter=action_jitter)
                    for _ in range(50):
                        times, observations = make_trajectory(delayer)
                        plt.plot(times, observations, color=color, alpha=0.1)

            plt.show()

if __name__ == '__main__':
    main()