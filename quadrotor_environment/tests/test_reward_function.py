import numpy as np

from quadrotor_environment.quadrotor_environment import QuadrotorEnvironment
from quadrotor_environment.reward_function import RewardFunction


def test_term_list_length():
    """
    Test if the internal lists contianing the lambdas shrink and grow when parameters flip from 0 to nonzero.
    """
    zero_config = dict(
        reward_scale=1,
        position=0,
        velocity=0,
        angular_velocity=0,
        position_h=0,
        position_v=0,
        velocity_h=0,
        velocity_v=0,
        rotation_h=0,
        rotation_v=0,
        angular_velocity_h=0,
        angular_velocity_v=0,
        propeller_speed_deviation=0,
        propeller_acceleration=0,
        inverted_huber_reward_scaling=True)
    helper_env = QuadrotorEnvironment()

    # Check that a reward function with all zero terms as no terms and spits out zero reward no matter what we put in.
    rf = RewardFunction(**zero_config)
    assert len(rf.rotation_reward_terms) == 0
    assert len(rf.propeller_speed_reward_terms) == 0
    assert len(rf.simple_reward_terms) == 0
    for _ in range(100):
        assert rf(helper_env.random_state(), np.random.uniform(0, 3, 4), np.random.uniform(0, 3, 4)) == 0

    # Check that there are more reward terms if we make some of the factors nonzero (one for each group)
    # Simple reward term
    zero_config['velocity_h'] = 1.2
    rf = RewardFunction(**zero_config)
    assert len(rf.rotation_reward_terms) == 0
    assert len(rf.propeller_speed_reward_terms) == 0
    assert len(rf.simple_reward_terms) == 1

    # Propeller reward term
    zero_config['propeller_acceleration'] = 7
    rf = RewardFunction(**zero_config)
    assert len(rf.rotation_reward_terms) == 0
    assert len(rf.propeller_speed_reward_terms) == 1
    assert len(rf.simple_reward_terms) == 1

    # rotation reward term
    zero_config['rotation_h'] = 7
    rf = RewardFunction(**zero_config)
    assert len(rf.rotation_reward_terms) == 1
    assert len(rf.propeller_speed_reward_terms) == 1
    assert len(rf.simple_reward_terms) == 1

#TODO: check if increasing scale increases output
#TODO: check that inverted huber loss scaling works as expected
#TODO: check that we always take absolute values, norms

