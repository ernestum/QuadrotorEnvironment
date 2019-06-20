import numpy as np
import pytest

from quadrotor_environment.propeller_speed_mapping import PropellerSpeedMapping


def test_direct_control():
    m = PropellerSpeedMapping(hovering_speed=100,
                              relative_min_speed=0,
                              relative_max_speed=2,
                              relative_max_propeller_acceleration=.1,
                              acceleration_constrain_mode='clip',
                              output_control_mode='direct')
    initial_relative_speed = m.reset(np.ones(4) * 100)
    assert np.all(np.isclose(initial_relative_speed, [1, 1, 1, 1]))

    # Test that just keeping the hovering speed behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 100, 100, 100]))
    assert np.all(np.isclose(relative_speed, [1, 1, 1, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, 0]))

    # Test increasing the speed within the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0.05, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 105, 100, 100]))
    assert np.all(np.isclose(relative_speed, [1, 1.05, 1, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0.05, 0, 0]))

    # Test increasing the speed at exactly the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0.05, 0.1, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 105, 110, 100]))
    assert np.all(np.isclose(relative_speed, [1, 1.05, 1.1, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, .1, 0]))

    # Test increasing the speed above the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0.05, 0.1, 0.2]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 105, 110, 110]))
    assert np.all(np.isclose(relative_speed, [1, 1.05, 1.1, 1.1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, .1]))

    # Test continued increasing the speed above the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0.05, 0.1, 0.2]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 105, 110, 120]))
    assert np.all(np.isclose(relative_speed, [1, 1.05, 1.1, 1.2]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, .1]))

    initial_relative_speed = m.reset(np.ones(4) * 100)
    assert np.all(np.isclose(initial_relative_speed, [1, 1, 1, 1]))

    # Test decreasing the speed within the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, -0.05, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 95, 100, 100]))
    assert np.all(np.isclose(relative_speed, [1, .95, 1, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, -0.05, 0, 0]))

    # Test decreasing the speed at exactly the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, -0.05, -0.1, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 95, 90, 100]))
    assert np.all(np.isclose(relative_speed, [1, 0.95, 0.9, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, -.1, 0]))

    # Test decreasing the speed above the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, -0.05, -0.1, -0.2]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 95, 90, 90]))
    assert np.all(np.isclose(relative_speed, [1, 0.95, 0.9, 0.9]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, -.1]))

    # Test continued decreasing the speed above the acceleration bounds behaves as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, -0.05, -0.1, -0.2]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 95, 90, 80]))
    assert np.all(np.isclose(relative_speed, [1, 0.95, 0.9, 0.8]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, -.1]))


def test_relative_control():
    m = PropellerSpeedMapping(hovering_speed=100,
                              relative_min_speed=0,
                              relative_max_speed=2,
                              relative_max_propeller_acceleration=.1,
                              acceleration_constrain_mode='clip',
                              output_control_mode='acceleration')
    initial_relative_speed = m.reset(np.ones(4) * 100)
    assert np.all(np.isclose(initial_relative_speed, [1, 1, 1, 1]))

    # Test that keeping the speed works as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 100, 100, 100]))
    assert np.all(np.isclose(relative_speed, [1, 1, 1, 1]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, 0]))

    # Test that increasing the speed within acceleration bounds works as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0, 0, 0.5]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 100, 100, 105]))
    assert np.all(np.isclose(relative_speed, [1, 1, 1, 1.05]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, 0.05]))

    # Test that increasing the speed at acceleration bounds works as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([1, 0, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [110, 100, 100, 105]))
    assert np.all(np.isclose(relative_speed, [1.1, 1, 1, 1.05]))
    assert np.all(np.isclose(relative_acceleration, [0.1, 0, 0, 0]))

    m.reset(np.ones(4) * 100)

    # Test that dereasing the speed within acceleration bounds works as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([0, 0, 0, -0.5]), 1)
    assert np.all(np.isclose(propeller_speed, [100, 100, 100, 95]))
    assert np.all(np.isclose(relative_speed, [1, 1, 1, 0.95]))
    assert np.all(np.isclose(relative_acceleration, [0, 0, 0, -0.05]))

    # Test that decreasing the speed at acceleration bounds works as expected
    propeller_speed, relative_speed, relative_acceleration = m(np.asarray([-1, 0, 0, 0]), 1)
    assert np.all(np.isclose(propeller_speed, [90, 100, 100, 95]))
    assert np.all(np.isclose(relative_speed, [0.9, 1, 1, 0.95]))
    assert np.all(np.isclose(relative_acceleration, [-0.1, 0, 0, 0]))

@pytest.mark.parametrize("output_control_mode", ["direct", "acceleration"])
@pytest.mark.parametrize("acceleration_constrain_mode", ["clip", "tanh"])
@pytest.mark.parametrize("hovering_speed", np.linspace(1, 1000, 5))
@pytest.mark.parametrize("relative_min_speed", np.linspace(-1, 1, 5))
@pytest.mark.parametrize("relative_speed_range", np.linspace(.01, 3, 5))
@pytest.mark.parametrize("relative_max_propeller_acceleration", np.linspace(0, 100, 5))
@pytest.mark.parametrize("dt", np.linspace(0.001, 1, 5))
def test_not_exceeding_bounds(output_control_mode, acceleration_constrain_mode, hovering_speed, relative_min_speed,
                              relative_speed_range, relative_max_propeller_acceleration, dt):
    relative_max_speed = relative_min_speed + relative_speed_range
    m = PropellerSpeedMapping(hovering_speed=hovering_speed,
                              relative_min_speed=relative_min_speed,
                              relative_max_speed=relative_max_speed,
                              relative_max_propeller_acceleration=relative_max_propeller_acceleration,
                              acceleration_constrain_mode=acceleration_constrain_mode,
                              output_control_mode=output_control_mode)
    relative_speed = m.reset(np.ones(1) * ((relative_max_speed-relative_min_speed) / 2 + relative_min_speed) * hovering_speed)
    assert np.all((relative_min_speed < relative_speed) & (relative_speed < relative_max_speed))
    for step in range(20):
        prev_relative_speed = np.copy(relative_speed)
        propeller_speed, relative_speed, relative_acceleration = m(np.random.uniform(-1, 1, 1), dt)

        # Check that the computed propeller acceleration is correct
        assert np.all(np.isclose((relative_speed - prev_relative_speed)/dt, relative_acceleration))

        # Check that relative and absolute speed match up
        assert np.all(np.isclose(relative_speed * hovering_speed, propeller_speed))

        # Check that the propeller acceleration is within bounds
        assert np.all(np.abs(relative_acceleration) <= relative_max_propeller_acceleration + 1e-10)

        # Check that the relative speed is within bounds
        assert np.all((relative_min_speed <= relative_speed) & (relative_speed <= relative_max_speed))

        # Check that the absolute speed is within bounds
        assert np.all((relative_min_speed * hovering_speed <= propeller_speed) & (propeller_speed <= relative_max_speed * hovering_speed))


