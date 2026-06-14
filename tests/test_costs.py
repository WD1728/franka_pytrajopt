import numpy as np

from franka_pytrajopt.trajopt.costs import smoothness_cost


def test_smoothness_cost_straight_line_constant_step():
    trajectory = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    assert np.isclose(smoothness_cost(trajectory), 2.0)


def test_smoothness_cost_zero_for_repeated_waypoints():
    trajectory = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    assert np.isclose(smoothness_cost(trajectory), 0.0)
