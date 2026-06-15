from __future__ import annotations

from pathlib import Path

import numpy as np

from franka_pytrajopt.logging.logger import ResultLogger
from franka_pytrajopt.trajopt.linearization import segment_sample_points
from franka_pytrajopt.trajopt.optimizer import TrajOptOptimizer
from franka_pytrajopt.trajopt.problem import ObstacleConfig, OptimizerConfig, Point2DConfig, ProblemConfig
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def make_test_config() -> Point2DConfig:
    return Point2DConfig(
        problem=ProblemConfig(num_waypoints=21, dimensions=2, start=[0.0, 0.0], goal=[10.0, 0.0]),
        obstacle=ObstacleConfig(center=[5.0, 0.0], size=[2.0, 3.0]),
        optimizer=OptimizerConfig(
            smoothness_weight=1.0,
            collision_weight=25.0,
            collision_margin=0.25,
            segment_collision_alphas=[0.25, 0.5, 0.75],
            trust_region_radius=0.75,
            min_trust_region_radius=0.01,
            trust_shrink_ratio=0.5,
            trust_expand_ratio=1.5,
            improve_ratio_threshold=0.25,
            max_merit_coeff_increases=4,
            merit_coeff_increase_ratio=10.0,
            constraint_tolerance=1.0e-3,
            max_iterations=25,
            min_approx_improve=1.0e-5,
            min_approx_improve_frac=1.0e-6,
            max_step=0.75,
            seed_y_offset=0.75,
            solver="OSQP",
        ),
    )


def test_segment_sampling_detects_waypoint_only_failure_case():
    box = AxisAlignedBox2D(center=[5.0, 0.0], size=[2.0, 3.0])
    trajectory = np.array([[3.75, 0.0], [6.25, 0.0]], dtype=float)

    waypoint_distances = np.asarray([box.signed_distance(point) for point in trajectory], dtype=float)
    assert np.all(waypoint_distances >= 0.0)

    sample_distances = np.asarray(
        [box.signed_distance(point) for _, _, point in segment_sample_points(trajectory, [0.25, 0.5, 0.75])],
        dtype=float,
    )
    assert float(np.min(sample_distances)) < 0.0
    assert box.segment_intersects(trajectory[0], trajectory[1])
    assert box.segment_collision_indices(trajectory) == [0]


def test_final_optimized_trajectory_has_no_sampled_penetration(tmp_path: Path):
    config = make_test_config()
    obstacle = AxisAlignedBox2D(center=config.obstacle.center, size=config.obstacle.size)
    logger = ResultLogger(tmp_path / "result")
    optimizer = TrajOptOptimizer(config=config, obstacle=obstacle, logger=logger)

    result = optimizer.optimize()
    final_traj = np.loadtxt(result.final_trajectory_csv, delimiter=",", skiprows=1, usecols=(1, 2))
    final_metrics = optimizer.evaluate_true_metrics(final_traj, penalty_coeff=config.optimizer.collision_weight)

    assert result.status == "converged_constraint_tolerance"
    assert final_metrics["sampled_max_penetration_depth"] <= 1.0e-8
    assert not result.final_has_segment_collision
    assert result.final_colliding_segments == []
    assert obstacle.segment_collision_indices(final_traj) == []


def test_default_seed_penetrates_but_is_not_centered(tmp_path: Path):
    config = make_test_config()
    obstacle = AxisAlignedBox2D(center=config.obstacle.center, size=config.obstacle.size)
    logger = ResultLogger(tmp_path / "result")
    optimizer = TrajOptOptimizer(config=config, obstacle=obstacle, logger=logger)

    seed = optimizer.seed_trajectory()
    seed_metrics = optimizer.evaluate_true_metrics(seed, penalty_coeff=config.optimizer.collision_weight)

    assert config.optimizer.seed_type == "offset_straight"
    assert seed_metrics["sampled_max_penetration_depth"] > 0.0
    assert obstacle.segment_collision_indices(seed) != []
    assert not np.allclose(seed[:, 1], 0.0)


def test_config_defaults_are_backward_compatible():
    config = Point2DConfig.from_dict(
        {
            "problem": {
                "num_waypoints": 5,
                "dimensions": 2,
                "start": [0.0, 0.0],
                "goal": [1.0, 0.0],
            },
            "obstacle": {
                "center": [0.5, 0.0],
                "size": [0.2, 0.2],
            },
            "optimizer": {
                "smoothness_weight": 1.0,
                "collision_weight": 10.0,
                "collision_margin": 0.1,
                "trust_region_radius": 0.5,
                "min_trust_region_radius": 0.01,
                "trust_shrink_ratio": 0.5,
                "trust_expand_ratio": 1.5,
                "improve_ratio_threshold": 0.25,
                "max_merit_coeff_increases": 2,
                "merit_coeff_increase_ratio": 10.0,
                "constraint_tolerance": 1.0e-3,
                "max_iterations": 10,
                "min_approx_improve": 1.0e-5,
                "min_approx_improve_frac": 1.0e-6,
                "solver": "OSQP",
            },
        }
    )

    assert config.optimizer.segment_collision_alphas == [0.25, 0.5, 0.75]
    assert config.optimizer.max_step is None
    assert config.optimizer.seed_type == "offset_straight"
    assert np.isclose(config.optimizer.seed_y_offset, 0.1)
