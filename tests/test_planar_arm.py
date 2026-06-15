from __future__ import annotations

from pathlib import Path

import numpy as np

from franka_pytrajopt.logging.logger import ResultLogger
from franka_pytrajopt.robot.planar_arm import PlanarArm2D
from franka_pytrajopt.trajopt.planar_arm_optimizer import PlanarArmTrajectoryOptimizer
from franka_pytrajopt.trajopt.problem import (
    ObstacleConfig,
    PlanarArmConfig,
    PlanarArmOptimizerConfig,
    PlanarArmProblemConfig,
    PlanarArmRobotConfig,
)
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def make_planar_arm_config() -> PlanarArmConfig:
    return PlanarArmConfig(
        problem=PlanarArmProblemConfig(
            num_waypoints=17,
            start_q=[-1.2, 1.8],
            goal_q=[1.2, 1.8],
        ),
        robot=PlanarArmRobotConfig(
            link_lengths=[1.2, 1.0],
            link_radius=0.08,
            base=[0.0, 0.0],
        ),
        obstacle=ObstacleConfig(
            center=[0.95, 0.55],
            size=[0.22, 0.32],
        ),
        optimizer=PlanarArmOptimizerConfig(
            smoothness_weight=1.0,
            collision_weight=30.0,
            collision_margin=0.15,
            link_sample_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
            segment_collision_alphas=[0.25, 0.5, 0.75],
            trust_region_radius=0.35,
            min_trust_region_radius=0.01,
            trust_shrink_ratio=0.5,
            trust_expand_ratio=1.5,
            improve_ratio_threshold=0.25,
            max_merit_coeff_increases=5,
            merit_coeff_increase_ratio=10.0,
            constraint_tolerance=1.0e-3,
            max_iterations=50,
            min_approx_improve=1.0e-5,
            min_approx_improve_frac=1.0e-6,
            solver="OSQP",
        ),
    )


def test_planar_arm_forward_kinematics():
    arm = PlanarArm2D(link_lengths=[1.0, 1.0], link_radius=0.05, base=[0.0, 0.0])
    fk = arm.forward_kinematics(np.array([0.0, np.pi / 2.0], dtype=float))

    assert np.allclose(fk["base"], [0.0, 0.0])
    assert np.allclose(fk["elbow"], [1.0, 0.0])
    assert np.allclose(fk["wrist"], [1.0, 1.0], atol=1.0e-8)


def test_planar_arm_point_jacobian_matches_finite_difference():
    arm = PlanarArm2D(link_lengths=[1.2, 0.9], link_radius=0.05, base=[0.0, 0.0])
    q = np.array([0.4, -0.8], dtype=float)
    eps = 1.0e-6

    for link_index in [0, 1]:
        for local_alpha in [0.25, 0.75]:
            analytic = arm.point_jacobian(q, link_index, local_alpha)
            numeric = np.zeros((2, 2), dtype=float)
            for joint_idx in range(2):
                dq = np.zeros(2, dtype=float)
                dq[joint_idx] = eps
                plus = arm.point_on_link(q + dq, link_index, local_alpha)
                minus = arm.point_on_link(q - dq, link_index, local_alpha)
                numeric[:, joint_idx] = (plus - minus) / (2.0 * eps)

            assert np.allclose(analytic, numeric, atol=1.0e-5)


def test_q_space_collision_linearization_gradient_sign_sanity():
    config = make_planar_arm_config()
    arm = PlanarArm2D(
        link_lengths=config.robot.link_lengths,
        link_radius=config.robot.link_radius,
        base=config.robot.base,
    )
    obstacle = AxisAlignedBox2D(center=config.obstacle.center, size=config.obstacle.size)
    q = np.array([0.2, 0.7], dtype=float)
    link_index = 1
    local_alpha = 0.5

    point = arm.point_on_link(q, link_index, local_alpha)
    sdf, normal = obstacle.signed_distance_with_gradient(point)
    jacobian = arm.point_jacobian(q, link_index, local_alpha)
    grad_q = normal @ jacobian

    assert grad_q.shape == (2,)

    if np.linalg.norm(grad_q) > 1.0e-8:
        direction = grad_q / np.linalg.norm(grad_q)
        eps = 1.0e-6
        point_plus = arm.point_on_link(q + eps * direction, link_index, local_alpha)
        point_minus = arm.point_on_link(q - eps * direction, link_index, local_alpha)
        numeric_dir_derivative = (obstacle.signed_distance(point_plus) - obstacle.signed_distance(point_minus)) / (2.0 * eps)
        analytic_dir_derivative = float(grad_q @ direction)
        assert np.isclose(analytic_dir_derivative, numeric_dir_derivative, atol=1.0e-4)
    else:
        assert np.isclose(sdf, obstacle.signed_distance(point))


def test_planar_arm_final_trajectory_has_no_inflated_box_collision(tmp_path: Path):
    config = make_planar_arm_config()
    arm = PlanarArm2D(
        link_lengths=config.robot.link_lengths,
        link_radius=config.robot.link_radius,
        base=config.robot.base,
    )
    obstacle = AxisAlignedBox2D(center=config.obstacle.center, size=config.obstacle.size)
    logger = ResultLogger(tmp_path / "result", trajectory_column_names=["q1", "q2"])
    optimizer = PlanarArmTrajectoryOptimizer(config=config, arm=arm, obstacle=obstacle, logger=logger)

    result = optimizer.optimize()
    final_traj = np.loadtxt(result.final_trajectory_csv, delimiter=",", skiprows=1, usecols=(1, 2))
    final_metrics = optimizer.evaluate_true_metrics(final_traj, penalty_coeff=config.optimizer.collision_weight)

    assert result.status == "converged_constraint_tolerance"
    assert result.final_has_link_collision is False
    assert result.final_colliding_waypoints == []
    assert result.final_colliding_segments == []
    assert final_metrics["has_link_collision"] is False
    assert final_metrics["sampled_max_penetration_depth"] <= 1.0e-8
