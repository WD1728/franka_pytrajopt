from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from franka_pytrajopt.logging.logger import ResultLogger
from franka_pytrajopt.robot.planar_arm import PlanarArm2D
from franka_pytrajopt.trajopt.constraints import fixed_endpoint_constraints, trust_region_constraints
from franka_pytrajopt.trajopt.costs import smoothness_cost, smoothness_cost_expr
from franka_pytrajopt.trajopt.linearization import segment_sample_points
from franka_pytrajopt.trajopt.problem import PlanarArmConfig
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


@dataclass(frozen=True)
class PlanarArmCollisionSample:
    left_index: int
    right_index: int | None
    segment_alpha: float | None
    q_current: np.ndarray
    workspace_point: np.ndarray
    distance: float
    gradient: np.ndarray
    link_index: int
    local_alpha: float
    jacobian: np.ndarray

    @property
    def kind(self) -> str:
        return "waypoint" if self.right_index is None else "segment"


@dataclass(frozen=True)
class PlanarArmOptimizationResult:
    status: str
    iterations: int
    final_objective: float
    final_sampled_min_signed_distance: float
    final_sampled_max_penetration_depth: float
    final_sampled_collision_penalty: float
    final_smoothness_cost: float
    final_has_link_collision: bool
    final_colliding_waypoints: list[int]
    final_colliding_segments: list[int]
    final_trajectory_csv: Path


class PlanarArmTrajectoryOptimizer:
    def __init__(
        self,
        config: PlanarArmConfig,
        arm: PlanarArm2D,
        obstacle: AxisAlignedBox2D,
        logger: ResultLogger,
    ) -> None:
        self.config = config
        self.arm = arm
        self.obstacle = obstacle
        self.logger = logger

    @property
    def link_sample_alphas(self) -> list[float]:
        return [float(alpha) for alpha in self.config.optimizer.link_sample_alphas]

    @property
    def segment_alphas(self) -> list[float]:
        return [float(alpha) for alpha in self.config.optimizer.segment_collision_alphas]

    @property
    def collision_offset(self) -> float:
        return float(self.config.optimizer.collision_margin + self.config.robot.link_radius)

    @property
    def link_sampling_buffer(self) -> float:
        sorted_alphas = sorted({0.0, 1.0, *self.link_sample_alphas})
        max_gap = max(
            sorted_alphas[idx + 1] - sorted_alphas[idx]
            for idx in range(len(sorted_alphas) - 1)
        )
        return 0.5 * float(np.max(self.arm.link_lengths_array)) * max_gap

    @property
    def sampled_clearance_threshold(self) -> float:
        return float(self.config.optimizer.collision_margin + self.link_sampling_buffer)

    def seed_trajectory(self) -> np.ndarray:
        alphas = np.linspace(0.0, 1.0, self.config.num_waypoints)
        return (1.0 - alphas[:, None]) * self.config.start_q + alphas[:, None] * self.config.goal_q

    def collision_samples_for_trajectory(self, trajectory: np.ndarray) -> list[PlanarArmCollisionSample]:
        samples: list[PlanarArmCollisionSample] = []

        for idx, q in enumerate(trajectory):
            samples.extend(self._state_collision_samples(q=q, left_index=idx, right_index=None, segment_alpha=None))

        for segment_index, segment_alpha, q_alpha in segment_sample_points(trajectory, self.segment_alphas):
            samples.extend(
                self._state_collision_samples(
                    q=q_alpha,
                    left_index=segment_index,
                    right_index=segment_index + 1,
                    segment_alpha=segment_alpha,
                )
            )

        return samples

    def _state_collision_samples(
        self,
        q: np.ndarray,
        left_index: int,
        right_index: int | None,
        segment_alpha: float | None,
    ) -> list[PlanarArmCollisionSample]:
        state_samples: list[PlanarArmCollisionSample] = []
        for link_index in range(2):
            for local_alpha in self.link_sample_alphas:
                point = self.arm.point_on_link(q, link_index, local_alpha)
                distance, gradient = self.obstacle.signed_distance_with_gradient(point)
                jacobian = self.arm.point_jacobian(q, link_index, local_alpha)
                state_samples.append(
                    PlanarArmCollisionSample(
                        left_index=left_index,
                        right_index=right_index,
                        segment_alpha=segment_alpha,
                        q_current=np.asarray(q, dtype=float),
                        workspace_point=np.asarray(point, dtype=float),
                        distance=float(distance),
                        gradient=np.asarray(gradient, dtype=float),
                        link_index=link_index,
                        local_alpha=float(local_alpha),
                        jacobian=np.asarray(jacobian, dtype=float),
                    )
                )
        return state_samples

    def sample_collision_distances(self, trajectory: np.ndarray) -> np.ndarray:
        distances = [sample.distance - self.config.robot.link_radius for sample in self.collision_samples_for_trajectory(trajectory)]
        return np.asarray(distances, dtype=float)

    def validate_trajectory(self, trajectory: np.ndarray) -> dict[str, list[int] | bool]:
        inflated_box = self.obstacle.inflated(self.collision_offset)
        colliding_waypoints: list[int] = []
        colliding_segments: list[int] = []

        for idx, q in enumerate(trajectory):
            if self._state_has_inflated_box_collision(inflated_box, q):
                colliding_waypoints.append(idx)

        for segment_index, _, q_alpha in segment_sample_points(trajectory, self.segment_alphas):
            if self._state_has_inflated_box_collision(inflated_box, q_alpha) and segment_index not in colliding_segments:
                colliding_segments.append(segment_index)

        return {
            "has_link_collision": bool(colliding_waypoints or colliding_segments),
            "colliding_waypoints": colliding_waypoints,
            "colliding_segments": colliding_segments,
        }

    def _state_has_inflated_box_collision(self, inflated_box: AxisAlignedBox2D, q: np.ndarray) -> bool:
        return any(inflated_box.segment_intersects(start, end) for start, end in self.arm.link_segments(q))

    def evaluate_true_metrics(self, trajectory: np.ndarray, penalty_coeff: float) -> dict[str, float | bool | list[int]]:
        distances = self.sample_collision_distances(trajectory)
        smooth = smoothness_cost(trajectory)
        hinge = np.maximum(0.0, self.sampled_clearance_threshold - distances)
        total_violation = float(np.sum(hinge))
        collision_penalty = penalty_coeff * total_violation
        merit = self.config.optimizer.smoothness_weight * smooth + collision_penalty
        validation = self.validate_trajectory(trajectory)

        return {
            "objective": float(merit),
            "smoothness_cost": float(smooth),
            "sampled_min_signed_distance": float(np.min(distances)),
            "sampled_max_penetration_depth": float(np.max(np.maximum(0.0, -distances))),
            "sampled_collision_penalty": float(collision_penalty),
            "constraint_violation": float(total_violation),
            "has_segment_collision": bool(validation["has_link_collision"]),
            "colliding_segments": list(validation["colliding_segments"]),
            "has_link_collision": bool(validation["has_link_collision"]),
            "colliding_waypoints": list(validation["colliding_waypoints"]),
        }

    def evaluate_model_metrics(
        self,
        candidate: np.ndarray,
        current: np.ndarray,
        penalty_coeff: float,
    ) -> dict[str, float]:
        _ = current
        samples = self.collision_samples_for_trajectory(current)
        affine_distances = []
        for sample in samples:
            q_sample = self.q_sample_value(candidate, sample)
            grad_q = sample.gradient @ sample.jacobian
            affine_distance = sample.distance + float(grad_q @ (q_sample - sample.q_current))
            affine_distances.append(affine_distance - self.config.robot.link_radius)

        affine_distances_arr = np.asarray(affine_distances, dtype=float)
        hinge = np.maximum(0.0, self.sampled_clearance_threshold - affine_distances_arr)
        total_violation = float(np.sum(hinge))
        collision_penalty = penalty_coeff * total_violation
        smooth = smoothness_cost(candidate)
        merit = self.config.optimizer.smoothness_weight * smooth + collision_penalty
        return {
            "objective": float(merit),
            "smoothness_cost": float(smooth),
            "sampled_collision_penalty": float(collision_penalty),
            "constraint_violation": float(total_violation),
        }

    def q_sample_expr(self, trajectory_var: cp.Expression, sample: PlanarArmCollisionSample) -> cp.Expression:
        if sample.right_index is None:
            return trajectory_var[sample.left_index]

        assert sample.segment_alpha is not None
        return (
            (1.0 - sample.segment_alpha) * trajectory_var[sample.left_index]
            + sample.segment_alpha * trajectory_var[sample.right_index]
        )

    def q_sample_value(self, trajectory: np.ndarray, sample: PlanarArmCollisionSample) -> np.ndarray:
        if sample.right_index is None:
            return trajectory[sample.left_index]

        assert sample.segment_alpha is not None
        return (
            (1.0 - sample.segment_alpha) * trajectory[sample.left_index]
            + sample.segment_alpha * trajectory[sample.right_index]
        )

    def solve_convex_subproblem(
        self,
        current: np.ndarray,
        trust_region_radius: float,
        penalty_coeff: float,
    ) -> tuple[np.ndarray, str]:
        num_waypoints = self.config.num_waypoints
        trajectory_var = cp.Variable((num_waypoints, 2))
        samples = self.collision_samples_for_trajectory(current)

        objective_terms: list[cp.Expression] = [
            self.config.optimizer.smoothness_weight * smoothness_cost_expr(trajectory_var)
        ]

        collision_terms: list[cp.Expression] = []
        for sample in samples:
            q_var_sample = self.q_sample_expr(trajectory_var, sample)
            grad_q = sample.gradient @ sample.jacobian
            affine_sdf = sample.distance + grad_q @ (q_var_sample - sample.q_current)
            collision_terms.append(cp.pos(self.collision_offset + self.link_sampling_buffer - affine_sdf))
        objective_terms.append(penalty_coeff * cp.sum(cp.hstack(collision_terms)))

        constraints = []
        constraints.extend(fixed_endpoint_constraints(trajectory_var, self.config.start_q, self.config.goal_q))
        constraints.extend(trust_region_constraints(trajectory_var, current, trust_region_radius))

        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
        problem.solve(solver=getattr(cp, self.config.optimizer.solver), warm_start=True)

        if trajectory_var.value is None:
            raise RuntimeError(f"Convex subproblem failed with status: {problem.status}")

        return np.asarray(trajectory_var.value, dtype=float), str(problem.status)

    def optimize(self) -> PlanarArmOptimizationResult:
        opt_cfg = self.config.optimizer
        current = self.seed_trajectory()
        penalty_coeff = opt_cfg.collision_weight
        trust_region_radius = opt_cfg.trust_region_radius
        current_metrics = self.evaluate_true_metrics(current, penalty_coeff)

        accepted_iteration = 0
        overall_iteration = 0
        status = "penalty_iteration_limit"
        converged = False

        self.logger.log_iteration(
            0,
            current,
            current_metrics,
            step_norm=0.0,
            subproblem_status="seed",
            trust_region_radius=trust_region_radius,
            penalty_coeff=penalty_coeff,
            approx_merit=current_metrics["objective"],
            exact_merit=current_metrics["objective"],
            approx_improve=0.0,
            exact_improve=0.0,
            merit_improve_ratio=0.0,
            accepted=True,
            penalty_iteration=0,
        )

        for penalty_iteration in range(opt_cfg.max_merit_coeff_increases):
            current_metrics = self.evaluate_true_metrics(current, penalty_coeff)
            converged_this_penalty = False

            while overall_iteration < opt_cfg.max_iterations:
                candidate, subproblem_status = self.solve_convex_subproblem(current, trust_region_radius, penalty_coeff)
                overall_iteration += 1

                model_metrics = self.evaluate_model_metrics(candidate, current, penalty_coeff)
                candidate_metrics = self.evaluate_true_metrics(candidate, penalty_coeff)

                approx_improve = current_metrics["objective"] - model_metrics["objective"]
                exact_improve = current_metrics["objective"] - candidate_metrics["objective"]
                merit_improve_ratio = 0.0 if abs(approx_improve) < 1.0e-12 else exact_improve / approx_improve
                step_norm = float(np.max(np.linalg.norm(candidate - current, axis=1)))
                accepted = exact_improve >= 0.0 and merit_improve_ratio >= opt_cfg.improve_ratio_threshold

                self.logger.log_iteration(
                    overall_iteration,
                    candidate,
                    candidate_metrics,
                    step_norm=step_norm,
                    subproblem_status=subproblem_status,
                    trust_region_radius=trust_region_radius,
                    penalty_coeff=penalty_coeff,
                    approx_merit=model_metrics["objective"],
                    exact_merit=candidate_metrics["objective"],
                    approx_improve=approx_improve,
                    exact_improve=exact_improve,
                    merit_improve_ratio=merit_improve_ratio,
                    accepted=accepted,
                    penalty_iteration=penalty_iteration,
                )

                if accepted:
                    current = candidate
                    current_metrics = candidate_metrics
                    accepted_iteration = overall_iteration
                    trust_region_radius *= opt_cfg.trust_expand_ratio

                    if (
                        current_metrics["constraint_violation"] <= opt_cfg.constraint_tolerance
                        and not current_metrics["has_link_collision"]
                    ):
                        status = "converged_constraint_tolerance"
                        converged = True
                        break

                    if approx_improve < opt_cfg.min_approx_improve:
                        converged_this_penalty = True
                        break

                    denom = max(abs(current_metrics["objective"]), 1.0e-12)
                    if (approx_improve / denom) < opt_cfg.min_approx_improve_frac:
                        converged_this_penalty = True
                        break

                    continue

                trust_region_radius *= opt_cfg.trust_shrink_ratio
                if trust_region_radius < opt_cfg.min_trust_region_radius:
                    converged_this_penalty = True
                    break

            if converged:
                break

            current_metrics = self.evaluate_true_metrics(current, penalty_coeff)
            if (
                current_metrics["constraint_violation"] <= opt_cfg.constraint_tolerance
                and not current_metrics["has_link_collision"]
            ):
                status = "converged_constraint_tolerance"
                break

            if overall_iteration >= opt_cfg.max_iterations:
                status = "max_iterations_reached"
                break

            penalty_coeff *= opt_cfg.merit_coeff_increase_ratio
            trust_region_radius = max(
                trust_region_radius,
                opt_cfg.min_trust_region_radius / opt_cfg.trust_shrink_ratio * 1.5,
            )

            if not converged_this_penalty:
                continue

        final_csv = self.logger.iteration_csv_path(accepted_iteration)
        final_metrics = self.evaluate_true_metrics(current, penalty_coeff)
        final_has_link_collision = bool(final_metrics["has_link_collision"])
        final_colliding_waypoints = list(final_metrics["colliding_waypoints"])
        final_colliding_segments = list(final_metrics["colliding_segments"])

        if status == "converged_constraint_tolerance" and final_has_link_collision:
            status = "converged_sampled_only_but_exact_link_collision"

        return PlanarArmOptimizationResult(
            status=status,
            iterations=overall_iteration,
            final_objective=float(final_metrics["objective"]),
            final_sampled_min_signed_distance=float(final_metrics["sampled_min_signed_distance"]),
            final_sampled_max_penetration_depth=float(final_metrics["sampled_max_penetration_depth"]),
            final_sampled_collision_penalty=float(final_metrics["sampled_collision_penalty"]),
            final_smoothness_cost=float(final_metrics["smoothness_cost"]),
            final_has_link_collision=final_has_link_collision,
            final_colliding_waypoints=final_colliding_waypoints,
            final_colliding_segments=final_colliding_segments,
            final_trajectory_csv=final_csv,
        )
