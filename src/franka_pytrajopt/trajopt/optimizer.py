from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from franka_pytrajopt.logging.logger import ResultLogger
from franka_pytrajopt.trajopt.constraints import fixed_endpoint_constraints, trust_region_constraints
from franka_pytrajopt.trajopt.costs import smoothness_cost, smoothness_cost_expr
from franka_pytrajopt.trajopt.linearization import linearize_signed_distance
from franka_pytrajopt.trajopt.problem import Point2DConfig
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    iterations: int
    final_objective: float
    final_min_signed_distance: float
    final_max_penetration_depth: float
    final_smoothness_cost: float
    final_trajectory_csv: Path


class TrajOptOptimizer:
    def __init__(
        self,
        config: Point2DConfig,
        obstacle: AxisAlignedBox2D,
        logger: ResultLogger,
    ) -> None:
        self.config = config
        self.obstacle = obstacle
        self.logger = logger

        if config.dimensions != 2:
            raise ValueError("This milestone only supports 2D trajectories.")
        if config.num_waypoints < 2:
            raise ValueError("At least two waypoints are required.")

    def seed_trajectory(self) -> np.ndarray:
        alphas = np.linspace(0.0, 1.0, self.config.num_waypoints)
        return (1.0 - alphas[:, None]) * self.config.start + alphas[:, None] * self.config.goal

    def evaluate_true_metrics(self, trajectory: np.ndarray, penalty_coeff: float) -> dict[str, float]:
        distances = np.asarray([self.obstacle.signed_distance(point) for point in trajectory], dtype=float)
        smooth = smoothness_cost(trajectory)
        margin = self.config.optimizer.collision_margin
        hinge = np.maximum(0.0, margin - distances)
        total_violation = float(np.sum(hinge))
        collision_penalty = penalty_coeff * total_violation
        merit = self.config.optimizer.smoothness_weight * smooth + collision_penalty
        return {
            "objective": float(merit),
            "smoothness_cost": float(smooth),
            "min_signed_distance": float(np.min(distances)),
            "max_penetration_depth": float(np.max(np.maximum(0.0, -distances))),
            "collision_penalty": float(collision_penalty),
            "constraint_violation": float(total_violation),
        }

    def evaluate_model_metrics(
        self,
        candidate: np.ndarray,
        current: np.ndarray,
        penalty_coeff: float,
    ) -> dict[str, float]:
        linearization = linearize_signed_distance(current, self.obstacle)
        smooth = smoothness_cost(candidate)
        affine_distances = []
        for idx in range(self.config.num_waypoints):
            affine_distance = linearization.distances[idx] + float(
                linearization.gradients[idx] @ (candidate[idx] - current[idx])
            )
            affine_distances.append(affine_distance)

        affine_distances_arr = np.asarray(affine_distances, dtype=float)
        hinge = np.maximum(0.0, self.config.optimizer.collision_margin - affine_distances_arr)
        total_violation = float(np.sum(hinge))
        collision_penalty = penalty_coeff * total_violation
        merit = self.config.optimizer.smoothness_weight * smooth + collision_penalty
        return {
            "objective": float(merit),
            "smoothness_cost": float(smooth),
            "collision_penalty": float(collision_penalty),
            "constraint_violation": float(total_violation),
        }

    def solve_convex_subproblem(
        self,
        current: np.ndarray,
        trust_region_radius: float,
        penalty_coeff: float,
    ) -> tuple[np.ndarray, str]:
        num_waypoints = self.config.num_waypoints
        dim = self.config.dimensions
        opt_cfg = self.config.optimizer

        linearization = linearize_signed_distance(current, self.obstacle)
        trajectory_var = cp.Variable((num_waypoints, dim))

        objective_terms = [
            opt_cfg.smoothness_weight * smoothness_cost_expr(trajectory_var),
        ]

        collision_terms = []
        for idx in range(num_waypoints):
            affine_sdf = linearization.distances[idx] + linearization.gradients[idx] @ (trajectory_var[idx] - current[idx])
            collision_terms.append(cp.pos(opt_cfg.collision_margin - affine_sdf))
        objective_terms.append(penalty_coeff * cp.sum(cp.hstack(collision_terms)))

        constraints = []
        constraints.extend(fixed_endpoint_constraints(trajectory_var, self.config.start, self.config.goal))
        constraints.extend(trust_region_constraints(trajectory_var, current, trust_region_radius))

        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
        problem.solve(solver=getattr(cp, opt_cfg.solver), warm_start=True)

        if trajectory_var.value is None:
            raise RuntimeError(f"Convex subproblem failed with status: {problem.status}")

        return np.asarray(trajectory_var.value, dtype=float), str(problem.status)

    def optimize(self) -> OptimizationResult:
        opt_cfg = self.config.optimizer
        current = self.seed_trajectory()
        penalty_coeff = opt_cfg.collision_weight
        trust_region_radius = opt_cfg.trust_region_radius
        current_metrics = self.evaluate_true_metrics(current, penalty_coeff)

        accepted_iteration = 0
        overall_iteration = 0
        status = "penalty_iteration_limit"

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
                if abs(approx_improve) < 1.0e-12:
                    merit_improve_ratio = 0.0
                else:
                    merit_improve_ratio = exact_improve / approx_improve

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

                if approx_improve < opt_cfg.min_approx_improve:
                    converged_this_penalty = True
                    break

                denom = max(abs(current_metrics["objective"]), 1.0e-12)
                if (approx_improve / denom) < opt_cfg.min_approx_improve_frac:
                    converged_this_penalty = True
                    break

                if accepted:
                    current = candidate
                    current_metrics = candidate_metrics
                    accepted_iteration = overall_iteration
                    trust_region_radius *= opt_cfg.trust_expand_ratio
                    break

                trust_region_radius *= opt_cfg.trust_shrink_ratio
                if trust_region_radius < opt_cfg.min_trust_region_radius:
                    converged_this_penalty = True
                    break

            current_metrics = self.evaluate_true_metrics(current, penalty_coeff)
            if current_metrics["constraint_violation"] <= opt_cfg.constraint_tolerance:
                status = "converged_constraint_tolerance"
                break

            if overall_iteration >= opt_cfg.max_iterations:
                status = "max_iterations_reached"
                break

            if converged_this_penalty:
                penalty_coeff *= opt_cfg.merit_coeff_increase_ratio
                trust_region_radius = max(
                    trust_region_radius,
                    opt_cfg.min_trust_region_radius / opt_cfg.trust_shrink_ratio * 1.5,
                )
                continue

            penalty_coeff *= opt_cfg.merit_coeff_increase_ratio
            trust_region_radius = max(
                trust_region_radius,
                opt_cfg.min_trust_region_radius / opt_cfg.trust_shrink_ratio * 1.5,
            )

        final_csv = self.logger.iteration_csv_path(accepted_iteration)
        final_metrics = self.evaluate_true_metrics(current, penalty_coeff)

        return OptimizationResult(
            status=status,
            iterations=overall_iteration,
            final_objective=final_metrics["objective"],
            final_min_signed_distance=final_metrics["min_signed_distance"],
            final_max_penetration_depth=final_metrics["max_penetration_depth"],
            final_smoothness_cost=final_metrics["smoothness_cost"],
            final_trajectory_csv=final_csv,
        )
