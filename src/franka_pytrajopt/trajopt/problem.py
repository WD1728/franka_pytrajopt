from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ObstacleConfig:
    center: list[float]
    size: list[float]


@dataclass(frozen=True)
class OptimizerConfig:
    smoothness_weight: float
    collision_weight: float
    collision_margin: float
    segment_collision_alphas: list[float]
    trust_region_radius: float
    min_trust_region_radius: float
    trust_shrink_ratio: float
    trust_expand_ratio: float
    improve_ratio_threshold: float
    max_merit_coeff_increases: int
    merit_coeff_increase_ratio: float
    constraint_tolerance: float
    max_iterations: int
    min_approx_improve: float
    min_approx_improve_frac: float
    max_step: float | None = None
    seed_type: str = "offset_straight"
    seed_y_offset: float = 0.1
    solver: str = "OSQP"


@dataclass(frozen=True)
class ProblemConfig:
    num_waypoints: int
    dimensions: int
    start: list[float]
    goal: list[float]


@dataclass(frozen=True)
class PlanarArmRobotConfig:
    link_lengths: list[float]
    link_radius: float
    base: list[float]


@dataclass(frozen=True)
class PlanarArmProblemConfig:
    num_waypoints: int
    start_q: list[float]
    goal_q: list[float]


@dataclass(frozen=True)
class PlanarArmOptimizerConfig:
    smoothness_weight: float
    collision_weight: float
    collision_margin: float
    link_sample_alphas: list[float]
    segment_collision_alphas: list[float]
    trust_region_radius: float
    min_trust_region_radius: float
    trust_shrink_ratio: float
    trust_expand_ratio: float
    improve_ratio_threshold: float
    max_merit_coeff_increases: int
    merit_coeff_increase_ratio: float
    constraint_tolerance: float
    max_iterations: int
    min_approx_improve: float
    min_approx_improve_frac: float
    solver: str = "OSQP"


@dataclass(frozen=True)
class Point2DConfig:
    problem: ProblemConfig
    obstacle: ObstacleConfig
    optimizer: OptimizerConfig

    @staticmethod
    def from_dict(data: dict) -> "Point2DConfig":
        optimizer_data = dict(data["optimizer"])
        optimizer_data.setdefault("segment_collision_alphas", [0.25, 0.5, 0.75])
        optimizer_data.setdefault("max_step", None)
        optimizer_data.setdefault("seed_type", "offset_straight")
        optimizer_data.setdefault("seed_y_offset", 0.1)
        return Point2DConfig(
            problem=ProblemConfig(**data["problem"]),
            obstacle=ObstacleConfig(**data["obstacle"]),
            optimizer=OptimizerConfig(**optimizer_data),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.problem.start, dtype=float)

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.problem.goal, dtype=float)

    @property
    def num_waypoints(self) -> int:
        return self.problem.num_waypoints

    @property
    def dimensions(self) -> int:
        return self.problem.dimensions


@dataclass(frozen=True)
class PlanarArmConfig:
    problem: PlanarArmProblemConfig
    robot: PlanarArmRobotConfig
    obstacle: ObstacleConfig
    optimizer: PlanarArmOptimizerConfig

    @staticmethod
    def from_dict(data: dict) -> "PlanarArmConfig":
        optimizer_data = dict(data["optimizer"])
        optimizer_data.setdefault("link_sample_alphas", [0.25, 0.5, 0.75])
        optimizer_data.setdefault("segment_collision_alphas", [0.25, 0.5, 0.75])
        optimizer_data.setdefault("min_trust_region_radius", 0.01)
        optimizer_data.setdefault("trust_shrink_ratio", 0.5)
        optimizer_data.setdefault("trust_expand_ratio", 1.5)
        optimizer_data.setdefault("improve_ratio_threshold", 0.25)
        optimizer_data.setdefault("max_merit_coeff_increases", 4)
        optimizer_data.setdefault("merit_coeff_increase_ratio", 10.0)
        optimizer_data.setdefault("constraint_tolerance", 1.0e-3)
        optimizer_data.setdefault("min_approx_improve", 1.0e-5)
        optimizer_data.setdefault("min_approx_improve_frac", 1.0e-6)
        optimizer_data.setdefault("solver", "OSQP")

        return PlanarArmConfig(
            problem=PlanarArmProblemConfig(**data["problem"]),
            robot=PlanarArmRobotConfig(**data["robot"]),
            obstacle=ObstacleConfig(**data["obstacle"]),
            optimizer=PlanarArmOptimizerConfig(**optimizer_data),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def start_q(self) -> np.ndarray:
        return np.asarray(self.problem.start_q, dtype=float)

    @property
    def goal_q(self) -> np.ndarray:
        return np.asarray(self.problem.goal_q, dtype=float)

    @property
    def num_waypoints(self) -> int:
        return self.problem.num_waypoints
