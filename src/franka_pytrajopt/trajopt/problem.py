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
    solver: str = "OSQP"


@dataclass(frozen=True)
class ProblemConfig:
    num_waypoints: int
    dimensions: int
    start: list[float]
    goal: list[float]


@dataclass(frozen=True)
class Point2DConfig:
    problem: ProblemConfig
    obstacle: ObstacleConfig
    optimizer: OptimizerConfig

    @staticmethod
    def from_dict(data: dict) -> "Point2DConfig":
        return Point2DConfig(
            problem=ProblemConfig(**data["problem"]),
            obstacle=ObstacleConfig(**data["obstacle"]),
            optimizer=OptimizerConfig(**data["optimizer"]),
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
