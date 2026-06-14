from __future__ import annotations

import cvxpy as cp
import numpy as np


def fixed_endpoint_constraints(
    trajectory_var: cp.Variable,
    start: np.ndarray,
    goal: np.ndarray,
) -> list[cp.Constraint]:
    return [
        trajectory_var[0] == start,
        trajectory_var[-1] == goal,
    ]


def trust_region_constraints(
    trajectory_var: cp.Variable,
    current_trajectory: np.ndarray,
    radius: float,
) -> list[cp.Constraint]:
    interior = trajectory_var[1:-1]
    current_interior = current_trajectory[1:-1]
    return [
        interior - current_interior <= radius,
        current_interior - interior <= radius,
    ]
