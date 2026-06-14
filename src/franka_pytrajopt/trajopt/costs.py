from __future__ import annotations

import cvxpy as cp
import numpy as np


def smoothness_cost(trajectory: np.ndarray) -> float:
    deltas = trajectory[1:] - trajectory[:-1]
    return float(np.sum(deltas * deltas))


def smoothness_cost_expr(trajectory_var: cp.Expression) -> cp.Expression:
    deltas = trajectory_var[1:] - trajectory_var[:-1]
    return cp.sum_squares(deltas)
