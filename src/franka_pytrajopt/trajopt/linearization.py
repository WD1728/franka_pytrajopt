from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_pytrajopt.world.box2d import AxisAlignedBox2D


@dataclass(frozen=True)
class SignedDistanceLinearization:
    distances: np.ndarray
    gradients: np.ndarray


def linearize_signed_distance(
    trajectory: np.ndarray,
    obstacle: AxisAlignedBox2D,
) -> SignedDistanceLinearization:
    distances = []
    gradients = []
    for point in trajectory:
        sdf, grad = obstacle.signed_distance_with_gradient(point)
        distances.append(sdf)
        gradients.append(grad)

    return SignedDistanceLinearization(
        distances=np.asarray(distances, dtype=float),
        gradients=np.asarray(gradients, dtype=float),
    )
