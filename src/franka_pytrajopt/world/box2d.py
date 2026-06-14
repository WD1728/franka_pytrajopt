from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as patches
import numpy as np


@dataclass(frozen=True)
class AxisAlignedBox2D:
    center: list[float]
    size: list[float]

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)

    @property
    def size_array(self) -> np.ndarray:
        return np.asarray(self.size, dtype=float)

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * self.size_array

    @property
    def min_corner(self) -> np.ndarray:
        return self.center_array - self.half_extents

    @property
    def max_corner(self) -> np.ndarray:
        return self.center_array + self.half_extents

    def signed_distance(self, point: np.ndarray) -> float:
        sdf, _ = self.signed_distance_with_gradient(point)
        return float(sdf)

    def signed_distance_with_gradient(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        point = np.asarray(point, dtype=float)
        q = point - self.center_array
        a = np.abs(q) - self.half_extents

        outside_vec = np.maximum(a, 0.0)
        outside_norm = float(np.linalg.norm(outside_vec))
        inside_term = float(min(max(a[0], a[1]), 0.0))
        sdf = outside_norm + inside_term

        if outside_norm > 1.0e-12:
            grad = np.zeros(2, dtype=float)
            for idx in range(2):
                if a[idx] > 0.0:
                    grad[idx] = np.sign(q[idx]) * outside_vec[idx] / outside_norm
            return float(sdf), grad

        axis = int(np.argmax(a))
        grad = np.zeros(2, dtype=float)
        sign = np.sign(q[axis])
        grad[axis] = sign if sign != 0.0 else 1.0
        return float(sdf), grad

    def plot(self, ax, **kwargs) -> None:
        min_corner = self.min_corner
        rect = patches.Rectangle(
            (min_corner[0], min_corner[1]),
            self.size_array[0],
            self.size_array[1],
            **kwargs,
        )
        ax.add_patch(rect)
