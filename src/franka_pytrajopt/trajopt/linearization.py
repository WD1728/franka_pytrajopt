from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_pytrajopt.world.box2d import AxisAlignedBox2D


@dataclass(frozen=True)
class CollisionSample:
    left_index: int
    right_index: int | None
    alpha: float | None
    point: np.ndarray
    distance: float
    gradient: np.ndarray

    @property
    def kind(self) -> str:
        return "waypoint" if self.right_index is None else "segment"


@dataclass(frozen=True)
class SignedDistanceLinearization:
    waypoint_samples: list[CollisionSample]
    segment_samples: list[CollisionSample]

    @property
    def all_samples(self) -> list[CollisionSample]:
        return [*self.waypoint_samples, *self.segment_samples]


def interpolate_segment_point(
    start: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(start, dtype=float) + alpha * np.asarray(end, dtype=float)


def segment_sample_points(
    trajectory: np.ndarray,
    alphas: list[float] | tuple[float, ...],
) -> list[tuple[int, float, np.ndarray]]:
    samples: list[tuple[int, float, np.ndarray]] = []
    for idx in range(len(trajectory) - 1):
        for alpha in alphas:
            point = interpolate_segment_point(trajectory[idx], trajectory[idx + 1], alpha)
            samples.append((idx, float(alpha), point))
    return samples


def linearize_signed_distance(
    trajectory: np.ndarray,
    obstacle: AxisAlignedBox2D,
    segment_alphas: list[float] | tuple[float, ...],
) -> SignedDistanceLinearization:
    waypoint_samples = []
    for idx, point in enumerate(trajectory):
        sdf, grad = obstacle.signed_distance_with_gradient(point)
        waypoint_samples.append(
            CollisionSample(
                left_index=idx,
                right_index=None,
                alpha=None,
                point=np.asarray(point, dtype=float),
                distance=float(sdf),
                gradient=np.asarray(grad, dtype=float),
            )
        )

    segment_samples = []
    for segment_index, alpha, point in segment_sample_points(trajectory, segment_alphas):
        sdf, grad = obstacle.signed_distance_with_gradient(point)
        segment_samples.append(
            CollisionSample(
                left_index=segment_index,
                right_index=segment_index + 1,
                alpha=float(alpha),
                point=np.asarray(point, dtype=float),
                distance=float(sdf),
                gradient=np.asarray(grad, dtype=float),
            )
        )

    return SignedDistanceLinearization(
        waypoint_samples=waypoint_samples,
        segment_samples=segment_samples,
    )
