from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarArm2D:
    link_lengths: list[float]
    link_radius: float
    base: list[float]

    @property
    def base_array(self) -> np.ndarray:
        return np.asarray(self.base, dtype=float)

    @property
    def link_lengths_array(self) -> np.ndarray:
        lengths = np.asarray(self.link_lengths, dtype=float)
        if lengths.shape != (2,):
            raise ValueError("PlanarArm2D currently expects exactly two link lengths.")
        return lengths

    def forward_kinematics(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (2,):
            raise ValueError("PlanarArm2D expects q with shape (2,).")

        l1, l2 = self.link_lengths_array
        theta1, theta2 = q
        base = self.base_array
        elbow = base + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)], dtype=float)
        wrist = elbow + np.array(
            [l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)],
            dtype=float,
        )
        return {
            "base": base,
            "elbow": elbow,
            "wrist": wrist,
        }

    def link_segments(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        fk = self.forward_kinematics(q)
        return [
            (fk["base"], fk["elbow"]),
            (fk["elbow"], fk["wrist"]),
        ]

    def point_on_link(self, q: np.ndarray, link_index: int, local_alpha: float) -> np.ndarray:
        segments = self.link_segments(q)
        start, end = segments[link_index]
        return (1.0 - local_alpha) * start + local_alpha * end

    def point_jacobian(self, q: np.ndarray, link_index: int, local_alpha: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        theta1, theta2 = q
        l1, l2 = self.link_lengths_array
        alpha = float(local_alpha)

        if link_index == 0:
            point = self.base_array + alpha * np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)], dtype=float)
            del point  # computed only to mirror the geometry derivation
            jac = np.array(
                [
                    [-alpha * l1 * np.sin(theta1), 0.0],
                    [alpha * l1 * np.cos(theta1), 0.0],
                ],
                dtype=float,
            )
            return jac

        if link_index == 1:
            jac_theta1 = np.array(
                [
                    -l1 * np.sin(theta1) - alpha * l2 * np.sin(theta1 + theta2),
                    l1 * np.cos(theta1) + alpha * l2 * np.cos(theta1 + theta2),
                ],
                dtype=float,
            )
            jac_theta2 = np.array(
                [
                    -alpha * l2 * np.sin(theta1 + theta2),
                    alpha * l2 * np.cos(theta1 + theta2),
                ],
                dtype=float,
            )
            return np.column_stack((jac_theta1, jac_theta2))

        raise ValueError(f"Unsupported link_index: {link_index}")
