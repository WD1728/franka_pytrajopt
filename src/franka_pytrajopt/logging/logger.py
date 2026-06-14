from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml


class ResultLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.iteration_dir = self.output_dir / "iterations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"

        if self.metrics_path.exists():
            self.metrics_path.unlink()
        for old_iteration_csv in self.iteration_dir.glob("iter_*.csv"):
            old_iteration_csv.unlink()

        self._metrics_header_written = False

    def save_config(self, config: dict) -> None:
        path = self.output_dir / "config_snapshot.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def save_summary(self, summary: dict) -> None:
        path = self.output_dir / "summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def iteration_csv_path(self, iteration: int) -> Path:
        return self.iteration_dir / f"iter_{iteration:03d}.csv"

    def log_iteration(
        self,
        iteration: int,
        trajectory: np.ndarray,
        metrics: dict[str, float],
        step_norm: float,
        subproblem_status: str,
        trust_region_radius: float,
        penalty_coeff: float,
        approx_merit: float,
        exact_merit: float,
        approx_improve: float,
        exact_improve: float,
        merit_improve_ratio: float,
        accepted: bool,
        penalty_iteration: int,
    ) -> None:
        self._write_trajectory(iteration, trajectory)
        self._append_metrics(
            iteration,
            metrics,
            step_norm,
            subproblem_status,
            trust_region_radius,
            penalty_coeff,
            approx_merit,
            exact_merit,
            approx_improve,
            exact_improve,
            merit_improve_ratio,
            accepted,
            penalty_iteration,
        )

    def _write_trajectory(self, iteration: int, trajectory: np.ndarray) -> None:
        path = self.iteration_csv_path(iteration)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["waypoint", "x", "y"])
            for idx, point in enumerate(trajectory):
                writer.writerow([idx, float(point[0]), float(point[1])])

    def _append_metrics(
        self,
        iteration: int,
        metrics: dict[str, float],
        step_norm: float,
        subproblem_status: str,
        trust_region_radius: float,
        penalty_coeff: float,
        approx_merit: float,
        exact_merit: float,
        approx_improve: float,
        exact_improve: float,
        merit_improve_ratio: float,
        accepted: bool,
        penalty_iteration: int,
    ) -> None:
        row = {
            "iteration": iteration,
            "penalty_iteration": penalty_iteration,
            "objective": metrics["objective"],
            "approx_merit": approx_merit,
            "exact_merit": exact_merit,
            "approx_improve": approx_improve,
            "exact_improve": exact_improve,
            "merit_improve_ratio": merit_improve_ratio,
            "smoothness_cost": metrics["smoothness_cost"],
            "min_signed_distance": metrics["min_signed_distance"],
            "max_penetration_depth": metrics["max_penetration_depth"],
            "collision_penalty": metrics["collision_penalty"],
            "constraint_violation": metrics["constraint_violation"],
            "step_norm": step_norm,
            "trust_region_radius": trust_region_radius,
            "penalty_coeff": penalty_coeff,
            "accepted": int(accepted),
            "subproblem_status": subproblem_status,
        }

        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not self._metrics_header_written:
                writer.writeheader()
                self._metrics_header_written = True
            writer.writerow(row)
