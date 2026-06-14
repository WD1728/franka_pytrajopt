from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
(REPO_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_pytrajopt.trajopt.linearization import segment_sample_points
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a saved 2D point-robot TrajOpt result.")
    parser.add_argument("--result", type=Path, required=True, help="Result directory from the demo run.")
    return parser.parse_args()


def load_trajectory(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row["x"]), float(row["y"])])
    return np.asarray(rows, dtype=float)


def select_iteration_files(iter_dir: Path) -> list[Path]:
    files = sorted(iter_dir.glob("iter_*.csv"))
    if not files:
        raise FileNotFoundError(f"No iteration CSV files found in {iter_dir}")
    if len(files) <= 4:
        return files

    selected = [files[0], files[len(files) // 2], files[-1]]
    deduped = []
    seen = set()
    for path in selected:
        if path.name not in seen:
            deduped.append(path)
            seen.add(path.name)
    return deduped


def plot_result(result_dir: Path) -> Path:
    config_path = result_dir / "config_used.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    obstacle = AxisAlignedBox2D(
        center=config["obstacle"]["center"],
        size=config["obstacle"]["size"],
    )
    segment_alphas = [float(alpha) for alpha in config["optimizer"].get("segment_collision_alphas", [0.25, 0.5, 0.75])]

    iteration_dir = result_dir / "iterations"
    selected = select_iteration_files(iteration_dir)
    trajectories = [(path.stem, load_trajectory(path)) for path in selected]

    fig, ax = plt.subplots(figsize=(8, 5))
    obstacle.plot(ax=ax, facecolor="lightcoral", edgecolor="darkred", alpha=0.4)

    seed_name, seed_traj = trajectories[0]
    plot_trajectory(ax, obstacle, seed_traj, seed_name, base_color="gray", linestyle="--", sample_alphas=segment_alphas)

    for name, traj in trajectories[1:-1]:
        plot_trajectory(ax, obstacle, traj, name, base_color="tab:orange", linestyle="-", sample_alphas=segment_alphas, alpha=0.8)

    final_name, final_traj = trajectories[-1]
    plot_trajectory(ax, obstacle, final_traj, final_name, base_color="tab:blue", linestyle="-", sample_alphas=segment_alphas, draw_markers=True)
    plot_segment_samples(ax, obstacle, final_traj, segment_alphas)

    colliding_segments = count_colliding_segments(obstacle, final_traj)
    min_sampled_sdf = compute_min_sampled_sdf(obstacle, final_traj, segment_alphas)

    ax.scatter(seed_traj[0, 0], seed_traj[0, 1], color="green", s=60, label="start")
    ax.scatter(seed_traj[-1, 0], seed_traj[-1, 1], color="black", s=60, label="goal")

    ax.set_title(
        f"Point2D TrajOpt Result | colliding segments={colliding_segments} | min sampled sdf={min_sampled_sdf:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = result_dir / "plot.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def compute_min_sampled_sdf(
    obstacle: AxisAlignedBox2D,
    trajectory: np.ndarray,
    sample_alphas: list[float],
) -> float:
    distances = [obstacle.signed_distance(point) for point in trajectory]
    distances.extend(
        obstacle.signed_distance(point)
        for _, _, point in segment_sample_points(trajectory, sample_alphas)
    )
    return float(np.min(np.asarray(distances, dtype=float)))


def count_colliding_segments(obstacle: AxisAlignedBox2D, trajectory: np.ndarray) -> int:
    return sum(
        int(obstacle.segment_intersects(trajectory[idx], trajectory[idx + 1]))
        for idx in range(len(trajectory) - 1)
    )


def plot_segment_samples(
    ax,
    obstacle: AxisAlignedBox2D,
    trajectory: np.ndarray,
    sample_alphas: list[float],
) -> None:
    safe_points = []
    colliding_points = []
    for _, _, point in segment_sample_points(trajectory, sample_alphas):
        if obstacle.signed_distance(point) < 0.0:
            colliding_points.append(point)
        else:
            safe_points.append(point)

    if safe_points:
        safe_array = np.asarray(safe_points, dtype=float)
        ax.scatter(safe_array[:, 0], safe_array[:, 1], s=12, color="tab:cyan", alpha=0.7, label="segment samples")
    if colliding_points:
        colliding_array = np.asarray(colliding_points, dtype=float)
        ax.scatter(
            colliding_array[:, 0],
            colliding_array[:, 1],
            s=24,
            color="red",
            marker="x",
            linewidths=1.5,
            label="penetrating samples",
        )


def plot_trajectory(
    ax,
    obstacle: AxisAlignedBox2D,
    trajectory: np.ndarray,
    label: str,
    base_color: str,
    linestyle: str,
    sample_alphas: list[float],
    alpha: float = 1.0,
    draw_markers: bool = False,
) -> None:
    labelled_ok = False
    labelled_bad = False
    for idx in range(len(trajectory) - 1):
        start = trajectory[idx]
        end = trajectory[idx + 1]
        collides = obstacle.segment_intersects(start, end)
        color = "red" if collides else base_color
        linewidth = 3 if collides else 2
        segment_label = None
        if collides and not labelled_bad:
            segment_label = f"{label} colliding segment"
            labelled_bad = True
        elif not collides and not labelled_ok:
            segment_label = label
            labelled_ok = True

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=segment_label,
        )

    if draw_markers:
        ax.scatter(trajectory[:, 0], trajectory[:, 1], s=16, color=base_color, zorder=3)


def main() -> None:
    args = parse_args()
    output_path = plot_result(args.result)
    print(f"Plot written to: {output_path}")


if __name__ == "__main__":
    main()
