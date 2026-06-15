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

from franka_pytrajopt.robot.planar_arm import PlanarArm2D
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a planar-arm TrajOpt result.")
    parser.add_argument("--result", type=Path, required=True, help="Result directory from the planar arm demo run.")
    return parser.parse_args()


def load_trajectory(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row["q1"]), float(row["q2"])])
    return np.asarray(rows, dtype=float)


def select_iteration_files(iter_dir: Path) -> list[Path]:
    files = sorted(iter_dir.glob("iter_*.csv"))
    if not files:
        raise FileNotFoundError(f"No iteration CSV files found in {iter_dir}")
    if len(files) <= 4:
        return files
    return [files[0], files[len(files) // 2], files[-1]]


def pose_collision(arm: PlanarArm2D, obstacle: AxisAlignedBox2D, q: np.ndarray, inflation: float) -> bool:
    inflated_box = obstacle.inflated(inflation)
    return any(inflated_box.segment_intersects(start, end) for start, end in arm.link_segments(q))


def draw_pose(ax, arm: PlanarArm2D, obstacle: AxisAlignedBox2D, q: np.ndarray, color: str, inflation: float, alpha: float) -> None:
    colliding = pose_collision(arm, obstacle, q, inflation)
    for start, end in arm.link_segments(q):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="red" if colliding else color,
            linewidth=2.5 if colliding else 1.8,
            alpha=alpha,
        )


def end_effector_path(arm: PlanarArm2D, trajectory: np.ndarray) -> np.ndarray:
    return np.asarray([arm.forward_kinematics(q)["wrist"] for q in trajectory], dtype=float)


def plot_result(result_dir: Path) -> Path:
    with (result_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    arm = PlanarArm2D(
        link_lengths=config["robot"]["link_lengths"],
        link_radius=float(config["robot"]["link_radius"]),
        base=config["robot"]["base"],
    )
    obstacle = AxisAlignedBox2D(
        center=config["obstacle"]["center"],
        size=config["obstacle"]["size"],
    )
    inflation = float(config["optimizer"]["collision_margin"]) + float(config["robot"]["link_radius"])

    trajectories = [(path.stem, load_trajectory(path)) for path in select_iteration_files(result_dir / "iterations")]
    seed_name, seed_traj = trajectories[0]
    final_name, final_traj = trajectories[-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    obstacle.plot(ax=ax, facecolor="lightcoral", edgecolor="darkred", alpha=0.4)
    obstacle.inflated(inflation).plot(ax=ax, facecolor="none", edgecolor="darkred", linestyle="--", linewidth=1.2, alpha=0.7)

    seed_path = end_effector_path(arm, seed_traj)
    final_path = end_effector_path(arm, final_traj)
    ax.plot(seed_path[:, 0], seed_path[:, 1], "--", color="gray", linewidth=1.5, label=seed_name + " EE path")
    ax.plot(final_path[:, 0], final_path[:, 1], "-", color="tab:blue", linewidth=2.0, label=final_name + " EE path")

    for q in seed_traj[np.linspace(0, len(seed_traj) - 1, num=min(6, len(seed_traj)), dtype=int)]:
        draw_pose(ax, arm, obstacle, q, color="gray", inflation=inflation, alpha=0.25)

    for q in final_traj[np.linspace(0, len(final_traj) - 1, num=min(6, len(final_traj)), dtype=int)]:
        draw_pose(ax, arm, obstacle, q, color="tab:blue", inflation=inflation, alpha=0.8)

    ax.scatter(arm.base_array[0], arm.base_array[1], color="black", s=40, label="base")
    ax.set_title("Planar 2-Link Arm TrajOpt")
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


def main() -> None:
    args = parse_args()
    output_path = plot_result(args.result)
    print(f"Plot written to: {output_path}")


if __name__ == "__main__":
    main()
