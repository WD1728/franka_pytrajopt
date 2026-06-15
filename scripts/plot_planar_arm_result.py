from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.lines import Line2D
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


def colliding_link_indices(arm: PlanarArm2D, obstacle: AxisAlignedBox2D, q: np.ndarray, inflation: float) -> list[int]:
    inflated_box = obstacle.inflated(inflation)
    return [
        idx
        for idx, (start, end) in enumerate(arm.link_segments(q))
        if inflated_box.segment_intersects(start, end)
    ]


def draw_pose(
    ax,
    arm: PlanarArm2D,
    obstacle: AxisAlignedBox2D,
    q: np.ndarray,
    color: str,
    inflation: float,
    alpha: float,
    linewidth: float,
    collision_linewidth: float,
) -> bool:
    colliding_indices = set(colliding_link_indices(arm, obstacle, q, inflation))
    any_collision = False
    for idx, (start, end) in enumerate(arm.link_segments(q)):
        is_colliding = idx in colliding_indices
        any_collision = any_collision or is_colliding
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="red" if is_colliding else color,
            linewidth=collision_linewidth if is_colliding else linewidth,
            alpha=alpha,
        )
    return any_collision


def end_effector_path(arm: PlanarArm2D, trajectory: np.ndarray) -> np.ndarray:
    return np.asarray([arm.forward_kinematics(q)["wrist"] for q in trajectory], dtype=float)


def sampled_pose_indices(num_waypoints: int, num_samples: int = 6) -> np.ndarray:
    return np.linspace(0, num_waypoints - 1, num=min(num_samples, num_waypoints), dtype=int)


def pose_workspace_points(arm: PlanarArm2D, q: np.ndarray) -> np.ndarray:
    fk = arm.forward_kinematics(q)
    return np.vstack([fk["base"], fk["elbow"], fk["wrist"]])


def axis_limits_from_trajectories(arm: PlanarArm2D, trajectories: list[np.ndarray], padding: float = 0.25) -> tuple[float, float, float, float]:
    all_points = []
    for trajectory in trajectories:
        for q in trajectory:
            all_points.append(pose_workspace_points(arm, q))
    stacked = np.vstack(all_points)
    x_min, y_min = np.min(stacked, axis=0)
    x_max, y_max = np.max(stacked, axis=0)
    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def add_panel_text(ax, lines: list[str]) -> None:
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )


def plot_panel(
    ax,
    arm: PlanarArm2D,
    obstacle: AxisAlignedBox2D,
    trajectory: np.ndarray,
    inflation: float,
    title: str,
    ee_path_color: str,
    ee_path_linestyle: str,
    ee_path_label: str,
    pose_color: str,
    pose_alpha: float,
    pose_linewidth: float,
) -> int:
    obstacle.plot(ax=ax, facecolor="lightcoral", edgecolor="darkred", alpha=0.4)
    obstacle.inflated(inflation).plot(
        ax=ax,
        facecolor="none",
        edgecolor="darkred",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    ee_path = end_effector_path(arm, trajectory)
    ax.plot(
        ee_path[:, 0],
        ee_path[:, 1],
        linestyle=ee_path_linestyle,
        color=ee_path_color,
        linewidth=3.0,
        marker="o",
        markersize=4.0,
        markerfacecolor=ee_path_color,
        markeredgecolor="white",
        label=ee_path_label,
    )

    colliding_pose_count = 0
    for idx in sampled_pose_indices(len(trajectory)):
        collided = draw_pose(
            ax=ax,
            arm=arm,
            obstacle=obstacle,
            q=trajectory[idx],
            color=pose_color,
            inflation=inflation,
            alpha=pose_alpha,
            linewidth=pose_linewidth,
            collision_linewidth=3.2,
        )
        colliding_pose_count += int(collided)

    ax.scatter(ee_path[0, 0], ee_path[0, 1], color="green", s=70, marker="o", label="start EE", zorder=4)
    ax.scatter(ee_path[-1, 0], ee_path[-1, 1], color="purple", s=110, marker="*", label="goal EE", zorder=4)
    ax.scatter(arm.base_array[0], arm.base_array[1], color="black", s=45, label="base", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    return colliding_pose_count


def add_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color="lightcoral", linewidth=8, alpha=0.5, label="obstacle"),
        Line2D([0], [0], color="darkred", linestyle="--", linewidth=1.5, label="inflated obstacle"),
        Line2D([0], [0], color="red", linewidth=3.0, label="colliding link"),
        Line2D([0], [0], color="black", linewidth=3.0, marker="o", markersize=5, label="EE path"),
        Line2D([0], [0], color="green", marker="o", linestyle="None", markersize=8, label="start EE"),
        Line2D([0], [0], color="purple", marker="*", linestyle="None", markersize=10, label="goal EE"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=6, label="base"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)


def plot_result(result_dir: Path) -> Path:
    with (result_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (result_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

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
    x_min, x_max, y_min, y_max = axis_limits_from_trajectories(arm, [seed_traj, final_traj])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    fig.suptitle("Planar 2-Link Arm TrajOpt: q-space collision optimization", fontsize=14)

    seed_colliding_pose_count = plot_panel(
        ax=axes[0],
        arm=arm,
        obstacle=obstacle,
        trajectory=seed_traj,
        inflation=inflation,
        title="Initial seed",
        ee_path_color="gray",
        ee_path_linestyle="--",
        ee_path_label=seed_name + " EE path",
        pose_color="lightgray",
        pose_alpha=0.55,
        pose_linewidth=1.2,
    )
    add_panel_text(axes[0], [f"Colliding sampled poses: {seed_colliding_pose_count}"])

    final_colliding_pose_count = plot_panel(
        ax=axes[1],
        arm=arm,
        obstacle=obstacle,
        trajectory=final_traj,
        inflation=inflation,
        title="Optimized final trajectory",
        ee_path_color="darkorange",
        ee_path_linestyle="-",
        ee_path_label=final_name + " EE path",
        pose_color="teal",
        pose_alpha=0.55,
        pose_linewidth=1.3,
    )
    final_validation_ok = not bool(summary["final_has_link_collision"])
    add_panel_text(
        axes[1],
        [
            f"Final validation passed: {'yes' if final_validation_ok else 'no'}",
            f"Colliding sampled poses: {final_colliding_pose_count}",
        ],
    )

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    add_legend(axes[1])

    output_path = result_dir / "plot.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = plot_result(args.result)
    print(f"Plot written to: {output_path}")


if __name__ == "__main__":
    main()
