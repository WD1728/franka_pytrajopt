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

    iteration_dir = result_dir / "iterations"
    selected = select_iteration_files(iteration_dir)
    trajectories = [(path.stem, load_trajectory(path)) for path in selected]

    fig, ax = plt.subplots(figsize=(8, 5))
    obstacle.plot(ax=ax, facecolor="lightcoral", edgecolor="darkred", alpha=0.4)

    seed_name, seed_traj = trajectories[0]
    ax.plot(seed_traj[:, 0], seed_traj[:, 1], "--", color="gray", linewidth=2, label=seed_name)

    for name, traj in trajectories[1:-1]:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.8, label=name)

    final_name, final_traj = trajectories[-1]
    ax.plot(final_traj[:, 0], final_traj[:, 1], "-o", color="tab:blue", linewidth=2, markersize=3, label=final_name)

    ax.scatter(seed_traj[0, 0], seed_traj[0, 1], color="green", s=60, label="start")
    ax.scatter(seed_traj[-1, 0], seed_traj[-1, 1], color="black", s=60, label="goal")

    ax.set_title("Point2D TrajOpt Result")
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
