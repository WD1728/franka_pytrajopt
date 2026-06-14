from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_pytrajopt.logging.logger import ResultLogger
from franka_pytrajopt.trajopt.optimizer import TrajOptOptimizer
from franka_pytrajopt.trajopt.problem import Point2DConfig
from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def load_config(path: Path) -> Point2DConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Point2DConfig.from_dict(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2D point-robot TrajOpt demo.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--output", type=Path, required=True, help="Output result directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    obstacle = AxisAlignedBox2D(center=config.obstacle.center, size=config.obstacle.size)

    logger = ResultLogger(args.output)
    logger.save_config(config.to_dict())

    optimizer = TrajOptOptimizer(config=config, obstacle=obstacle, logger=logger)
    result = optimizer.optimize()

    summary = {
        "status": result.status,
        "iterations": result.iterations,
        "final_objective": result.final_objective,
        "final_min_signed_distance": result.final_min_signed_distance,
        "final_max_penetration_depth": result.final_max_penetration_depth,
        "final_smoothness_cost": result.final_smoothness_cost,
        "final_trajectory_csv": str(result.final_trajectory_csv),
    }
    logger.save_summary(summary)

    config_copy = args.output / "config_used.yaml"
    shutil.copyfile(args.config, config_copy)

    print(json.dumps(summary, indent=2))
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
