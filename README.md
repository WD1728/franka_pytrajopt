# franka_pytrajopt

Milestone 1 is a minimal Python research repository that implements a TrajOpt-inspired optimizer core for a 2D point robot with a single axis-aligned rectangular obstacle.

This repo intentionally stays small:

- no copied Tesseract code
- no Tesseract wrapper yet
- no PRM or RRT
- no PyBullet
- no Franka model yet

The current formulation now performs collision evaluation at both waypoint positions and sampled points along every segment, which fixes the classic failure mode where all waypoints are outside the obstacle but the straight line between them still cuts through it.

## Run

```bash
python scripts/run_point2d_demo.py --config config/point2d_box.yaml --output results/point2d_box
python scripts/plot_point2d_result.py --result results/point2d_box
```

## Formulation

Let the trajectory be:

```text
X = [x_0, x_1, ..., x_{N-1}],  x_i in R^2
```

with fixed endpoints:

```text
x_0 = x_start
x_{N-1} = x_goal
```

The smoothness objective is:

```text
J_smooth(X) = sum_{i=0}^{N-2} ||x_{i+1} - x_i||_2^2
```

For an axis-aligned box with center `c` and half extents `h`, the signed distance is:

```text
q = x - c
a = |q| - h
sdf(x) = ||max(a, 0)||_2 + min(max(a_x, a_y), 0)
```

We use a hinge-style collision penalty with clearance margin `d_safe`:

```text
J_coll(X) = w_coll * sum_j max(0, d_safe - sdf(z_j))
```

where the collision sample set `z_j` includes:

- all waypoints `x_i`
- sampled segment points `z_{i,alpha} = (1 - alpha) x_i + alpha x_{i+1}`

with default segment alphas `[0.25, 0.5, 0.75]`.

At each SCP iteration we linearize the signed distance around the current trajectory:

```text
sdf(z_j) ~= sdf(z_j^k) + g_j^T (z_j - z_j^k)
```

and solve a convex subproblem with:

- smoothness cost
- linearized hinge penalties for waypoint samples
- linearized hinge penalties for segment samples
- fixed endpoint constraints
- a trust region around the current trajectory
- an optional per-segment max-step bound

The trust-region loop uses the same high-level TrajOpt pattern as the upstream implementation:

- solve a convexified subproblem
- compare approximate merit improvement to exact merit improvement
- accept or reject the step
- shrink or expand the trust region
- increase penalty weight if collision violation remains

## Outputs

The demo writes:

- `iterations/iter_000.csv`, `iter_001.csv`, ...
- `metrics.csv`
- `summary.json`
- `config_snapshot.yaml`
- `config_used.yaml`
- `plot.png`

Metrics are computed from the combined waypoint-plus-segment sample set, so `final_max_penetration_depth` reflects continuous sampled validation instead of waypoint-only validation.
