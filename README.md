# franka_pytrajopt

`franka_pytrajopt` is a small Python research repo for validating a TrajOpt-style sequential convex optimization core before adding PyBullet or Franka-specific models.

This milestone focuses on a 2D point robot moving around a single axis-aligned rectangular obstacle.

The implementation is intentionally simple:

- no copied Tesseract code
- no Tesseract wrapper yet
- no PRM or RRT
- no PyBullet
- no Franka model yet

## Run

```bash
python scripts/run_point2d_demo.py --config config/point2d_box.yaml --output results/point2d_box
python scripts/plot_point2d_result.py --result results/point2d_box
```

## Problem Setup

We optimize a 2D point-robot trajectory from a fixed start to a fixed goal in the presence of one axis-aligned rectangular obstacle.

The trajectory is represented by waypoints:

```text
P = [p_0, p_1, ..., p_N],   p_i in R^2
```

where:

- `p_0` is the fixed start
- `p_N` is the fixed goal
- the intermediate waypoints are optimized

The obstacle is an axis-aligned rectangle with center `c` and half extents `h`.

## Optimization Variables

The code optimizes all waypoints together and enforces the endpoints with equality constraints:

```text
p_0 = start
p_N = goal
```

This keeps the implementation small and matches the standard TrajOpt style of building a single trajectory optimization variable.

## Objective Function

The convex subproblem objective follows the TrajOpt pattern:

```text
min_P  w_smooth * sum_i ||p_{i+1} - p_i||^2
       + w_collision * sum_j hinge(margin - sdf(sample_j))
```

where:

- `w_smooth` is the smoothness weight
- `w_collision` is the collision penalty weight
- `hinge(z) = max(0, z)`
- `sample_j` runs over both waypoint samples and segment-interior samples

### Smoothness Term

The smoothness cost is:

```text
J_smooth(P) = sum_{i=0}^{N-1} ||p_{i+1} - p_i||_2^2
```

This penalizes large waypoint-to-waypoint motion and encourages short, smooth paths.

### Signed Distance

For a point `x`, define:

```text
q = x - c
a = |q| - h
```

Then the signed distance to the axis-aligned box is:

```text
sdf(x) = ||max(a, 0)||_2 + min(max(a_x, a_y), 0)
```

Interpretation:

- `sdf(x) > 0` means outside the box
- `sdf(x) = 0` means on the boundary
- `sdf(x) < 0` means inside the box

### Waypoint Collision Samples

Every waypoint contributes a collision hinge term:

```text
hinge(margin - sdf(p_i))
```

### Segment Collision Samples

Waypoint-only collision checking is not enough. A trajectory can have safe waypoint positions while the straight segment between them still tunnels through the obstacle.

To reduce this tunneling failure mode, each segment is sampled at fixed interpolation values:

```text
p_alpha = (1 - alpha) p_i + alpha p_{i+1}
```

with default:

```text
alpha in {0.25, 0.5, 0.75}
```

These sampled points are also included in the collision hinge penalty:

```text
hinge(margin - sdf(p_alpha))
```

So the collision objective includes:

- all waypoint samples
- all segment-interior sampled points

## Constraints

### Endpoint Constraints

```text
p_0 = start
p_N = goal
```

### Trust Region

Each convex subproblem is solved inside a box trust region around the current trajectory:

```text
|p_i - p_i_current| <= trust_box_size
```

This is implemented componentwise with `L_inf`-style bounds on the interior waypoints.

### Optional Max Axis Step Constraint

The config supports an optional `max_step`.

To keep OSQP compatibility, this is implemented as a componentwise bound:

```text
|p_{i+1,x} - p_{i,x}| <= max_step
|p_{i+1,y} - p_{i,y}| <= max_step
```

Important: this is **not** the Euclidean constraint

```text
||p_{i+1} - p_i||_2 <= max_step
```

It is an axis-aligned step-size bound.

## Sequential Convex Optimization

The signed distance function is non-convex as a trajectory objective, so we optimize it with a sequential convex optimization loop.

At each iteration, the SDF is linearized around the current trajectory:

```text
sdf(p) approx sdf(p_current) + grad_sdf(p_current)^T (p - p_current)
```

For a segment sample:

```text
p_alpha_current = (1 - alpha) p_i_current + alpha p_{i+1,current}
p_alpha_var     = (1 - alpha) p_i_var     + alpha p_{i+1,var}
```

and the affine approximation is:

```text
affine_sdf = sdf(p_alpha_current)
             + grad_sdf(p_alpha_current)^T (p_alpha_var - p_alpha_current)
```

The convex collision penalty then becomes:

```text
hinge(margin - affine_sdf)
```

This gives a convex quadratic-plus-hinge subproblem that OSQP can solve.

## Segment-Sampled Collision Checking

Waypoint-only checking can miss collisions like:

- waypoint A outside the box
- waypoint B outside the box
- line segment A -> B passing directly through the box

That is why the code includes segment interior samples:

```text
p_alpha = (1 - alpha) p_i + alpha p_{i+1}
```

This sampled formulation improves the optimizer significantly, but sampled validation is still not identical to an exact segment intersection test.

## Exact Final Validation

The repo now uses two different ideas during and after optimization:

### Sampled Signed-Distance Metrics

These are computed from:

- all waypoints
- all segment-interior sampled points

and reported in the summary as:

- `final_sampled_min_signed_distance`
- `final_sampled_max_penetration_depth`
- `final_sampled_collision_penalty`

### Exact Segment Intersection Validation

For the final path, we also check each line segment exactly with:

```text
AxisAlignedBox2D.segment_intersects(start, end)
```

This gives:

- `final_has_segment_collision`
- `final_colliding_segments`

The optimizer is not allowed to report `converged_constraint_tolerance` if any exact segment still intersects the box.

## Why The Default Seed Is Offset

The demo should start from an infeasible seed that penetrates the obstacle, because that is the whole point of validating the optimizer core.

However, a perfectly centered straight line through a symmetric rectangular obstacle can create an ambiguous or degenerate case:

- the geometry is symmetric
- the collision gradients can become locally uninformative or balanced
- the optimizer can get stuck in an artificial centerline local behavior

To make the experiment meaningful, the default seed is still infeasible but slightly asymmetric:

- `seed_type = offset_straight`
- `seed_y_offset = 0.1`

This produces a nearly straight path that still penetrates the obstacle, but does not pass perfectly through the box centerline.

The optimizer still has to solve the problem through collision costs, trust-region updates, and penalty logic. The default is **not** a pre-bent obstacle-avoiding detour.

## Seed Modes

Supported seed modes:

- `offset_straight`: default; nearly straight, infeasible, slightly asymmetric
- `centered_straight`: exactly straight through the obstacle if start-goal line intersects it
- `upper_detour`: obstacle-avoiding curved seed above the box
- `lower_detour`: obstacle-avoiding curved seed below the box

The default should stay `offset_straight` for this demo.

## Pseudocode

```text
Initialize trajectory P from offset straight-line seed

for penalty_iteration in range(max_penalty_iterations):
    for convex_iteration in range(max_convex_iterations):
        collect waypoint samples
        collect segment samples

        evaluate exact SDF values and SDF gradients at current samples

        build convexified collision penalties:
            hinge(margin - affine_sdf(sample))

        solve convex subproblem with:
            smoothness cost
            waypoint collision hinge penalties
            segment collision hinge penalties
            endpoint constraints
            trust-region constraints
            optional max axis step constraints

        evaluate true sampled metrics on the candidate
        evaluate exact segment intersection on the candidate

        if candidate improves merit:
            accept candidate
            expand trust region
        else:
            reject candidate
            shrink trust region

        if sampled convergence is satisfied and exact segment validation passes:
            stop

    if sampled or exact collision remains:
        increase collision penalty

save iteration trajectories, metrics, summary, and plot
```

## Outputs

The demo writes:

- `iterations/iter_000.csv`, `iter_001.csv`, ...
- `metrics.csv`
- `summary.json`
- `config_snapshot.yaml`
- `config_used.yaml`
- `plot.png`

`metrics.csv` records sampled collision metrics and exact segment-collision indicators for every saved optimizer iteration.
