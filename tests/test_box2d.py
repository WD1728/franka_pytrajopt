import numpy as np

from franka_pytrajopt.world.box2d import AxisAlignedBox2D


def test_signed_distance_outside_box():
    box = AxisAlignedBox2D(center=[0.0, 0.0], size=[2.0, 4.0])
    point = np.array([3.0, 0.0], dtype=float)
    sdf = box.signed_distance(point)
    assert np.isclose(sdf, 2.0)


def test_signed_distance_on_boundary():
    box = AxisAlignedBox2D(center=[0.0, 0.0], size=[2.0, 4.0])
    point = np.array([1.0, 0.5], dtype=float)
    sdf = box.signed_distance(point)
    assert np.isclose(sdf, 0.0)


def test_signed_distance_inside_box():
    box = AxisAlignedBox2D(center=[0.0, 0.0], size=[2.0, 4.0])
    point = np.array([0.0, 0.0], dtype=float)
    sdf = box.signed_distance(point)
    assert np.isclose(sdf, -1.0)
