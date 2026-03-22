import math

import numpy as np
import pytest

from src.features import (
    GOAL_CENTER_Y,
    GOAL_LINE_X,
    POST_Y_LEFT,
    POST_Y_RIGHT,
    defenders_in_shot_cone,
    point_in_triangle,
    shot_angle_radians,
    shot_distance,
)


def test_shot_distance_goal_center_zero():
    d = shot_distance(GOAL_LINE_X, GOAL_CENTER_Y)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_shot_distance_symmetry():
    d1 = shot_distance(100.0, 40.0)
    d2 = shot_distance(100.0, 40.0)
    assert d1 == d2
    assert d1 > 0


def test_shot_angle_positive():
    ang = shot_angle_radians(95.0, 40.0)
    assert ang > 0
    assert ang < math.pi


def test_point_in_triangle_center_of_goal_mouth():
    a = (100.0, 40.0)
    b = (GOAL_LINE_X, POST_Y_LEFT)
    c = (GOAL_LINE_X, POST_Y_RIGHT)
    mid = (GOAL_LINE_X - 0.5, GOAL_CENTER_Y)
    assert point_in_triangle(mid, a, b, c) is True


def test_defenders_in_shot_cone_counts_opponents():
    ff = [
        {"teammate": True, "location": [110.0, 40.0]},
        {"teammate": False, "location": [118.0, 40.0]},
        {"teammate": False, "location": [50.0, 10.0]},
    ]
    n = defenders_in_shot_cone(100.0, 40.0, ff)
    assert n >= 1


def test_post_constants_align_with_arrays():
    assert POST_Y_LEFT < POST_Y_RIGHT
    assert GOAL_LINE_X == 120.0
