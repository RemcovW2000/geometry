import numpy as np
import pytest

from geometry import Point


@pytest.mark.parametrize(
    "p1, p2, expected_distance",
    [
        (Point(0, 0, 0), Point(0, 0, 0), 0),
        (Point(0, 0, 0), Point(1, 0, 0), 1),
        (Point(0, 0, 0), Point(1, 1, 0), np.sqrt(2)),
        (Point(0, 0, 0), Point(1, 1, 1), np.sqrt(3)),
        (Point(1, 1, 1), Point(1, 1, 1), 0),
        (Point(1, 1, 1), Point(0, 0, 1), np.sqrt(2)),
    ],
)
def test_distance_to(p1: Point, p2: Point, expected_distance: float) -> None:
    assert p1.distance_to(p2) == expected_distance
