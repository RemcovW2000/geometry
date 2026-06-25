import numpy as np
import pytest

from geometry import Vector


def test_vector_cross() -> None:
    v1 = Vector(x=1, y=0, z=0)
    v2 = Vector(x=1, y=1, z=0)
    expected_cross = Vector(x=0, y=0, z=1)

    cross = v1.cross(v2)
    assert (cross.x, cross.y, cross.z) == (
        expected_cross.x,
        expected_cross.y,
        expected_cross.z,
    ), "Vector cross product is incorrect."


@pytest.mark.parametrize(
    "vector, angle, axis, expected_normalized",
    [
        (Vector(1, 0, 0), np.pi / 2, Vector(0, 0, 1), Vector(0, 1, 0)),
        (Vector(1, 0, 0), np.pi / 2, Vector(0, 1, 0), Vector(0, 0, -1)),
    ],
)
def test_vector_rotate(
    vector: Vector, angle: float, axis: Vector, expected_normalized: Vector
) -> None:
    rotated_v = vector.rotate(angle, axis)
    print(rotated_v)

    assert abs(rotated_v.x - expected_normalized.x) < 1e-6
    assert abs(rotated_v.y - expected_normalized.y) < 1e-6
    assert abs(rotated_v.z - expected_normalized.z) < 1e-6
