import numpy as np
import pytest

from geometry import Orientation, Vector


@pytest.mark.parametrize(
    "global_vector, orientation, local_vector",
    [
        (
            Vector(1, 0, 0),
            Orientation(
                Vector(1, 0, 0),
                Vector(0, 1, 0),
                Vector(0, 0, 1),
            ),
            Vector(1, 0, 0),
        ),
        (
            Vector(0, 1, 0),
            Orientation(
                Vector(0, 1, 0),
                Vector(-1, 0, 0),
                Vector(0, 0, 1),
            ),
            Vector(1, 0, 0),
        ),
        (
            Vector(1, 1, 1),
            Orientation(
                Vector(1, 1, 0).normalize(),
                Vector(-1, 1, 0).normalize(),
                Vector(0, 0, 1).normalize(),
            ),
            Vector(np.sqrt(2), 0, 1),
        ),
    ],
)
def test_global_to_local(
    global_vector: Vector, orientation: Orientation, local_vector: Vector
) -> None:
    calculated_local_vector = orientation.global_to_local(global_vector)
    assert calculated_local_vector.is_equal_to(
        local_vector
    ), "Expected local vector does not match calculated local vector."

    calculated_global_vector = orientation.local_to_global(local_vector)
    assert calculated_global_vector.is_equal_to(
        global_vector
    ), "Expected global vector does not match calculated global vector."
