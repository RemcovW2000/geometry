import st

from geometry.mesh import Node
from geometry import Orientation, Point, Position, Vector


@pytest.mark.parametrize(
    "node, position, expected_point",
    [
        (
            Node(1, 0, 0),
            Position(
                origin=Point(0, 0, 0),
                orientation=Orientation(
                    Vector(1, 0, 0),
                    Vector(0, 1, 0),
                    Vector(0, 0, 1),
                ),
            ),
            Point(1, 0, 0),
        ),
        (
            Node(0, 1, 0),
            Position(
                origin=Point(1, 1, 1),
                orientation=Orientation(
                    Vector(0, -1, 0),
                    Vector(1, 0, 0),
                    Vector(0, 0, 1),
                ),
            ),
            Point(2.0, 1.0, 1.0),
        ),
        (
            Node(0, 0, 1),
            Position(
                origin=Point(0, 0, 0),
                orientation=Orientation(
                    Vector(0, 0, 1),
                    Vector(1, 0, 0),
                    Vector(0, 1, 0),
                ),
            ),
            Point(0, 1, 0),
        ),
    ],
)
def test_node_in_position(node: Node, position: Position, expected_point: Point) -> None:
    new_node = position.point_in_global(node)
    assert new_node.is_at_point(expected_point)
