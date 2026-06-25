import numpy as np

from geometry import InterpolatedLine, Orientation, Point, Position, Vector


def test_curve_transform_translation_only() -> None:
    curve = InterpolatedLine(points=[Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0)])

    pos = Position(
        origin=Point(10, 0, 0),
        orientation=Orientation(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)),
    )

    moved = curve.transform(pos)

    for u, base in [(0.0, Point(0, 0, 0)), (0.5, Point(1, 0, 0)), (1.0, Point(2, 0, 0))]:
        p = moved.point_at_parameter(u)
        assert p.is_at_point(Point(base.x + 10.0, base.y, base.z))


def test_curve_transform_rotation_z_90() -> None:
    curve = InterpolatedLine(points=[Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0)])

    # Rotate local x-axis to global y-axis (90 deg about +z)
    pos = Position(
        origin=Point(0, 0, 0),
        orientation=Orientation(Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, 0, 1)),
    )

    rotated = curve.transform(pos)

    # x-axis line should become y-axis line
    p0 = rotated.point_at_parameter(0.0)
    p1 = rotated.point_at_parameter(0.5)
    p2 = rotated.point_at_parameter(1.0)

    assert p0.is_at_point(Point(0, 0, 0))
    assert p1.is_at_point(Point(0, 1, 0), tol=1e-5)
    assert p2.is_at_point(Point(0, 2, 0), tol=1e-5)

    # sanity: points should remain collinear in y
    ys = np.array([p0.y, p1.y, p2.y])
    assert np.all(np.diff(ys) > 0)
