"""Planarity checks for sets of geometric points."""
import numpy as np

from geometry.primitives import Point


def are_points_planar(points: list[Point], tol: float = 1e-6) -> bool:
    """Check if a set of points are coplanar."""
    if len(points) < 4:  # noqa: PLR2004
        return True  # Any three points are always coplanar

    p0 = points[0].as_array()
    p1 = points[1].as_array()
    p2 = points[2].as_array()

    # Define two vectors in the plane
    v1 = p1 - p0
    v2 = p2 - p0

    # Compute the normal vector to the plane
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Check the distance of each point from the plane
    for point in points[3:]:
        p = point.as_array()
        distance = np.dot(normal, p - p0)
        if abs(distance) > tol:
            return False

    return True
