"""
Basic geometry primitives.

Defines the very basic primitives useful for geometric operations. These include:
- Point
- Vector
- Orientation
"""
from __future__ import annotations

import numpy as np

from geometry.errors import ConstructionError


class Point:
    """Point in 3D space."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def as_array(self) -> np.ndarray:
        """Convert this point to a numpy array."""
        return np.array([self.x, self.y, self.z], float)

    def is_at_point(self, other: Point, tol: float = 1e-6) -> bool:
        """Return True if this point at most a distance of tol to another point."""
        return (
            abs(self.x - other.x) < tol
            and abs(self.y - other.y) < tol
            and abs(self.z - other.z) < tol
        )

    def distance_to(self, other: Point) -> float:
        """Return the distance between this point and another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


class Vector:
    """Vector in 3D space."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.norm = self.calculate_norm()

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

    def calculate_norm(self) -> float:
        """Return the norm (magnitude) of the vector."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other: Vector) -> float:
        """Return the dot product between this vector and another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def is_orthogonal_to(self, other: Vector, tol: float = 1e-6) -> bool:
        """Return True if this vector is orthogonal to another vector within a tolerance."""
        return abs(self.dot(other)) < tol

    def cross(self, other: Vector) -> Vector:
        """Return the cross product between this vector and another vector."""
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x
        return Vector(cx, cy, cz)

    def normalize(self) -> Vector:
        """Return the normalized vector (with norm 1)."""
        n = self.norm
        if n == 0:
            raise ConstructionError("Cannot normalize a zero vector.")
        return Vector(self.x / n, self.y / n, self.z / n)

    def rotate(self, angle: float, axis: Vector) -> Vector:
        """Return this vector rotated by ``angle`` radians about ``axis``."""
        unit_axis = axis.normalize()
        v = np.array([self.x, self.y, self.z], float)
        k = np.array([unit_axis.x, unit_axis.y, unit_axis.z], float)

        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        rotated = v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1.0 - cos_t)
        return Vector(*rotated)

    def is_equal_to(self, other: Vector, tol: float = 1e-6) -> bool:
        """Return True if all components are equal within tol."""
        return (
            abs(self.x - other.x) < tol
            and abs(self.y - other.y) < tol
            and abs(self.z - other.z) < tol
        )

    def as_array(self) -> np.ndarray:
        """Return this vector as a numpy array."""
        return np.array([self.x, self.y, self.z], float)


class Orientation:
    """
    Orientation in 3D space.

    Defines x, y, and z axes. Axes must be orthogonal.
    """

    def __init__(self, vx: Vector, vy: Vector, vz: Vector):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.orthogonality_tolerance = 1e-6

        if not self.is_normalized():
            raise ConstructionError("Orientation vectors must be normalized.")
        if not self.is_orthogonal():
            raise ConstructionError("Orientation vectors must be orthogonal.")
        if not self.is_right_handed():
            raise ConstructionError("Orientation vectors must form a right-handed system.")

    def is_normalized(self, tol: float = 1e-6) -> bool:
        """Return True if this orientation's 3 vectors are normalized within tol."""
        return (
            abs(self.vx.norm - 1.0) < tol
            and abs(self.vy.norm - 1.0) < tol
            and abs(self.vz.norm - 1.0) < tol
        )

    def is_orthogonal(self) -> bool:
        """Return True if this orientation is orthogonal."""
        return (
            self.vx.is_orthogonal_to(self.vy, self.orthogonality_tolerance)
            and self.vy.is_orthogonal_to(self.vz, self.orthogonality_tolerance)
            and self.vz.is_orthogonal_to(self.vx, self.orthogonality_tolerance)
        )

    def is_right_handed(self, tol: float = 1e-6) -> bool:
        """Return True if this orientation is right handed."""
        cross = self.vx.cross(self.vy)
        return abs(cross.dot(self.vz) - 1.0) < tol

    def as_matrix(self) -> np.ndarray:
        """Return a 3×3 matrix with columns [ex ey ez] (local→global)."""
        return np.array(
            [
                [self.vx.x, self.vy.x, self.vz.x],
                [self.vx.y, self.vy.y, self.vz.y],
                [self.vx.z, self.vy.z, self.vz.z],
            ],
            dtype=float,
        )

    def local_to_global(self, local_vector: Vector) -> Vector:
        """Rotate a vector from local to global coordinates."""
        R = self.as_matrix()
        global_vec = R @ local_vector.as_array()
        return Vector(float(global_vec[0]), float(global_vec[1]), float(global_vec[2]))

    def global_to_local(self, global_vector: Vector) -> Vector:
        """Rotate a vector from global to local coordinates."""
        R = self.as_matrix().T
        local_vec = R @ global_vector.as_array()
        return Vector(float(local_vec[0]), float(local_vec[1]), float(local_vec[2]))

    def rotate(self, angle: float, axis: Vector) -> Orientation:
        """Return a new Orientation rotated by ``angle`` radians about ``axis``."""
        return Orientation(
            vx=self.vx.rotate(angle, axis),
            vy=self.vy.rotate(angle, axis),
            vz=self.vz.rotate(angle, axis),
        )


class Position:
    """
    Position in 3D space.

    Defined by an origin as well as an orientation.
    """

    def __init__(self, origin: Point, orientation: Orientation):
        self.origin = origin
        self.orientation = orientation

    def __repr__(self) -> str:
        return f"Position(origin={self.origin}, orientation={self.orientation})"

    def point_in_global(self, local_point: Point) -> Point:
        """Convert a point from local to global coordinates."""
        local_vec = Vector(local_point.x, local_point.y, local_point.z)
        global_vec = self.orientation.local_to_global(local_vec).as_array()
        global_pt = global_vec + self.origin.as_array()
        return Point(float(global_pt[0]), float(global_pt[1]), float(global_pt[2]))


class Line:
    """Represents an infinite line in 3d space."""

    def __init__(self, point: Point, direction: Vector):
        if direction.norm == 0:
            raise ConstructionError("Direction vector cannot be zero.")
        self.point = point
        self.direction = direction.normalize()

    def __repr__(self) -> str:
        return f"Line(point={self.point}, direction={self.direction})"

    def point_at_parameter(self, t: float) -> Point:
        """Return the point on the line at parameter t."""
        x = self.point.x + t * self.direction.x
        y = self.point.y + t * self.direction.y
        z = self.point.z + t * self.direction.z
        return Point(x, y, z)

    def plane_intersection(self, plane: Plane, tol: float = 1e-12) -> Point:
        """Return the unique intersection of this line with a plane.

        Line is: p(t) = p0 + t * d
        Solve: n · (p(t) - p_plane) = 0

        Raises:
            ConstructionError: when line is parallel to the plane (no intersection)
                or lies in the plane (infinitely many intersections).
        """
        denom = plane.normal.dot(self.direction)
        if abs(denom) < tol:
            if plane.is_point_on_plane(self.point, tol=tol):
                raise ConstructionError("Line lies in the plane; infinite intersections.")
            raise ConstructionError("Line is parallel to the plane; no intersection.")

        t = -plane.signed_distance(self.point) / denom
        return self.point_at_parameter(t)


class Plane:
    """Represents a plane in 3d space."""

    def __init__(self, point: Point, normal: Vector):
        if normal.norm == 0:
            raise ConstructionError("Normal vector cannot be zero.")
        self.point = point
        self.normal = normal.normalize()

    def __repr__(self) -> str:
        return f"Plane(point={self.point}, normal={self.normal})"

    def signed_distance(self, test_point: Point) -> float:
        """Signed distance from point to plane.

        Since `self.normal` is normalized, this equals the standard signed
        distance: n · (p - p0).
        """
        vec = Vector(
            test_point.x - self.point.x,
            test_point.y - self.point.y,
            test_point.z - self.point.z,
        )
        return vec.dot(self.normal)

    def is_point_on_plane(self, test_point: Point, tol: float = 1e-6) -> bool:
        """Check if a point lies on the plane within a tolerance."""
        return abs(self.signed_distance(test_point)) < tol
