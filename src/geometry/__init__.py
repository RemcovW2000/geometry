"""Geometry primitives, curves, surfaces, and pure-geometry meshing.

This package is standalone: it depends only on numpy/matplotlib and has no
knowledge of structural analysis, FEM, or composites. Other projects can
depend on it for geometry primitives without pulling in ``structures``.
"""
from geometry.curves import InterpolatedLine, IsoCurve, TrimmedCurve
from geometry.errors import (
    ConstructionError,
    GeometryError,
    NotPlanarError,
    ToleranceError,
)
from geometry.primitives import Line, Orientation, Plane, Point, Position, Vector
from geometry.surfaces import GordonPatch, GordonSurface, SewnSurface, Surface, TrimmedSurface

__all__ = [
    "Point",
    "Vector",
    "Orientation",
    "Position",
    "Line",
    "Plane",
    "InterpolatedLine",
    "IsoCurve",
    "TrimmedCurve",
    "Surface",
    "TrimmedSurface",
    "SewnSurface",
    "GordonPatch",
    "GordonSurface",
    "GeometryError",
    "ConstructionError",
    "NotPlanarError",
    "ToleranceError",
]
