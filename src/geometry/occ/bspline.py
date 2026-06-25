"""Convert an evaluable surface into a B-spline CAD face (gmsh OCC).

The bridge has two parts:

1. :func:`surface_to_bspline_data` (pure numpy/scipy, no gmsh): samples a
   surface on a parameter grid and builds a tensor-product B-spline **control
   net** that *interpolates* those samples, via two passes of 1-D spline
   interpolation (``scipy.interpolate.make_interp_spline``, whose coefficients
   are exactly the B-spline control points). The resulting surface passes
   through the sampled points to interpolation accuracy.

2. :func:`add_bspline_surface` (needs gmsh): feeds that control net, with the
   interpolation's own knot vectors and degrees, to
   ``gmsh.model.occ.addBSplineSurface`` to create a real OCCT B-spline face.

:class:`BSplineSurfaceData` can also evaluate itself (scipy tensor product), so
the interpolation can be validated without gmsh present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from geometry.primitives import Point


class _SurfaceLike(Protocol):
    def point_at_parameter(self, u: float, v: float) -> Point: ...


@dataclass
class BSplineSurfaceData:
    """A tensor-product B-spline surface in a CAD-kernel-friendly form.

    Attributes:
        control_points: (n_u, n_v, 3) control net.
        degree_u, degree_v: spline degrees.
        knots_u, knots_v: *distinct* knot values (strictly increasing).
        mult_u, mult_v: multiplicities of those distinct knots.
    """

    control_points: np.ndarray
    degree_u: int
    degree_v: int
    knots_u: list[float]
    mult_u: list[int]
    knots_v: list[float]
    mult_v: list[int]

    def _full_knots(self, axis: str) -> np.ndarray:
        if axis == "u":
            return np.repeat(self.knots_u, self.mult_u)
        return np.repeat(self.knots_v, self.mult_v)

    def evaluate(self, u: float, v: float) -> np.ndarray:
        """Evaluate the surface at (u, v) via a scipy tensor product (no gmsh)."""
        tu = self._full_knots("u")
        tv = self._full_knots("v")
        # Interpolate the control net along u -> control points of the v-curve.
        row = BSpline(tu, self.control_points, self.degree_u, axis=0)(u)  # (n_v, 3)
        return np.asarray(BSpline(tv, row, self.degree_v, axis=0)(v), dtype=float)


def _distinct_knots(t: np.ndarray) -> tuple[list[float], list[int]]:
    """Collapse a full knot vector into (distinct values, multiplicities)."""
    values, counts = np.unique(t, return_counts=True)
    return values.tolist(), counts.astype(int).tolist()


def surface_to_bspline_data(
    surface: _SurfaceLike,
    us: list[float],
    vs: list[float],
    degree_u: int = 3,
    degree_v: int = 3,
) -> BSplineSurfaceData:
    """Sample ``surface`` on the (us, vs) grid and build an interpolating B-spline.

    Args:
        surface: anything with ``point_at_parameter(u, v) -> Point``.
        us, vs: strictly increasing parameter stations in [0, 1]; these become
            the interpolation nodes (and the sample count per direction).
        degree_u, degree_v: requested degrees (clamped to len-1).

    Returns:
        A :class:`BSplineSurfaceData` that interpolates the sampled grid.
    """
    us = list(map(float, us))
    vs = list(map(float, vs))
    n_u, n_v = len(us), len(vs)
    ku = min(degree_u, n_u - 1)
    kv = min(degree_v, n_v - 1)

    # Sampled grid G[i, j] = surface(us[i], vs[j]).
    grid = np.empty((n_u, n_v, 3), dtype=float)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            grid[i, j] = surface.point_at_parameter(u, v).as_array()

    # Pass 1: interpolate each v-row across u -> intermediate control points.
    spl_u0 = make_interp_spline(us, grid[:, 0, :], k=ku, axis=0)
    knots_u = spl_u0.t
    n_cu = spl_u0.c.shape[0]
    inter = np.empty((n_cu, n_v, 3), dtype=float)
    inter[:, 0, :] = spl_u0.c
    for j in range(1, n_v):
        inter[:, j, :] = make_interp_spline(us, grid[:, j, :], k=ku, axis=0).c

    # Pass 2: interpolate each u-control-row across v -> final control net.
    spl_v0 = make_interp_spline(vs, inter[0, :, :], k=kv, axis=0)
    knots_v = spl_v0.t
    n_cv = spl_v0.c.shape[0]
    ctrl = np.empty((n_cu, n_cv, 3), dtype=float)
    ctrl[0, :, :] = spl_v0.c
    for a in range(1, n_cu):
        ctrl[a, :, :] = make_interp_spline(vs, inter[a, :, :], k=kv, axis=0).c

    ku_vals, ku_mult = _distinct_knots(knots_u)
    kv_vals, kv_mult = _distinct_knots(knots_v)
    return BSplineSurfaceData(
        control_points=ctrl,
        degree_u=ku,
        degree_v=kv,
        knots_u=ku_vals,
        mult_u=ku_mult,
        knots_v=kv_vals,
        mult_v=kv_mult,
    )


def add_bspline_surface(gmsh, data: BSplineSurfaceData) -> int:
    """Create an OCCT B-spline face from ``data`` and return its surface tag.

    Must be called inside an active :class:`~geometry.occ.GmshSession`.

    Control points are emitted v-major (u index fastest) with
    ``numPointsU = n_u``; if a surface ever comes out transposed, that ordering
    is the thing to flip (the smoke-test example checks it).
    """
    occ = gmsh.model.occ
    n_u, n_v, _ = data.control_points.shape

    point_tags: list[int] = []
    for j in range(n_v):
        for i in range(n_u):
            x, y, z = data.control_points[i, j]
            point_tags.append(occ.addPoint(float(x), float(y), float(z)))

    return occ.addBSplineSurface(
        point_tags,
        n_u,
        degreeU=data.degree_u,
        degreeV=data.degree_v,
        knotsU=data.knots_u,
        multiplicitiesU=data.mult_u,
        knotsV=data.knots_v,
        multiplicitiesV=data.mult_v,
    )


def add_surface(gmsh, surface: _SurfaceLike, us: list[float], vs: list[float],
                degree_u: int = 3, degree_v: int = 3) -> int:
    """Convenience: sample, interpolate, and add a surface in one call."""
    data = surface_to_bspline_data(surface, us, vs, degree_u, degree_v)
    return add_bspline_surface(gmsh, data)
