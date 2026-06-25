import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import brentq

from geometry.errors import ToleranceError
from geometry.primitives import Plane, Point, Position


class Curve:
    # SciPy spline representation: (t, c, k)
    tck: tuple

    def point_at_parameter(self, u: float) -> Point:
        x, y, z = splev(u, self.tck)
        return Point(x, y, z)

    def parameter_at_point(
        self, point: Point, tol: float = 1e-6, n_initial_samples: int = 20
    ) -> float:
        """Find the parameter value corresponding to a given point on the line.

        Uses a two-stage approach:

        1. Coarse sampling to locate the global closest-approach region and a
           bracket on the derivative of the squared distance.
        2. Root-finding on that derivative with ``brentq`` to pinpoint ``u``.

        The derivative-based formulation gives a proper distance-tolerance check
        and is robust to the curve having multiple local minima, provided the
        coarse grid is fine enough to isolate the global one.

        Args:
            point: The point to find on the curve.
            tol: Tolerance on the Euclidean distance between the curve at the
                returned ``u`` and the target point.
            n_initial_samples: Number of samples used for the coarse search.

        Returns:
            The parameter value ``u`` in [0, 1] where the curve is closest to the
            point.

        Raises:
            ToleranceError: If the closest point on the curve is farther from the
                target than ``tol``.
        """
        target = point.as_array()

        def curve(u: float) -> np.ndarray:
            return np.asarray(splev(u, self.tck))

        def distance(u: float) -> float:
            return float(np.linalg.norm(curve(u) - target))

        def d_dist_sq(u: float) -> float:
            # d/du ||C(u) - p||^2 = 2 (C(u) - p) . C'(u)
            diff = curve(u) - target
            deriv = np.asarray(splev(u, self.tck, der=1))
            return float(2.0 * np.dot(diff, deriv))

        # ------------------------------------------------------------------
        # Stage 1: coarse sampling on [0, 1]
        # ------------------------------------------------------------------
        u_samples = np.linspace(0.0, 1.0, n_initial_samples)
        distances = np.array([distance(u) for u in u_samples])
        best_idx = int(np.argmin(distances))

        # ------------------------------------------------------------------
        # Stage 2: refine
        # ------------------------------------------------------------------
        # Try to bracket a sign change of d_dist_sq across the best sample. If one
        # exists, brentq converges to the stationary point to machine precision.
        # Otherwise the minimum is on (or against) a boundary of [0, 1], in which
        # case the best sample itself is already the answer.
        u_opt = u_samples[best_idx]

        lo_idx = max(0, best_idx - 1)
        hi_idx = min(n_initial_samples - 1, best_idx + 1)
        u_lo, u_hi = u_samples[lo_idx], u_samples[hi_idx]

        g_lo = d_dist_sq(u_lo)
        g_hi = d_dist_sq(u_hi)

        if g_lo == 0.0:  # noqa plr2004
            u_opt = u_lo
        elif g_hi == 0.0:  # noqa plr2004
            u_opt = u_hi
        elif g_lo < 0.0 < g_hi:  # noqa plr2004
            # Sign change => interior minimum in [u_lo, u_hi].
            # xtol here is on u; rtol gives us the distance-scale precision.
            u_opt = float(brentq(d_dist_sq, u_lo, u_hi, xtol=1e-12, rtol=1e-12))
        # else: no interior minimum in this bracket; keep the coarse best.

        final_dist = distance(u_opt)
        if final_dist > tol:
            raise ToleranceError(
                f"Point {point} is not on the curve "
                f"(distance = {final_dist:.2e} > tol = {tol:.2e})"
            )

        return float(np.clip(u_opt, 0.0, 1.0))

    def plane_intersection_point(
        self,
        plane: Plane,
        u_min: float = 0.0,
        u_max: float = 1.0,
        n_samples: int = 100,
        tol: float = 1e-9,
        max_iter: int = 80,
    ) -> list[Point]:
        """Find curve-plane intersections by root finding on the curve parameter.

        We solve f(u) = n·(C(u) - p0) = 0 for u in [u_min, u_max].

        Algorithm:
          1) Sample f(u) on a grid of `n_samples`
          2) Detect sign changes (brackets)
          3) Refine each bracket with SciPy's `brentq` (robust bracketed solver)
          4) Deduplicate roots

        Returns:
            List of (u, Point) pairs.

        Notes:
            - Requires SciPy (used elsewhere in this repo for splines).
            - Bracketed solvers need a sign change; tangent contacts may be missed
              unless they land near a sample point.
        """
        if u_max <= u_min:
            raise ValueError("u_max must be > u_min")

        def f(u: float) -> float:
            return plane.signed_distance(self.point_at_parameter(u))

        us = np.linspace(u_min, u_max, n_samples, dtype=float)
        fs = np.empty(n_samples, dtype=float)
        for i in range(len(us)):
            u = float(us[i])
            fs[i] = f(u)

        brackets: list[tuple[float, float]] = []
        for i in range(n_samples - 1):
            u0 = float(us[i])
            u1 = float(us[i + 1])
            f0 = float(fs[i])
            f1 = float(fs[i + 1])

            # exact-ish hit at sample
            if abs(f0) <= tol:
                brackets.append((u0, u0))
                continue

            # sign change means there's a root in (u0, u1)
            if f0 * f1 < 0.0:  # noqa PLR2004
                brackets.append((u0, u1))

        roots: list[float] = []
        for a, b in brackets:
            if a == b:
                u_root = a
            else:
                u_root = float(brentq(f, a, b, xtol=tol, maxiter=max_iter))
            roots.append(u_root)

        roots.sort()
        return [self.point_at_parameter(u) for u in roots]

    def transform(self, position: Position) -> "Curve":
        """Return a rigidly transformed copy of this curve.

        This applies the given :class:`~geometry.primitives.Position` to the
        curve geometry:

        - rotation (via ``position.orientation``)
        - translation (via ``position.origin``)

        Implementation detail:
            SciPy splines store their geometry in ``tck``, where the *control points*
            are held in ``tck[1]`` (c). A rigid transform can be applied by transforming
            those control points directly while keeping the knot vector and degree
            unchanged.

        Returns:
            A new curve instance with an updated ``tck``.
        """
        t, c, k = self.tck

        # c is usually a list/array of 3 arrays [cx, cy, cz]
        cx, cy, cz = (
            np.asarray(c[0], dtype=float),
            np.asarray(c[1], dtype=float),
            np.asarray(c[2], dtype=float),
        )
        ctrl = np.vstack([cx, cy, cz])  # (3, n_ctrl)

        # Apply orientation (local->global rotation). Orientation matrix columns are basis in global.
        R = position.orientation.as_matrix()  # (3,3)
        rotated = R @ ctrl

        # Apply translation
        translated = rotated + position.origin.as_array().reshape(3, 1)

        new_c = [translated[0, :], translated[1, :], translated[2, :]]
        new_tck = (t, new_c, k)

        out = Curve.__new__(Curve)
        out.tck = new_tck
        return out


class InterpolatedLine(Curve):
    def __init__(self, points: list[Point], params: list[float] | None = None):
        """Interpolating spline through the given points.

        Args:
            points: Points the curve passes through, in order.
            params: Optional explicit parameter value for each point (strictly
                increasing). When given, ``point_at_parameter`` uses exactly
                these values, so callers can pin known features (e.g. a leading
                edge) to a chosen parameter. When omitted, SciPy's default
                chord-length parameterization is used.
        """
        self.points = points
        self.params = params

        self.start = points[0]
        self.end = points[-1]

        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        k = min(3, len(points) - 1)  # Spline degree
        if params is not None:
            self.tck, _u = splprep([xs, ys, zs], u=params, s=0.0, k=k)
        else:
            self.tck, _u = splprep([xs, ys, zs], s=0.0, k=k)

    def transform(self, position: Position) -> "InterpolatedLine":
        """Transform the line by transforming its defining points and re-splining.

        For an interpolated line, the control points depend on the interpolation
        algorithm. Transforming the original through-points and regenerating the
        spline ensures the transformed line still passes through the transformed
        points.
        """
        new_points = [position.point_in_global(p) for p in self.points]
        return InterpolatedLine(points=new_points, params=self.params)


class TrimmedCurve(Curve):
    def __init__(self, base_curve: Curve, u_start: float, u_end: float):
        if not (0.0 <= u_start <= 1.0 and 0.0 <= u_end <= 1.0):  # Noqa PLR2004
            raise ValueError("`u_start` and `u_end` must be in [0, 1].")
        if u_end <= u_start:
            raise ValueError("`u_end` must be > `u_start`.")

        self.base_curve = base_curve
        self.u_start = float(u_start)
        self.u_end = float(u_end)

        self.start = self.base_curve.point_at_parameter(self.u_start)
        self.end = self.base_curve.point_at_parameter(self.u_end)

    def _to_base_u(self, u_trim: float) -> float:
        """Convert parameter from trimmed curve to base curve."""
        return self.u_start + float(u_trim) * (self.u_end - self.u_start)

    def _from_base_u(self, u_base: float) -> float:
        """Convert parameter from base curve to trimmed curve."""
        return (float(u_base) - self.u_start) / (self.u_end - self.u_start)

    def point_at_parameter(self, u: float) -> Point:
        """Get point on trimmed curve at parameter u in [0, 1]."""
        return self.base_curve.point_at_parameter(self._to_base_u(u))

    def parameter_at_point(
        self, point: Point, tol: float = 1e-9, n_initial_samples: int = 20
    ) -> float:
        """Find parameter on trimmed curve corresponding to given point."""
        u_base = self.base_curve.parameter_at_point(
            point, tol=tol, n_initial_samples=n_initial_samples
        )
        return self._from_base_u(u_base)

    def transform(self, position: Position) -> "TrimmedCurve":
        """Transform the trimmed curve without changing its parameter trimming."""
        return TrimmedCurve(self.base_curve.transform(position), self.u_start, self.u_end)
