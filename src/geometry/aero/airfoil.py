"""Airfoil geometry: an upper and a lower surface curve y(x/c) plus a trailing-edge gap."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar

from geometry.aero.utils import tanh_density

CURVATURE_FINITE_DIFF_H = 1e-6
LE_RADIUS_STATIONS = (0.002, 0.004, 0.006)  # x where the nose thickness defines the LE radius
TE_SEARCH_X = 0.7  # the rounded trailing edge is searched aft of this chord fraction
TE_CROSS_SLOPE = 0.05  # continuation past the rounded TE, so wing sections find the closure


class AirfoilCurve:
    """One airfoil surface y(x), x in [0, 1]. Subclasses implement `sample`."""

    def sample(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return y at x: a float for a scalar, an array for an array."""
        raise NotImplementedError

    def get_callable(self) -> Callable[[float], float]:
        """Return the curve as a function x -> y."""
        return self.sample

    def curvature_at(
        self, x: float | np.ndarray, h: float = CURVATURE_FINITE_DIFF_H
    ) -> float | np.ndarray:
        """Approximate curvature kappa = |y''| / (1 + y'^2)^1.5 using finite differences.

        Three-point stencils: forward at x <= h, backward at x >= 1 - h, central
        elsewhere. A scalar x takes the original scalar path; for an array of x
        all stencil points are evaluated in ONE `sample` call (its vectorised
        arithmetic can differ from the scalar path in the last bit, which the
        h^2 of the second derivative amplifies to ~1e-5 relative in kappa —
        irrelevant for a sampling density).
        """
        if np.ndim(x) != 0:
            return self._curvature_array(np.asarray(x, dtype=float), h)
        if x <= h:
            y0, y1, y2 = self.sample(x), self.sample(x + h), self.sample(x + 2 * h)
            first_derivative = (-3 * y0 + 4 * y1 - y2) / (2 * h)
            second_derivative = (y0 - 2 * y1 + y2) / (h * h)
        elif x >= 1 - h:
            y0, y1, y2 = self.sample(x - 2 * h), self.sample(x - h), self.sample(x)
            first_derivative = (3 * y2 - 4 * y1 + y0) / (2 * h)
            second_derivative = (y0 - 2 * y1 + y2) / (h * h)
        else:
            y0, y1, y2 = self.sample(x - h), self.sample(x), self.sample(x + h)
            first_derivative = (y2 - y0) / (2 * h)
            second_derivative = (y2 - 2 * y1 + y0) / (h * h)
        return abs(second_derivative) / ((1 + first_derivative * first_derivative) ** 1.5)

    def _curvature_array(self, xs: np.ndarray, h: float) -> np.ndarray:
        lo, hi = xs <= h, xs >= 1.0 - h
        mid = ~(lo | hi)
        xm, xl, xh = xs[mid], xs[lo], xs[hi]
        stencil = np.concatenate(
            [xm - h, xm, xm + h, xl, xl + h, xl + 2 * h, xh - 2 * h, xh - h, xh]
        )
        ys = np.asarray(self.sample(stencil), dtype=float)
        first = np.empty_like(xs)
        second = np.empty_like(xs)
        m, n_lo, n_hi = len(xm), len(xl), len(xh)
        y0, y1, y2 = ys[:m], ys[m : 2 * m], ys[2 * m : 3 * m]
        first[mid] = (y2 - y0) / (2 * h)
        second[mid] = (y2 - 2 * y1 + y0) / (h * h)
        o = 3 * m
        y0, y1, y2 = ys[o : o + n_lo], ys[o + n_lo : o + 2 * n_lo], ys[o + 2 * n_lo : o + 3 * n_lo]
        first[lo] = (-3 * y0 + 4 * y1 - y2) / (2 * h)
        second[lo] = (y0 - 2 * y1 + y2) / (h * h)
        o += 3 * n_lo
        y0, y1, y2 = ys[o : o + n_hi], ys[o + n_hi : o + 2 * n_hi], ys[o + 2 * n_hi :]
        first[hi] = (3 * y2 - 4 * y1 + y0) / (2 * h)
        second[hi] = (y0 - 2 * y1 + y2) / (h * h)
        return np.abs(second) / ((1 + first * first) ** 1.5)

    def normalized_cumsum(self, values: list[float] | np.ndarray) -> np.ndarray:
        """Normalize values in list."""
        cumsum = np.cumsum(values)
        cumsum += -cumsum[0]  # Shift to start at 0
        cumsum /= cumsum[-1]  # Normalize to end at 1
        return cumsum

    def curvature_adaptive_spacing(self, n: int, n_seed: int = 501) -> list[float]:
        """Generate x-stations with density proportional to local curvature."""
        xs_seed = np.linspace(0.0, 1.0, n_seed)
        cumsum = self.normalized_cumsum(self.curvature_at(xs_seed))
        return np.interp(np.linspace(0.0, 1.0, n), cumsum, xs_seed).tolist()

    def tahn_spacing(self, n: int, p: float = 0.8, n_seed: int = 501) -> list[float]:
        """Generate x-stations with density proportional to tanh-based function."""
        x_seed = np.linspace(0.0, 1.0, n_seed)
        cumsum = self.normalized_cumsum(tanh_density(x_seed, p))
        return np.interp(np.linspace(0.0, 1.0, n), cumsum, x_seed).tolist()

    def hybrid_spacing(self, n: int, p: float = 0.8, n_seed: int = 501) -> list[float]:
        """Generate x-stations w density proportional max of tahn vs curvature densities."""
        x_seed = np.linspace(0.0, 1.0, n_seed)
        tahn_sampling_density = np.asarray(tanh_density(x_seed, p), dtype=float)
        tahn_normalized_density = tahn_sampling_density / np.cumsum(tahn_sampling_density)[-1]

        curvatures = self.curvature_at(x_seed)  # one vectorised evaluation of all seeds
        curvature_normalized_density = curvatures / np.cumsum(curvatures)[-1]

        combined_density = np.maximum(tahn_normalized_density, curvature_normalized_density)
        combined_cumsum = self.normalized_cumsum(combined_density)
        return np.interp(np.linspace(0.0, 1.0, n), combined_cumsum, x_seed).tolist()

    def trailing_edge_angle(self) -> float:
        """Approximate trailing edge angle in degrees using finite difference slope at x=1."""
        h = CURVATURE_FINITE_DIFF_H
        slope = (self.sample(1.0) - self.sample(1.0 - h)) / h
        return np.degrees(np.arctan(slope))


class InterpolatedAirfoilCurve(AirfoilCurve):
    """One airfoil surface interpolated between given points.

    Parameters
    ----------
    x, y : Points of the surface, in any order.
    kind : Interpolation between the points: "linear", "quadratic" or "cubic".
    """

    def __init__(
        self, x: list[float] | np.ndarray, y: list[float] | np.ndarray, kind: str = "linear"
    ) -> None:
        order = np.argsort(x)
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]
        self.kind = kind
        self._interp = interp1d(
            self.x,
            self.y,
            kind=kind,
            assume_sorted=True,
            bounds_error=False,
            fill_value="extrapolate",
        )

    def __repr__(self) -> str:
        return f"InterpolatedAirfoilCurve(n_points={len(self.x)}, kind={self.kind!r})"

    @classmethod
    def from_dat(
        cls, path: Path | str, kind: str = "linear", header_lines: int = 1
    ) -> InterpolatedAirfoilCurve:
        """Read one surface (an x y pair per line) from a .dat file."""
        x, y = np.loadtxt(path, skiprows=header_lines, usecols=(0, 1), unpack=True)
        return cls(x, y, kind)

    def sample(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return the interpolated y at x."""
        y = self._interp(x)
        return float(y) if np.ndim(x) == 0 else y


class RoundedTrailingEdge(NamedTuple):
    """Circle closing a rounded TE: centre, radius and the x of the tangent points."""

    xc: float
    yc: float
    r: float
    x_tangent_upper: float
    x_tangent_lower: float


class Airfoil:
    """
    Airfoil defined by an upper and a lower surface curve, both y(x/c) with the LE at x = 0.

    Parameters
    ----------
    upper : Upper surface, e.g. a CSTPolynomial or an InterpolatedAirfoilCurve.
    lower : Lower surface.
    trailing_edge_gap : TE thickness in chord fractions. The surfaces are opened linearly
        from the LE, y_upper + gap/2 * x and y_lower - gap/2 * x (XFOIL's TGAP with
        blending 1); `upper` and `lower` stay the closed base shape.
    rounded_trailing_edge : Close the cut TE of a nonzero gap with a circular arc tangent to
        both opened surfaces; the arc ends at x = 1, where the two surfaces meet.
    """

    def __init__(
        self,
        upper: AirfoilCurve,
        lower: AirfoilCurve,
        trailing_edge_gap: float = 0.0,
        rounded_trailing_edge: bool = False,
    ) -> None:
        self.upper = upper
        self.lower = lower
        self.trailing_edge_gap = trailing_edge_gap
        self.rounded_trailing_edge = rounded_trailing_edge
        rounded = rounded_trailing_edge and trailing_edge_gap > 0.0
        self.trailing_edge = self._fit_rounded_trailing_edge() if rounded else None

    def __repr__(self) -> str:
        return (
            f"Airfoil(upper={self.upper!r}, lower={self.lower!r}, "
            f"trailing_edge_gap={self.trailing_edge_gap!r}, "
            f"rounded_trailing_edge={self.rounded_trailing_edge!r})"
        )

    @classmethod
    def from_dat(
        cls,
        path: Path | str,
        kind: str = "linear",
        trailing_edge_gap: float = 0.0,
        rounded_trailing_edge: bool = False,
    ) -> Airfoil:
        """Airfoil interpolated on the points of a Selig-style .dat file (TE -> LE -> TE)."""
        upper, lower = parse_selig_dat(Path(path))
        return cls(
            InterpolatedAirfoilCurve(*zip(*upper), kind=kind),
            InterpolatedAirfoilCurve(*zip(*lower), kind=kind),
            trailing_edge_gap,
            rounded_trailing_edge,
        )

    def y_upper(self, x: float | np.ndarray) -> float | np.ndarray:
        """Upper surface y at x, TE gap (and rounding) included."""
        return self._surface(self.upper, 1.0, x)

    def y_lower(self, x: float | np.ndarray) -> float | np.ndarray:
        """Lower surface y at x, TE gap (and rounding) included."""
        return self._surface(self.lower, -1.0, x)

    def _opened(
        self, curve: AirfoilCurve, sign: float, x: float | np.ndarray
    ) -> float | np.ndarray:
        return curve.sample(x) + sign * 0.5 * self.trailing_edge_gap * np.asarray(x)

    def _surface(
        self, curve: AirfoilCurve, sign: float, x: float | np.ndarray
    ) -> float | np.ndarray:
        te = self.trailing_edge
        if te is None:
            return self._opened(curve, sign, x)
        x_tangent = te.x_tangent_upper if sign > 0 else te.x_tangent_lower
        xs = np.asarray(x, dtype=float)
        skin = self._opened(curve, sign, np.minimum(xs, x_tangent))
        arc = te.yc + sign * np.sqrt(np.maximum(te.r**2 - (xs - te.xc) ** 2, 0.0))
        beyond = te.yc - sign * TE_CROSS_SLOPE * (xs - 1.0)
        y = np.where(xs < x_tangent, skin, np.where(xs <= 1.0, arc, beyond))
        return float(y) if np.ndim(x) == 0 else y

    def _fit_rounded_trailing_edge(self) -> RoundedTrailingEdge:
        """Circle tangent to both opened surfaces whose aft point lies at x = 1."""
        upper = lambda x: self._opened(self.upper, 1.0, x)
        lower = lambda x: self._opened(self.lower, -1.0, x)

        def distance(fn: Callable, xc: float, yc: float) -> tuple[float, float]:
            res = minimize_scalar(
                lambda x: math.hypot(x - xc, fn(x) - yc),
                bounds=(TE_SEARCH_X, 1.0),
                method="bounded",
                options={"xatol": 1e-12},
            )
            return float(res.fun), float(res.x)

        def center_y(xc: float) -> float:
            y_lo, y_hi = lower(xc), upper(xc)
            t = y_hi - y_lo
            return brentq(
                lambda yc: distance(upper, xc, yc)[0] - distance(lower, xc, yc)[0],
                y_lo + 0.05 * t,
                y_hi - 0.05 * t,
                xtol=1e-14,
            )

        xc = brentq(
            lambda xc: xc + distance(upper, xc, center_y(xc))[0] - 1.0,
            TE_SEARCH_X + 0.02,
            1.0,
            xtol=1e-13,
        )
        yc = center_y(xc)
        return RoundedTrailingEdge(
            xc, yc, 1.0 - xc, distance(upper, xc, yc)[1], distance(lower, xc, yc)[1]
        )

    def coordinates(
        self, n_points: int, tanh_p_sampling: float
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Get upper and lower surface coordinates (LE -> TE lists of (x, y) per surface)."""
        xs_upper = np.asarray(self.upper.hybrid_spacing(n_points, p=tanh_p_sampling))
        xs_lower = np.asarray(self.lower.hybrid_spacing(n_points, p=tanh_p_sampling))
        ys_upper = np.asarray(self.y_upper(xs_upper), dtype=float)
        ys_lower = np.asarray(self.y_lower(xs_lower), dtype=float)
        upper_pts = list(zip(xs_upper.tolist(), ys_upper.tolist(), strict=True))
        lower_pts = list(zip(xs_lower.tolist(), ys_lower.tolist(), strict=True))
        return upper_pts, lower_pts

    def inverted_surface(self, n_points: int = 101, tanh_p: float = 0.8) -> bool:
        """Check that upper surface is always above lower surface and vice versa."""
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)
        xu, yu = zip(*upper_pts)
        xl, yl = zip(*lower_pts)
        if any(y < np.interp(x, xl, yl) for x, y in upper_pts):
            return True
        return any(y > np.interp(x, xu, yu) for x, y in lower_pts)

    def to_dat(
        self, path: str, name: str = "airfoil", n_points: int = 101, tanh_p: float = 1.0
    ) -> None:
        """Write a Selig-style .dat file readable by XFOIL, TE gap included."""
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)
        selig_coords = list(reversed(upper_pts)) + lower_pts[1:]
        with open(path, "w") as f:
            f.write(f"{name}\n")
            for x, y in selig_coords:
                f.write(f"  {x: .7f}  {y: .7f}\n")

    def plot(self, n_points: int = 101, tanh_p: float = 2.0, show: bool = True) -> None:
        """Plot the airfoil using matplotlib."""
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)
        plt.figure()
        plt.plot(*zip(*upper_pts), label="upper")
        plt.plot(*zip(*lower_pts), label="lower")
        plt.axis("equal")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.legend()
        plt.title("Airfoil")
        if show:
            plt.show()

    def thickness_at(self, x: float) -> float:
        """Return local thickness (y_upper - y_lower) at given x, TE gap included."""
        return self.y_upper(x) - self.y_lower(x)

    def thickest_point(self, n_search_points: int = 100) -> tuple[float, float]:
        """Return (x, thickness) of the thickest point on the airfoil."""
        xs = np.linspace(0.0, 1.0, n_search_points)
        thicknesses = [self.thickness_at(x) for x in xs]
        max_index = np.argmax(thicknesses)
        return xs[max_index], thicknesses[max_index]

    def leading_edge_radius(self, stations: tuple[float, ...] = LE_RADIUS_STATIONS) -> float:
        """Radius of the LE circle that is as thick as the closed contour near the nose.

        A circle of radius r through the LE is 2 sqrt(2 r x - x^2) thick at x, so the contour
        thickness t gives r = (t^2 / 4 + x^2) / (2 x); the smallest value over the stations counts.
        A surface that leaves the LE flat therefore cannot pass as a round nose.
        """
        return min((self.thickness_at(x) ** 2 / 4 + x**2) / (2 * x) for x in stations)

    def trailing_edge_angle(self) -> float:
        """Included angle between the upper and lower surface at the TE, in degrees."""
        h = CURVATURE_FINITE_DIFF_H
        slope_upper = (self.y_upper(1.0) - self.y_upper(1.0 - h)) / h
        slope_lower = (self.y_lower(1.0) - self.y_lower(1.0 - h)) / h
        return float(np.degrees(np.arctan(slope_lower) - np.arctan(slope_upper)))


def parse_selig_dat(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parse a Selig-style .dat file into (upper, lower) points, both LE -> TE."""
    with open(path) as f:
        lines = f.readlines()
    coords = [tuple(map(float, line.split()[:2])) for line in lines[1:] if line.strip()]
    leading_edge_index = coords.index(min(coords, key=lambda pt: pt[0]))
    upper_surface_coords = coords[: leading_edge_index + 1][::-1]  # LE -> TE, increasing x
    lower_surface_coords = coords[leading_edge_index:]  # LE -> TE, increasing x
    return upper_surface_coords, lower_surface_coords
