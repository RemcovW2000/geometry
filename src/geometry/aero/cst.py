"""CST polynomials."""
# ----------------------------
# Core classes
# ----------------------------
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from geometry.aero.utils import (
    ReprCallable,
    bernstein_poly,
    get_inverse_tanh_mapping,
    tanh_density,
)

STANDARD_CST_SHAPE = ReprCallable(lambda _: 1, repr_str="lambda _: 1")
STANDARD_CST_X_MAPPING = ReprCallable(lambda x: x, repr_str="lambda x: x")
NUMERICAL_TOL = 1e-12
CURVATURE_FINITE_DIFF_H = 1e-6
LE_RADIUS_X_OFFSET = 1e-4


class CSTPolynomial:
    """
    y(x) = shape(u) * sum_i A_i * B_i^n(u), where u = mapping(x) or x.

    - coeffs:  Bernstein coefficients A_i (length n+1)
    - shape:   callable shape(u) -> float   (usually class function, but can be anything)
    - mapping: callable mapping(x) -> float (optional x-warp). If None -> identity.
    """

    def __init__(
        self,
        coeffs: list[float],
        shape: ReprCallable = STANDARD_CST_SHAPE,
        x_mapping: ReprCallable = STANDARD_CST_X_MAPPING,
    ) -> None:
        self.coeffs = coeffs
        self.shape = shape
        self.x_mapping = x_mapping

    def __repr__(self) -> str:
        return (
            f"CSTPolynomial(coeffs={self.coeffs!r}, "
            f"shape={self.shape!r}, "
            f"x_mapping={self.x_mapping!r})"
        )

    def sample(self, x: float) -> float:
        """Sample point on polynomial at x."""
        u = self.x_mapping(x) if self.x_mapping is not None else x
        u = float(u)
        if u < 0.0 or u > 1.0:  # noqa PLR2004
            raise ValueError(f"Input x={x} maps to u={u} outside [0,1].")
        return self.shape(u) * bernstein_poly(self.coeffs, u)

    def get_callable(self) -> Callable[[float], float]:
        """Return a scalar function: float -> float."""
        return self.sample

    def curvature_at(self, x: float, h: float = CURVATURE_FINITE_DIFF_H) -> float:
        """Approximate curvature kappa = |y''| / (1 + y'^2)^1.5 using finite differences."""
        if x <= h:
            x0 = x
            x1 = x + h
            x2 = x + 2 * h

            y0 = self.sample(x0)
            y1 = self.sample(x1)
            y2 = self.sample(x2)

            first_derivative = (-3 * y0 + 4 * y1 - y2) / (2 * h)
            second_derivative = (y0 - 2 * y1 + y2) / (h * h)

        elif x >= 1 - h:
            x0 = x - 2 * h
            x1 = x - h
            x2 = x

            y0 = self.sample(x0)
            y1 = self.sample(x1)
            y2 = self.sample(x2)

            first_derivative = (3 * y2 - 4 * y1 + y0) / (2 * h)
            second_derivative = (y0 - 2 * y1 + y2) / (h * h)

        else:
            x0 = x - h
            x1 = x
            x2 = x + h

            y0 = self.sample(x0)
            y1 = self.sample(x1)
            y2 = self.sample(x2)

            first_derivative = (y2 - y0) / (2 * h)
            second_derivative = (y2 - 2 * y1 + y0) / (h * h)

        return abs(second_derivative) / ((1 + first_derivative * first_derivative) ** 1.5)

    def normalized_cumsum(self, values: list[float] | np.ndarray) -> np.ndarray:
        """Normalize values in list."""
        cumsum = np.cumsum(values)
        cumsum += -cumsum[0]  # Shift to start at 0
        cumsum /= cumsum[-1]  # Normalize to end at 1
        return cumsum

    def curvature_adaptive_spacing(
        self,
        n: int,
        n_seed: int = 501,
    ) -> list[float]:
        """Generate x-stations with density proportional to local curvature."""
        # Fine uniform grid to estimate curvature
        xs_seed = [i / (n_seed - 1) for i in range(n_seed)]
        curvatures = [self.curvature_at(x) for x in xs_seed]
        cumsum = self.normalized_cumsum(curvatures)

        # Invert: for each target in [0, 1], find corresponding x via interpolation
        targets = np.linspace(0.0, 1.0, n)
        xs_out = np.interp(targets, cumsum, xs_seed)
        return xs_out.tolist()

    def tahn_spacing(
        self,
        n: int,
        p: float = 0.8,
        n_seed: int = 501,
    ) -> list[float]:
        """Generate x-stations with density proportional to tanh-based function."""
        tahn_sampling_density = [tanh_density(x, p) for x in np.linspace(0.0, 1.0, n_seed)]
        cumsum = self.normalized_cumsum(tahn_sampling_density)

        targets = np.linspace(0.0, 1.0, n)
        xs_out = np.interp(targets, cumsum, np.linspace(0.0, 1.0, n_seed))
        return xs_out.tolist()

    def hybrid_spacing(
        self,
        n: int,
        p: float = 0.8,
        n_seed: int = 501,
    ) -> list[float]:
        """Generate x-stations w density proportional max of tahn vs curvature densities."""
        x_seed = np.linspace(0.0, 1.0, n_seed)
        tahn_sampling_density = [tanh_density(x, p) for x in x_seed]
        tahn_cumsum = np.cumsum(tahn_sampling_density)
        tahn_normalized_density = tahn_sampling_density / tahn_cumsum[-1]

        curvatures = [self.curvature_at(x) for x in x_seed]
        curvature_cumsum = np.cumsum(curvatures)
        curvature_normalized_density = curvatures / curvature_cumsum[-1]

        combined_density = np.maximum(tahn_normalized_density, curvature_normalized_density)
        combined_cumsum = self.normalized_cumsum(combined_density)

        targets = np.linspace(0.0, 1.0, n)
        xs_out = np.interp(targets, combined_cumsum, x_seed)
        return xs_out.tolist()

    def trailing_edge_angle(self) -> float:
        """Approximate trailing edge angle in degrees using finite difference slope at x=1."""
        h = CURVATURE_FINITE_DIFF_H
        slope = (self.sample(1.0) - self.sample(1.0 - h)) / h
        return np.degrees(np.arctan(slope))


class AirfoilCST:
    """
    Airfoil defined by upper and lower CST polynomials.

    Parameters
    ----------
    upper : CST polynomial for the upper surface (y >= 0).
    lower : CST polynomial for the lower surface (y <= 0 typically).
    """

    def __init__(self, upper: CSTPolynomial, lower: CSTPolynomial) -> None:
        self.upper = upper
        self.lower = lower

    def __repr__(self) -> str:
        return f"AirfoilCST(upper={self.upper!r}, lower={self.lower!r})"

    def inverted_surface(self, n_points: int = 101, tanh_p: float = 0.8) -> bool:
        """Check that upper surface is always above lower surface and vice versa."""
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)

        xu, yu = zip(*upper_pts)
        xl, yl = zip(*lower_pts)

        # Check upper points are above interpolated lower surface
        for x, y_upper in upper_pts:
            y_lower_interp = np.interp(x, xl, yl)
            if y_upper < y_lower_interp:
                return True

        # Check lower points are below interpolated upper surface
        for x, y_lower in lower_pts:
            y_upper_interp = np.interp(x, xu, yu)
            if y_lower > y_upper_interp:
                return True

        return False

    def coordinates(
        self, n_points: int, tanh_p_sampling: float
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Get upper and lower surface coordinates."""
        xs_upper = self.upper.hybrid_spacing(n_points, p=tanh_p_sampling)
        xs_lower = self.lower.hybrid_spacing(n_points, p=tanh_p_sampling)

        upper_pts = [(x, self.upper.sample(x)) for x in xs_upper]
        lower_pts = [(x, self.lower.sample(x)) for x in xs_lower]
        return upper_pts, lower_pts

    def to_dat(
        self,
        path: str,
        name: str = "airfoil",
        n_points: int = 101,
        tanh_p: float = 1.0,
    ) -> None:
        """
        Write Selig-style .dat file readable by XFOIL.

        The file lists coordinates from upper TE -> LE -> lower TE.
        """
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)

        # Selig format: upper TE -> LE (reversed), then lower LE -> TE (skip duplicate LE)
        selig_coords = list(reversed(upper_pts)) + lower_pts[1:]

        with open(path, "w") as f:
            f.write(f"{name}\n")
            for x, y in selig_coords:
                f.write(f"  {x: .7f}  {y: .7f}\n")

    def plot(
        self,
        n_points: int = 101,
        tanh_p: float = 2.0,
        show: bool = True,
    ) -> None:
        """
        Plot the airfoil using matplotlib.

        Parameters
        ----------
        n_points : Number of sample points per surface.
        spacing : "hybrid" (default), "curvature", or "tanh".
        tanh_p : Parameter for tanh clustering.
        show : If True, call plt.show().
        """
        upper_pts, lower_pts = self.coordinates(n_points, tanh_p_sampling=tanh_p)

        xu, yu = zip(*upper_pts)
        xl, yl = zip(*lower_pts)

        plt.figure()
        plt.plot(xu, yu, label="upper")
        plt.plot(xl, yl, label="lower")
        plt.axis("equal")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.legend()
        plt.title("Airfoil (CST)")

        if show:
            plt.show()

    def thickness_at(self, x: float) -> float:
        """Return local thickness (y_upper - y_lower) at given x."""
        return self.upper.sample(x) - self.lower.sample(x)

    def thickest_point(self, n_search_points: int = 100) -> tuple[float, float]:
        """Return (x, thickness) of the thickest point on the airfoil."""
        xs = np.linspace(0.0, 1.0, n_search_points)
        thicknesses = [self.thickness_at(x) for x in xs]
        max_index = np.argmax(thicknesses)
        return xs[max_index], thicknesses[max_index]

    def leading_edge_radius(
        self, x_offset: float = LE_RADIUS_X_OFFSET, h: float = CURVATURE_FINITE_DIFF_H
    ) -> float:
        """Approximate leading edge radius using curvature at x=x_offset."""
        max_curvature = max(
            self.upper.curvature_at(x_offset, h=h), self.lower.curvature_at(x_offset, h=h)
        )
        if max_curvature == 0.0:  # noqa PLR2004
            return float("inf")  # Flat leading edge
        return 1.0 / max_curvature

    def trailing_edge_angle(self) -> float:
        """Approximate trailing edge angle in degrees using finite difference slope at x=1."""
        lower_slope = self.lower.trailing_edge_angle()
        upper_slope = self.upper.trailing_edge_angle()
        return np.degrees(np.arctan(upper_slope - lower_slope))


class UpperLower(Enum):
    """Enum used for choosing upper or lower surfaces in airfoils."""

    UPPER = "UPPER"
    LOWER = "LOWER"


def _parse_airfoil_dat(
    airfoil_file_path: Path
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parse a Selig-style .dat file and return (upper_coords, lower_coords), both sorted by increasing x."""
    with open(airfoil_file_path) as f:
        lines = f.readlines()

    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        x, y = map(float, parts[:2])
        coords.append((x, y))

    leading_edge_point = min(coords, key=lambda pt: pt[0])
    leading_edge_index = coords.index(leading_edge_point)
    upper_surface_coords = coords[: leading_edge_index + 1][::-1]  # LE -> TE, increasing x
    lower_surface_coords = coords[leading_edge_index:]  # LE -> TE, increasing x
    return upper_surface_coords, lower_surface_coords


def fit_cst_to_airfoil(
    airfoil_file_path: Path,
    upper_lower: UpperLower,
    n_coeffs: int = 6,
    x_mapping: ReprCallable | None = None,
) -> CSTPolynomial:
    """
    Fit a CST polynomial to an airfoil surface from a .dat file using least-squares.

    The standard CST shape function u^0.5 * (1-u)^1.0 is used, which naturally
    enforces a vertical tangent at the leading edge.
    """
    upper_coords, lower_coords = _parse_airfoil_dat(airfoil_file_path)
    surface_coords = upper_coords if upper_lower == UpperLower.UPPER else lower_coords

    # Exclude LE (x=0) and TE (x=1) from fitting targets since shape function is zero there
    data_points = [(x, y) for x, y in surface_coords if NUMERICAL_TOL < x < 1.0 - NUMERICAL_TOL]
    x_data = np.array([pt[0] for pt in data_points])
    y_data = np.array([pt[1] for pt in data_points])

    shape = lambda u: u**0.5 * (1.0 - u) ** 1.0

    def residuals(coeffs: np.ndarray) -> np.ndarray:
        poly = CSTPolynomial(coeffs=coeffs.tolist(), shape=shape, x_mapping=x_mapping)
        y_pred = np.array([poly.sample(x) for x in x_data])
        return y_pred - y_data

    initial_guess = np.zeros(n_coeffs)
    result = least_squares(residuals, initial_guess)

    return CSTPolynomial(coeffs=result.x.tolist(), shape=shape, x_mapping=x_mapping)


def create_airfoil_y_callable(
    airfoil_file_path: Path,
    upper_lower: UpperLower,
    n_cst_coeffs: int = 20,
    x_mapping: ReprCallable | None = None,
) -> ReprCallable:
    """Create callable that takes an x/c value and returns corresponding y/c value for an existing airfoil defined in a .dat file."""
    poly = fit_cst_to_airfoil(airfoil_file_path, upper_lower, n_cst_coeffs, x_mapping)
    repr_str = (
        f"create_airfoil_y_callable("
        f"Path({str(airfoil_file_path)!r}), "
        f"UpperLower.{upper_lower.name}, "
        f"n_cst_coeffs={n_cst_coeffs!r}),"
        f"x_mapping={x_mapping if x_mapping is not None else 'None'}"
    )
    dep_str = (
        "from airfoil_parametrisation.cst import create_airfoil_y_callable, UpperLower\n"
        "from pathlib import Path"
    )
    return ReprCallable(poly.get_callable(), repr_str, dep_str)


if __name__ == "__main__":
    # Example: create upper and lower CST polynomials
    mapping = get_inverse_tanh_mapping(p=0.8)
    shape = lambda u: u**0.5 * (1 - u) ** 1.0

    upper_coeffs = [0.15, 0.25, 0.20, 0.15, 0.10]
    lower_coeffs = [-0.05, -0.08, -0.06, -0.04, -0.02]

    upper_poly = CSTPolynomial(coeffs=upper_coeffs, shape=shape, x_mapping=mapping)
    lower_poly = CSTPolynomial(coeffs=lower_coeffs, shape=shape, x_mapping=mapping)

    airfoil = AirfoilCST(upper=upper_poly, lower=lower_poly)

    # Write to .dat file
    airfoil.to_dat("example_airfoil.dat", name="ExampleCST", n_points=81)
    print("Wrote example_airfoil.dat")

    # Plot
    airfoil.plot(n_points=81)
