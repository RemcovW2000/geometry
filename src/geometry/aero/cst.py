"""CST polynomials."""
# ----------------------------
# Core classes
# ----------------------------
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from geometry.aero.airfoil import (  # noqa: F401
    CURVATURE_FINITE_DIFF_H,
    Airfoil,
    AirfoilCurve,
    parse_selig_dat,
)
from geometry.aero.utils import (
    ReprCallable,
    bernstein_poly,
    get_inverse_tanh_mapping,
)

STANDARD_CST_SHAPE = ReprCallable(lambda _: 1, repr_str="lambda _: 1")
STANDARD_CST_X_MAPPING = ReprCallable(lambda x: x, repr_str="lambda x: x")
NUMERICAL_TOL = 1e-12
AirfoilCST = Airfoil  # old name, kept until the thesis code uses Airfoil


def _apply_elementwise(fn: Callable, xs: np.ndarray) -> np.ndarray:
    """Evaluate fn over an array: one vectorised call when fn supports it, else point by point.

    Shape functions and mappings are user callables; the ones in this package
    (class functions, inverse-tanh mapping, nested CSTPolynomials) take arrays,
    a scalar-only lambda falls back to a Python loop with the same result.
    """
    try:
        out = np.asarray(fn(xs), dtype=float)
    except (TypeError, ValueError):
        out = None
    if out is not None and out.ndim == 0:  # a constant shape function such as lambda _: 1
        out = np.full(xs.shape, float(out))
    if out is None or out.shape != xs.shape:
        out = np.array([float(fn(float(x))) for x in xs.ravel()]).reshape(xs.shape)
    return out


class CSTPolynomial(AirfoilCurve):
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

    def sample(self, x: float | np.ndarray) -> float | np.ndarray:
        """Sample point(s) on the polynomial at x: a float for a scalar, an array for an array.

        Array evaluation is a single vectorised pass through the mapping, the
        shape function and the Bernstein sum (so nested CSTPolynomials — a
        fitted polynomial used as the shape function of another — evaluate
        arrays end to end). A shape function or mapping that only accepts
        scalars is evaluated point by point.
        """
        if np.ndim(x) == 0:
            u = self.x_mapping(x) if self.x_mapping is not None else x
            u = float(u)
            if u < 0.0 or u > 1.0:  # noqa PLR2004
                raise ValueError(f"Input x={x} maps to u={u} outside [0,1].")
            return self.shape(u) * bernstein_poly(self.coeffs, u)
        xs = np.asarray(x, dtype=float)
        us = _apply_elementwise(self.x_mapping, xs) if self.x_mapping is not None else xs
        if np.any(us < 0.0) or np.any(us > 1.0):
            bad = xs[(us < 0.0) | (us > 1.0)][0]
            raise ValueError(f"Input x={bad} maps to u outside [0,1].")
        return _apply_elementwise(self.shape, us) * bernstein_poly(self.coeffs, us)


class UpperLower(Enum):
    """Enum used for choosing upper or lower surfaces in airfoils."""

    UPPER = "UPPER"
    LOWER = "LOWER"


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
    upper_coords, lower_coords = parse_selig_dat(Path(airfoil_file_path))
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

    airfoil = Airfoil(upper=upper_poly, lower=lower_poly)

    # Write to .dat file
    airfoil.to_dat("example_airfoil.dat", name="ExampleCST", n_points=81)
    print("Wrote example_airfoil.dat")

    # Plot
    airfoil.plot(n_points=81)
