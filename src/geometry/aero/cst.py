"""CST polynomials and CST airfoil bases."""
# ----------------------------
# Core classes
# ----------------------------
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from geometry.aero.airfoil import Airfoil, AirfoilCurve, parse_selig_dat
from geometry.aero.utils import (
    ReprCallable,
    bernstein_matrix,
    bernstein_poly,
    create_cst_class_function,
    get_inverse_tanh_mapping,
)

STANDARD_CST_SHAPE = ReprCallable(lambda _: 1, repr_str="lambda _: 1")
STANDARD_CST_X_MAPPING = ReprCallable(lambda x: x, repr_str="lambda x: x")
NUMERICAL_TOL = 1e-12


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


class CSTBasis:
    """
    Shape functions of a CST airfoil family: y = shape(x) * sum_i a_i B_i^n(x) on each surface.

    A coefficient vector holds the `n_coeffs` upper-surface coefficients, then the lower ones.
    """

    def __init__(self, upper: Callable, lower: Callable, n_coeffs: int = 5) -> None:
        self.upper = upper
        self.lower = lower
        self.n_coeffs = n_coeffs

    def __repr__(self) -> str:
        return f"CSTBasis(upper={self.upper!r}, lower={self.lower!r}, n_coeffs={self.n_coeffs!r})"

    @classmethod
    def class_function(cls, n_coeffs: int = 5, n1: float = 0.5, n2: float = 1.0) -> CSTBasis:
        """Kulfan's class function x^n1 (1 - x)^n2 on both surfaces (standard CST)."""
        shape = create_cst_class_function(n1, n2)
        return cls(shape, shape, n_coeffs)

    @classmethod
    def from_dat(
        cls, path: Path | str, n_coeffs: int = 5, n_fit_coeffs: int = 20, tanh_p: float = 0.8
    ) -> CSTBasis:
        """The airfoil of a Selig .dat file as shape functions, so all coefficients 1 reproduce it.

        Each surface is fitted with an `n_fit_coeffs` CST polynomial on the inverse-tanh mapping.
        """
        mapping = get_inverse_tanh_mapping(p=tanh_p)
        upper, lower = (
            create_airfoil_y_callable(Path(path), side, n_fit_coeffs, mapping)
            for side in (UpperLower.UPPER, UpperLower.LOWER)
        )
        return cls(upper, lower, n_coeffs)

    def airfoil(self, coeffs: list[float] | np.ndarray, **kwargs: float | bool) -> Airfoil:
        """Airfoil of a coefficient vector; keyword arguments (TE gap, flap) go to `Airfoil`."""
        c = [float(v) for v in coeffs]
        return Airfoil(
            CSTPolynomial(c[: self.n_coeffs], shape=self.upper),
            CSTPolynomial(c[self.n_coeffs :], shape=self.lower),
            **kwargs,
        )

    def matrices(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(len(x), n_coeffs) matrices M with y_upper = M_upper @ a_upper, y_lower = M_lower @ a_lower."""
        x = np.asarray(x, dtype=float)
        b = bernstein_matrix(self.n_coeffs, x)
        return (
            _apply_elementwise(self.upper, x)[:, None] * b,
            _apply_elementwise(self.lower, x)[:, None] * b,
        )

    def fit(self, airfoil: Airfoil, n_stations: int = 241) -> np.ndarray:
        """Coefficients that best reproduce the base surfaces of `airfoil` (no TE gap or flap).

        Least squares on cosine-spaced stations.
        """
        x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_stations)))[1:-1]
        m_upper, m_lower = self.matrices(x)
        a_upper = np.linalg.lstsq(m_upper, airfoil.upper.sample(x), rcond=None)[0]
        a_lower = np.linalg.lstsq(m_lower, airfoil.lower.sample(x), rcond=None)[0]
        return np.concatenate([a_upper, a_lower])


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
