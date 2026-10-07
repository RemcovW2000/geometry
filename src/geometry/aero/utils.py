from __future__ import annotations

import math
from typing import Callable

import numpy as np

# ----------------------------
# Bernstein building blocks
# ----------------------------


def bernstein_basis(n: int, i: int, x: float) -> float:
    """B_i^n(x) = C(n,i) * x^i * (1-x)^(n-i), x in [0,1]."""
    if i < 0 or i > n:
        raise ValueError(f"i must be in [0,n]. Got i={i}, n={n}.")
    return math.comb(n, i) * (x**i) * ((1.0 - x) ** (n - i))


def bernstein_poly(coeffs: list[float], x: float | np.ndarray) -> float | np.ndarray:
    """S(x) = sum_{i=0..n} A_i * B_i^n(x), where n=len(coeffs)-1.

    `x` may be a scalar (a float is returned) or an array of any shape (an array
    of the same shape is returned; one vectorised evaluation).
    """
    if len(coeffs) == 0:
        raise ValueError("coeffs must not be empty.")
    n = len(coeffs) - 1
    if np.ndim(x) == 0:
        s = 0.0
        for i, a in enumerate(coeffs):
            s += a * bernstein_basis(n, i, float(x))
        return s
    return bernstein_matrix(n + 1, np.asarray(x, dtype=float)) @ np.asarray(coeffs, dtype=float)


def bernstein_matrix(n_coeffs: int, x: np.ndarray) -> np.ndarray:
    """(*x.shape, n_coeffs) array of the Bernstein basis B_i^n(x), n = n_coeffs - 1.

    B_i^n(x) = C(n, i) x^i (1 - x)^(n - i); the powers come from cumulative
    products, so the whole basis costs a handful of numpy passes over `x`.
    """
    n = n_coeffs - 1
    x = np.asarray(x, dtype=float)
    binomials = np.array([math.comb(n, i) for i in range(n + 1)], dtype=float)
    powers = np.ones(x.shape + (n + 1,))
    powers[..., 1:] = x[..., None]
    powers = np.cumprod(powers, axis=-1)  # x^0 .. x^n
    powers_c = np.ones(x.shape + (n + 1,))
    powers_c[..., 1:] = (1.0 - x)[..., None]
    powers_c = np.cumprod(powers_c, axis=-1)[..., ::-1]  # (1-x)^n .. (1-x)^0
    return binomials * powers * powers_c


# ----------------------------
# repr callables used for distributions, shape functions and mappings
# ----------------------------


class ReprCallable:
    """Wrapper that gives a callable a custom __repr__."""

    def __init__(self, fn: Callable[[float], float], repr_str: str, dep_str: str = "") -> None:
        self._fn = fn
        self._repr_str: str = repr_str
        self.dep_str: str = dep_str

    def __call__(self, x: float) -> float:
        return self._fn(x)

    def __repr__(self) -> str:
        return self._repr_str


def create_cst_class_function(N1: float = 0.5, N2: float = 1.0) -> ReprCallable:
    """CST class function: C(x) = x^N1 * (1-x)^N2."""
    repr_str = f"create_cst_class_function(" f"N1={N1}, " f"N2={N2}, " ")"
    dep_str = "from airfoil_parametrisation import create_cst_class_function"
    return ReprCallable(lambda x: (x**N1) * ((1.0 - x) ** N2), repr_str, dep_str)


def get_inverse_tanh_mapping(p: float) -> ReprCallable:
    """
    Symmetric monotone mapping u(x) in [0,1]:
    """
    if p <= 0.0:  # noqa PLR2004
        raise ValueError("p must be > 0.")

    tanh_p = math.tanh(p)

    def mapping(x: float | np.ndarray) -> float | np.ndarray:
        if np.ndim(x) == 0:
            u = 0.5 * (1 + math.atanh((2 * float(x) - 1) * tanh_p) / p)
            return max(0.0, min(1.0, u))  # Clamp to [0,1] to avoid numerical issues
        # arrays: the same formula vectorised (numpy's atanh may differ from libm's in
        # the last bit, i.e. ~1e-16 in u)
        u = 0.5 * (1.0 + np.arctanh((2.0 * np.asarray(x, dtype=float) - 1.0) * tanh_p) / p)
        return np.clip(u, 0.0, 1.0)

    repr_str = f"lambda x: max(0.0, min(1.0, 0.5*(1 + math.atanh((2*x-1)*math.tanh({p}))/({p}))))"
    return ReprCallable(mapping, repr_str=repr_str, dep_str="import math")


# Convenience functions for chord distributions
def constant_distr(root_val: float) -> ReprCallable:
    """Create a constant chord distribution (rectangular wing)."""
    repr_str = f"lambda chord: chord = {root_val}"
    return ReprCallable(lambda eta: root_val, repr_str)


def linear_distr(root_val: float, tip_val: float) -> ReprCallable:
    """Create a linear distribution between root_val and tip_val."""
    repr_str = f"lambda eta: {root_val} + ({tip_val} - {root_val}) * eta"
    return ReprCallable(lambda eta: root_val + (tip_val - root_val) * eta, repr_str)


def elliptic_distr(root_val: float) -> ReprCallable:
    """Create an elliptic chord distribution."""
    repr_str = f"lambda eta: {root_val} * np.sqrt(1 - eta**2)"
    dep_str = "import numpy as np"
    return ReprCallable(lambda eta: root_val * np.sqrt(1 - eta**2), repr_str, dep_str)


# ----------------------------
# misc helpers
# ----------------------------


def tanh_density(x: float | np.ndarray, p: float) -> float | np.ndarray:
    """
    Density implied by the inverse-tanh-symmetric mapping (scalar or array `x`).

    The mapping is u(x) = 0.5*(1 + atanh((2x-1)*tanh(p))/p).
    Density ~ du/dx, which clusters points near x=0 and x=1.

    Parameters
    ----------
    x : float
        Position in [0, 1].
    p : float
        Clustering parameter (higher = more LE/TE clustering).

    Returns
    -------
    float
        The density (du/dx) at x.
    """
    if p <= 0:
        return 1.0 if np.ndim(x) == 0 else np.ones(np.shape(x))  # fallback to uniform
    t = math.tanh(p)
    if np.ndim(x) == 0:
        arg = (2.0 * float(x) - 1.0) * t
        # Clamp to avoid division issues at boundaries
        arg = max(-0.9999, min(0.9999, arg))
        # du/dx = t / (p * (1 - arg^2))
        return t / (p * (1.0 - arg * arg))
    arg = np.clip((2.0 * np.asarray(x, dtype=float) - 1.0) * t, -0.9999, 0.9999)
    return t / (p * (1.0 - arg * arg))


def cosine_spacing(n: int) -> list[float]:
    """
    Generate *n* points in [0,1] with half-cosine spacing (denser at LE/TE).

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    list[float]
        x-stations in [0, 1].
    """
    return [0.5 * (1.0 - math.cos(math.pi * i / (n - 1))) for i in range(n)]
