"""Build a Gordon surface for a wing from a WingShape.

A :class:`~geometry.aero.wing_shape.WingShape` only describes *distributions*
(chord, twist, airfoil vs. spanwise position). :class:`WingSurface` turns that
into actual 3D geometry: it samples airfoil sections at a number of spanwise
stations, turns each into a closed section curve (the Gordon *profiles*), and
builds *guide* curves along the leading and trailing edges. These are fed to a
strict :class:`~geometry.surfaces.GordonSurface`.

Coordinate convention:
    x  chordwise (toward the trailing edge before twist)
    y  spanwise  (root at y=0, tip at y=semispan)
    z  "up" / thickness direction

Each section is scaled by its chord and placed relative to a straight, unswept
reference line at x=0, z=0 (the quarter-chord line by default). The section is
rotated by its twist about that reference point (positive twist = leading edge
up) and translated to its spanwise station. With the default quarter-chord
reference, the leading edge therefore sweeps forward as the chord changes while
the quarter-chord line stays straight.

Parameterization (chosen so the strict Gordon network is exactly compatible):
    - profiles run TE(upper) -> LE -> TE(lower) with u=0 at the upper TE,
      u=0.5 pinned at the LE, u=1 at the lower TE.
    - guides are parameterized by the spanwise fraction eta, so every guide
      shares the same v at a given station.
"""
from __future__ import annotations

import math

import numpy as np

from geometry.aero.cst import AirfoilCST
from geometry.aero.wing_shape import WingShape
from geometry.curves import InterpolatedLine
from geometry.primitives import Point
from geometry.surfaces import GordonSurface


class WingSurface:
    """A Gordon surface skinned over the sections of a :class:`WingShape`.

    Args:
        wing_shape: The wing parametrisation to build geometry from.
        n_sections: Number of spanwise airfoil stations (profiles), >= 2.
        n_chord: Number of chordwise sample points per airfoil surface.
        tanh_p: Clustering parameter for the airfoil chordwise sampling.
        tol: Compatibility tolerance passed to the Gordon surface.
    """

    def __init__(
        self,
        wing_shape: WingShape,
        n_sections: int = 8,
        n_chord: int = 60,
        tanh_p: float = 1.0,
        reference_chord_fraction: float = 0.25,
        tol: float = 1e-6,
    ):
        if n_sections < 2:  # noqa: PLR2004
            raise ValueError("n_sections must be >= 2.")
        self.wing_shape = wing_shape
        self.n_sections = n_sections
        self.n_chord = n_chord
        self.tanh_p = tanh_p
        # Fraction of the chord that defines the straight, unswept reference line
        # the sections are centred on and twisted about (0.25 = quarter chord).
        self.reference_chord_fraction = reference_chord_fraction

        self.etas = list(np.linspace(0.0, 1.0, n_sections))

        self.profiles: list[InterpolatedLine] = []
        le_points: list[Point] = []
        te_points: list[Point] = []
        ref_points: list[Point] = []
        for eta in self.etas:
            profile, le_pt, te_pt, ref_pt = self._build_section(eta)
            self.profiles.append(profile)
            le_points.append(le_pt)
            te_points.append(te_pt)
            ref_points.append(ref_pt)

        # Guides parameterized by spanwise fraction so all share v_i = eta_i.
        self.le_guide = InterpolatedLine(le_points, params=self.etas)
        self.te_guide = InterpolatedLine(te_points, params=self.etas)
        # Straight reference (quarter-chord) line, for reference/plotting.
        self.reference_guide = InterpolatedLine(ref_points, params=self.etas)

        # Profiles run TE -> LE -> TE, so the boundary guides (u=0 and u=1) are
        # both the trailing edge, with the leading edge as the interior guide.
        self.surface = GordonSurface(
            profiles=self.profiles,
            guides=[self.te_guide, self.le_guide, self.te_guide],
            profile_v_params=self.etas,
            guide_u_params=[0.0, 0.5, 1.0],
            tol=tol,
        )

    def _section_point(
        self, x_c: float, y_c: float, chord: float, twist_rad: float, span_y: float
    ) -> Point:
        """Map a normalized airfoil coordinate (x/c, y/c) to a 3D point.

        The chord-line point at ``reference_chord_fraction`` is placed on the
        straight reference line (x=0, z=0); twist rotates the section about it.
        """
        x = (x_c - self.reference_chord_fraction) * chord
        z = y_c * chord
        # Rotate about the reference point (Y axis); positive twist = LE up.
        cos_t, sin_t = math.cos(twist_rad), math.sin(twist_rad)
        x_r = x * cos_t + z * sin_t
        z_r = -x * sin_t + z * cos_t
        return Point(x_r, span_y, z_r)

    def _build_section(self, eta: float) -> tuple[InterpolatedLine, Point, Point, Point]:
        """Build one section and return (profile, le_point, te_point, reference_point)."""
        airfoil: AirfoilCST = self.wing_shape.airfoil_distribution(eta)
        chord = float(self.wing_shape.chord_distribution(eta))
        twist_rad = math.radians(float(self.wing_shape.twist_distribution(eta)))
        span_y = eta * self.wing_shape.semispan

        upper_pts, lower_pts = airfoil.coordinates(self.n_chord, tanh_p_sampling=self.tanh_p)

        # Loop order: upper TE -> LE (reverse upper), then LE -> lower TE.
        upper_rev = list(reversed(upper_pts))  # x: 1 -> 0
        lower = lower_pts  # x: 0 -> 1
        le_idx = len(upper_rev) - 1  # LE sits at the junction

        loop = upper_rev + lower[1:]  # drop duplicated LE
        pts3d = [
            self._section_point(x_c, y_c, chord, twist_rad, span_y) for (x_c, y_c) in loop
        ]

        params = self._loop_params(loop, le_idx)

        profile = InterpolatedLine(pts3d, params=params)
        le_point = pts3d[le_idx]
        te_point = pts3d[0]  # upper TE (== lower TE for a sharp trailing edge)
        ref_point = self._section_point(
            self.reference_chord_fraction, 0.0, chord, twist_rad, span_y
        )
        return profile, le_point, te_point, ref_point

    @staticmethod
    def _loop_params(loop: list[tuple[float, float]], le_idx: int) -> list[float]:
        """Parameter values for the section loop: upper -> [0, 0.5], lower -> [0.5, 1].

        Within each half the spacing follows chord-length, but the leading edge
        is pinned to exactly 0.5 so it is shared across all sections.
        """
        coords = np.array(loop, dtype=float)
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        params = np.empty(len(loop), dtype=float)
        # Upper half: indices 0..le_idx -> [0, 0.5]
        upper_len = cum[le_idx] if cum[le_idx] > 0 else 1.0
        params[: le_idx + 1] = 0.5 * cum[: le_idx + 1] / upper_len
        # Lower half: indices le_idx..end -> [0.5, 1]
        lower_cum = cum[le_idx:] - cum[le_idx]
        lower_len = lower_cum[-1] if lower_cum[-1] > 0 else 1.0
        params[le_idx:] = 0.5 + 0.5 * lower_cum / lower_len
        return params.tolist()

    def point_at_parameter(self, u: float, v: float) -> Point:
        """Evaluate the wing surface at chordwise u and spanwise v, each in [0, 1]."""
        return self.surface.point_at_parameter(u, v)

    def sample_grid(self, n_u: int = 60, n_v: int = 30) -> np.ndarray:
        """Sample the wing surface on a regular (n_u, n_v, 3) grid."""
        return self.surface.sample_grid(n_u, n_v)
