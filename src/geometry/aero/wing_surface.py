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
reference, the leading edge sweeps forward as the chord changes while the
quarter-chord line stays straight.

Parameterization (chosen so the strict Gordon network is exactly compatible):
    - profiles run TE(upper) -> LE -> TE(lower) with u=0 at the upper TE,
      u=0.5 pinned at the LE, u=1 at the lower TE.
    - the chordwise parameter law is selectable via ``chordwise_spacing``:
        * "arclength" (default): u tracks cumulative arc length along the
          section (normalized per half). Uniform u then gives roughly uniform
          spacing around the section, so the leading edge is well resolved
          without special clustering.
        * "chord": u is a linear function of chord fraction x/c, so an iso-u
          line follows a constant x/c across the span (useful e.g. for placing
          spanwise features), at the cost of poor leading-edge node density.
    - guides are parameterized by spanwise fraction eta, so every guide shares
      the same v at a given station.
"""
from __future__ import annotations

import math

import numpy as np

from geometry.aero.cst import AirfoilCST
from geometry.aero.wing_shape import WingShape
from geometry.curves import InterpolatedLine
from geometry.mesh import Mesh, mesh_surface
from geometry.primitives import Point
from geometry.surfaces import GordonSurface


class WingSurface:
    """A Gordon surface skinned over the sections of a :class:`WingShape`.

    Args:
        wing_shape: The wing parametrisation to build geometry from.
        n_sections: Number of spanwise airfoil stations (profiles), >= 2.
        n_chord: Number of chordwise sample points per airfoil surface.
        reference_chord_fraction: Chord fraction of the straight, unswept
            reference line the sections are centred on and twisted about
            (0.25 = quarter chord).
        scale: Uniform length multiplier applied to all coordinates (chord, span
            and thickness). Handy for unit conversion, e.g. ``scale=1000`` turns a
            WingShape defined in metres into millimetres.
        tol: Compatibility tolerance passed to the Gordon surface.
    """

    def __init__(
        self,
        wing_shape: WingShape,
        n_sections: int = 8,
        n_chord: int = 60,
        reference_chord_fraction: float = 0.25,
        chordwise_spacing: str = "arclength",
        scale: float = 1.0,
        tol: float = 1e-6,
    ):
        if n_sections < 2:  # noqa: PLR2004
            raise ValueError("n_sections must be >= 2.")
        if n_chord < 2:  # noqa: PLR2004
            raise ValueError("n_chord must be >= 2.")
        if chordwise_spacing not in ("arclength", "chord"):
            raise ValueError("chordwise_spacing must be 'arclength' or 'chord'.")
        self.wing_shape = wing_shape
        self.n_sections = n_sections
        self.n_chord = n_chord
        self.reference_chord_fraction = reference_chord_fraction
        self.scale = scale
        self.chordwise_spacing = chordwise_spacing

        # Shared chordwise sample stations (cosine: clustered at LE and TE).
        k = np.arange(n_chord)
        self._xs = 0.5 * (1.0 - np.cos(np.pi * k / (n_chord - 1)))  # x/c in [0, 1]
        # Chord-fraction parameter law (used when chordwise_spacing == "chord").
        self._chord_loop_u = self._chord_fraction_params(self._xs)

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

    @staticmethod
    def _chord_fraction_params(xs: np.ndarray) -> list[float]:
        """Chord-fraction parameter for the TE->LE->TE loop.

        Upper surface (TE -> LE): u = 0.5 * (1 - x/c), so u runs 0 -> 0.5.
        Lower surface (LE -> TE): u = 0.5 + 0.5 * x/c, so u runs 0.5 -> 1.
        """
        upper_u = [0.5 * (1.0 - float(xc)) for xc in xs[::-1]]
        lower_u = [0.5 + 0.5 * float(xc) for xc in xs[1:]]
        return upper_u + lower_u

    @staticmethod
    def _arclength_params(pts3d: list[Point], le_idx: int) -> list[float]:
        """Arc-length parameter for the loop, normalized per half.

        The upper half (indices 0..le_idx) maps to [0, 0.5] and the lower half
        (le_idx..end) to [0.5, 1] by cumulative 3D arc length, so the leading
        edge stays pinned at u=0.5 and the trailing edge at u=0 and u=1.
        """
        eps = 1e-12
        coords = np.array([p.as_array() for p in pts3d], dtype=float)
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        params = np.empty(len(pts3d), dtype=float)
        upper_len = cum[le_idx]
        if upper_len > eps:
            params[: le_idx + 1] = 0.5 * cum[: le_idx + 1] / upper_len
        else:
            # Degenerate (e.g. zero-chord) section: fall back to uniform.
            params[: le_idx + 1] = np.linspace(0.0, 0.5, le_idx + 1)
        lower_cum = cum[le_idx:] - cum[le_idx]
        lower_len = lower_cum[-1]
        if lower_len > eps:
            params[le_idx:] = 0.5 + 0.5 * lower_cum / lower_len
        else:
            params[le_idx:] = np.linspace(0.5, 1.0, len(pts3d) - le_idx)
        return params.tolist()

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
        return Point(self.scale * x_r, self.scale * span_y, self.scale * z_r)

    def _build_section(self, eta: float) -> tuple[InterpolatedLine, Point, Point, Point]:
        """Build one section and return (profile, le_point, te_point, reference_point)."""
        airfoil: AirfoilCST = self.wing_shape.airfoil_distribution(eta)
        chord = float(self.wing_shape.chord_distribution(eta))
        twist_rad = math.radians(float(self.wing_shape.twist_distribution(eta)))
        span_y = eta * self.wing_shape.semispan

        # Sample upper/lower at the shared x/c stations.
        upper = [(float(xc), airfoil.upper.sample(float(xc))) for xc in self._xs]
        lower = [(float(xc), airfoil.lower.sample(float(xc))) for xc in self._xs]

        # Loop order: upper TE -> LE (reverse upper), then LE -> lower TE.
        loop = upper[::-1] + lower[1:]  # drop the duplicated LE point
        pts3d = [self._section_point(x_c, y_c, chord, twist_rad, span_y) for (x_c, y_c) in loop]

        le_idx = self.n_chord - 1
        if self.chordwise_spacing == "chord":
            params = self._chord_loop_u
        else:  # "arclength"
            params = self._arclength_params(pts3d, le_idx)
        profile = InterpolatedLine(pts3d, params=params)
        le_point = pts3d[le_idx]
        te_point = pts3d[0]  # upper TE (== lower TE for a sharp trailing edge)
        ref_point = self._section_point(
            self.reference_chord_fraction, 0.0, chord, twist_rad, span_y
        )
        return profile, le_point, te_point, ref_point

    def section_loop(self, eta: float, n_loop: int = 80) -> list[Point]:
        """Closed airfoil loop (TE->LE->TE) at an arbitrary span fraction eta.

        Returns ``n_loop`` points around the loop with the trailing-edge endpoint
        dropped, so the loop does not repeat a point (ready to close into a wire).
        """
        profile, *_ = self._build_section(eta)
        us = np.linspace(0.0, 1.0, n_loop, endpoint=False)
        return [profile.point_at_parameter(float(u)) for u in us]

    def section_edges(self, eta: float, n_half: int = 40) -> tuple[list[Point], list[Point]]:
        """Upper and lower airfoil curves at span fraction eta.

        Returns ``(upper, lower)`` where ``upper`` runs TE->LE and ``lower`` runs
        LE->TE; they share the LE point (upper[-1] == lower[0]) and the TE point
        (lower[-1] == upper[0]). Useful for building a 2-edge section wire so a
        loft produces separate (4-sided) upper/lower faces.
        """
        profile, *_ = self._build_section(eta)
        u_upper = np.linspace(0.0, 0.5, n_half)   # TE -> LE
        u_lower = np.linspace(0.5, 1.0, n_half)   # LE -> TE
        upper = [profile.point_at_parameter(float(u)) for u in u_upper]
        lower = [profile.point_at_parameter(float(u)) for u in u_lower]
        return upper, lower

    def point_at_parameter(self, u: float, v: float) -> Point:
        """Evaluate the wing surface at chordwise u and spanwise v, each in [0, 1]."""
        return self.surface.point_at_parameter(u, v)

    def sample_grid(self, n_u: int = 60, n_v: int = 30) -> np.ndarray:
        """Sample the wing surface on a regular (n_u, n_v, 3) grid."""
        return self.surface.sample_grid(n_u, n_v)

    def mesh(
        self,
        u_params: list[float] | None = None,
        v_params: list[float] | None = None,
        n_u: int = 80,
        n_v: int = 20,
        wrap_u: bool = True,
    ) -> Mesh:
        """Build a quad :class:`~geometry.mesh.Mesh` over the wing surface.

        Args:
            u_params: Explicit chordwise parameter stations in [0, 1]. When given,
                ``n_u``/``wrap_u`` endpoint handling is up to the caller.
            v_params: Explicit spanwise parameter stations in [0, 1].
            n_u: Number of chordwise stations if ``u_params`` is not given.
            n_v: Number of spanwise stations if ``v_params`` is not given.
            wrap_u: Close the mesh in the chordwise direction (the section loop
                wraps around the trailing edge). When generating default
                ``u_params`` the trailing-edge endpoint is dropped to avoid
                duplicate nodes at the seam.
        """
        if u_params is None:
            u_params = list(np.linspace(0.0, 1.0, n_u, endpoint=not wrap_u))
        if v_params is None:
            v_params = list(np.linspace(0.0, 1.0, n_v))
        return mesh_surface(self, u_params, v_params, wrap_u=wrap_u)
