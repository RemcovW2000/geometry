"""Wingtip caps grown from a tip profile curve. Currently: the Hoerner tip.

There is no canonical published equation for the Hoerner tip; its defining
features (Hoerner, *Fluid-Dynamic Drag*) are: the lower surface carries
straight through to the tip, the upper surface rolls down onto it, and the
two meet in a SHARP edge on the lower side, so the tip vortex sheds from the
lowest point and the wing keeps close to its full effective span.
"""
from __future__ import annotations

import math

import numpy as np

from geometry.curves import Curve
from geometry.primitives import Point, Vector
from geometry.surfaces import Surface


class HoernerTip(Surface):
    """Hoerner-style wingtip cap as a parametric surface grown from a profile.

    The profile is a single closed section Curve, TE -> LE -> TE with the LE at
    parameter 0.5 (the ``WingSurface`` convention: ``wing.surface.iso_v(eta)``
    or ``wing.section_curve(eta)``). It is referenced directly, never copied,
    so the cap's root section S(u, 0) equals the profile EXACTLY -- ready for
    :class:`~geometry.surfaces.SewnSurface` stitching.

    Definition: the lower half (u >= 0.5) is the profile extruded along
    ``direction``; the upper half blends toward its chordwise partner
    ``profile(1 - u)`` with the quarter-ellipse law g(v) = sqrt(1 - v^2),
    collapsing onto the lower curve at v = 1: the sharp Hoerner edge.

    Args:
        profile: closed tip section curve (LE at parameter 0.5).
        direction: outboard build direction (normalized internally).
        length: spanwise extent of the cap along ``direction``.
        n_stations: number of spanwise stations for :meth:`stations`.
    """

    def __init__(self, profile: Curve, direction: Vector, length: float,
                 n_stations: int = 9):
        norm = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        if norm == 0.0 or length <= 0.0:
            raise ValueError("direction must be non-zero and length positive")
        self.profile = profile
        self._d = (direction.x / norm, direction.y / norm, direction.z / norm)
        self.length = float(length)
        self.n_stations = n_stations

    def point_at_parameter(self, u: float, v: float) -> Point:
        u, v = float(u), float(v)
        g = math.sqrt(max(0.0, 1.0 - v * v))
        ox, oy, oz = (c * self.length * v for c in self._d)
        if u >= 0.5:  # noqa: PLR2004  (lower surface: straight extrusion)
            p = self.profile.point_at_parameter(u)
            return Point(p.x + ox, p.y + oy, p.z + oz)
        p = self.profile.point_at_parameter(u)
        q = self.profile.point_at_parameter(1.0 - u)  # chordwise partner, lower side
        return Point(q.x + g * (p.x - q.x) + ox,
                     q.y + g * (p.y - q.y) + oy,
                     q.z + g * (p.z - q.z) + oz)

    def stations(self) -> list[float]:
        """Spanwise fractions v in [0, 1], uniform along the ellipse arc."""
        return [float(math.sin(a)) for a in np.linspace(0.0, math.pi / 2.0, self.n_stations)]

    def section_edges(self, v: float, n_half: int = 40) -> tuple[list[Point], list[Point]]:
        """(upper TE->LE, lower LE->TE) point curves at station v, for lofting.

        Matches the ``WingSurface.section_edges`` convention, so the same
        2-edge-wire solid loft works for wing and cap alike.
        """
        upper = [self.point_at_parameter(u, v) for u in np.linspace(0.0, 0.5, n_half)]
        lower = [self.point_at_parameter(u, v) for u in np.linspace(0.5, 1.0, n_half)]
        return upper, lower

    def section_loop(self, v: float, n_loop: int = 80) -> list[Point]:
        """Closed loop (TE -> LE -> TE) at station v, first point not repeated."""
        us = np.linspace(0.0, 1.0, n_loop, endpoint=False)
        return [self.point_at_parameter(u, v) for u in us]

    @property
    def sharp_edge(self) -> Curve:
        """The sharp lower edge at v = 1 (an iso-curve of this surface)."""
        return self.iso_v(1.0)
