"""Wingtip caps grown from a WingSurface.

A tip is a Surface skinned over morphing airfoil sections beyond the wing's
cut station. It is built from the wing itself (same airfoil distribution,
chordwise stations and parameterization), so its root section equals the
wing's cut section exactly -- use ``wing.fit_tip(...)`` and
``wing.tipped_surface`` to get the sewn result.

All lengths are in WingShape units (pre-scale); the built surface is in the
wing's scaled coordinates. z is the wing-local "up".
"""
from __future__ import annotations

import math

import numpy as np

from geometry.aero.wing_surface import WingSurface, build_section_curve
from geometry.curves import InterpolatedLine
from geometry.surfaces import GordonSurface, Surface

#: Named blend laws f: [0,1] -> [0,1] with f(0)=0, f(1)=1. The seam is at v=0.
_BLENDS = {
    "linear": lambda v: v,                                   # C0: kinked seam
    "c1": lambda v: v * v,                                   # zero slope at seam
    "c2": lambda v: v**3,                                    # zero slope + curvature
    "smooth": lambda v: v * v * (3.0 - 2.0 * v),             # zero slope both ends
    "smoother": lambda v: v**3 * (10.0 - 15.0 * v + 6.0 * v * v),  # + curvature
    "elliptic": lambda v: 1.0 - math.sqrt(max(0.0, 1.0 - v * v)),  # C1 seam,
    # vertical tangent at the tip: the classic rounded closure.
}


def blend_function(spec):
    """Resolve a blend law: a name from ``_BLENDS`` or a callable f(0)=0, f(1)=1."""
    if isinstance(spec, str):
        try:
            return _BLENDS[spec]
        except KeyError:
            raise ValueError(f"unknown blend {spec!r}; choose from {sorted(_BLENDS)}") from None
    if abs(spec(0.0)) > 1e-9 or abs(spec(1.0) - 1.0) > 1e-9:
        raise ValueError("custom blend must satisfy f(0)=0 and f(1)=1")
    return spec


class WingTip(Surface):
    """Base wingtip cap: skins stations of morphing sections (Gordon).

    Subclasses implement ``_station_section(v)`` returning
    ``(upper, lower, chord, twist_rad, ref_offset_x, ref_offset_z)`` --
    airfoil-space y(x/c) callables plus placement -- for the station at cap
    fraction v in [0, 1]. Station v=0 must reproduce the wing's cut section
    (this makes the seam exact).

    Args:
        wing: the WingSurface this cap continues (prefer ``wing.fit_tip``).
        length: spanwise extent of the cap, WingShape units.
        n_stations: number of skinning stations.
    """

    def __init__(self, wing: WingSurface, length: float, n_stations: int = 9):
        semispan = float(wing.wing_shape.semispan)
        self.wing = wing
        self.length = float(length)
        self.eta_cut = 1.0 - self.length / semispan
        if not 0.0 < self.eta_cut < 1.0:
            raise ValueError("tip length must be a fraction of the semispan")
        self.n_stations = n_stations
        shape = wing.wing_shape
        self._root_airfoil = shape.airfoil_distribution(self.eta_cut)
        self._root_chord = float(shape.chord_distribution(self.eta_cut))
        self._root_twist = math.radians(float(shape.twist_distribution(self.eta_cut)))
        self._setup()
        self._build()

    def _setup(self) -> None:
        """Subclass hook run before the stations are skinned."""

    def _station_section(self, v: float):
        raise NotImplementedError

    def stations(self) -> list[float]:
        """Cap fractions v in [0, 1], uniform along the closing ellipse arc."""
        return [float(math.sin(a)) for a in np.linspace(0.0, math.pi / 2.0, self.n_stations)]

    def _build(self) -> None:
        wing, shape = self.wing, self.wing.wing_shape
        params = wing._chord_loop_u if wing.chordwise_spacing == "chord" else "arclength"
        vs = self.stations()
        profiles, le_pts, te_pts = [], [], []
        for v in vs:
            upper, lower, chord, twist_rad, dx, dz = self._station_section(v)
            span_y = self.eta_cut * float(shape.semispan) + v * self.length
            profile, le, te, _ = build_section_curve(
                wing._xs, upper, lower, chord, twist_rad, span_y,
                reference_chord_fraction=wing.reference_chord_fraction,
                scale=wing.scale, ref_offset_x=dx, ref_offset_z=dz, params=params,
            )
            profiles.append(profile)
            le_pts.append(le)
            te_pts.append(te)
        self._gordon = GordonSurface(
            profiles=profiles,
            guides=[InterpolatedLine(te_pts, params=vs),
                    InterpolatedLine(le_pts, params=vs),
                    InterpolatedLine(te_pts, params=vs)],
            profile_v_params=vs,
            guide_u_params=[0.0, 0.5, 1.0],
        )

    # -- Surface interface ---------------------------------------------------- #
    def point_at_parameter(self, u: float, v: float):
        return self._gordon.point_at_parameter(u, v)

    # -- conveniences ---------------------------------------------------------- #
    def section_edges(self, v: float, n_half: int = 40):
        """(upper TE->LE, lower LE->TE) sampled point curves at cap fraction v."""
        upper = [self.point_at_parameter(u, v) for u in np.linspace(0.0, 0.5, n_half)]
        lower = [self.point_at_parameter(u, v) for u in np.linspace(0.5, 1.0, n_half)]
        return upper, lower

    def section_loop(self, v: float, n_loop: int = 80):
        """Closed loop (TE -> LE -> TE) at cap fraction v, first point not repeated."""
        us = np.linspace(0.0, 1.0, n_loop, endpoint=False)
        return [self.point_at_parameter(u, v) for u in us]

    def loft_sections(self, n_half: int = 40, v_end: float = 0.999):
        """Station sections for an OCC solid loft.

        The last station stops just short of the zero-thickness tip: OCC's
        ThruSections cannot skin to a degenerate wire (and gmsh cannot
        tessellate the resulting face, which breaks viewer rendering).
        """
        vs = self.stations()
        vs[-1] = v_end
        return [self.section_edges(v, n_half) for v in vs]

    @property
    def sharp_edge(self):
        """The zero-thickness closing edge at v = 1 (an iso-curve of this surface)."""
        return self.iso_v(1.0)


class HoernerTip(WingTip):
    """Hoerner-style cap (Hoerner, *Fluid-Dynamic Drag*): the cap closes onto
    the root section's CAMBER LINE, raised to the height of the root section's
    upper surface -- both skins blend onto that zero-thickness edge with the
    ``blend`` law (default "elliptic"). Chord and twist are held at the cut.
    """

    def __init__(self, wing: WingSurface, length: float, n_stations: int = 9,
                 blend="elliptic"):
        self._f = blend_function(blend)
        super().__init__(wing, length, n_stations)

    def _setup(self) -> None:
        a = self._root_airfoil
        xs = np.linspace(0.0, 1.0, 101)
        upper = np.array([float(a.upper.sample(float(x))) for x in xs])
        camber = np.array([0.5 * (float(a.upper.sample(float(x))) + float(a.lower.sample(float(x))))
                           for x in xs])
        self._raise_dz = float(upper.max() - camber.max())

    def _station_section(self, v: float):
        a, dz = self._root_airfoil, self._raise_dz
        f = self._f(v)  # 0 at the cut, 1 at the tip

        def end(x: float) -> float:
            return 0.5 * (float(a.upper.sample(x)) + float(a.lower.sample(x))) + dz

        def upper(x: float, f=f) -> float:
            return (1.0 - f) * float(a.upper.sample(x)) + f * end(x)

        def lower(x: float, f=f) -> float:
            return (1.0 - f) * float(a.lower.sample(x)) + f * end(x)

        return upper, lower, self._root_chord, self._root_twist, 0.0, 0.0


class ParametricTip(WingTip):
    """Freely-shaped cap: continues the wing's airfoil distribution while
    collapsing thickness elliptically to zero, blending twist to zero,
    tapering the chord to ``tip_chord`` and moving the quarter-chord to
    (``delta_x``, ``delta_z``) at the very tip.

    Args (beyond WingTip's): all in WingShape units, wing-local axes --
        tip_chord: chord at the very tip.
        delta_x: tip quarter-chord shift, positive toward the TRAILING edge
            (sweep back). In an aircraft frame with x forward this is -x.
        delta_z: tip quarter-chord shift, positive UP (wing-local z).
        camber_scale: camber multiplier reached at the tip (1 = keep, 0 = flat).
        blends: per-quantity blend laws (name from ``_BLENDS`` or callable),
            keys: "chord", "twist", "offset", "camber", "thickness".
            Defaults: linear everywhere, "elliptic" for thickness. E.g.
            ``blends={"chord": "c1"}`` makes the chord leave the seam with
            zero slope (no taper kink).
    """

    DEFAULT_BLENDS = {
        "chord": "c2", "twist": "c2", "offset": "c2",
        "camber": "c2", "thickness": "c2",
    }

    def __init__(self, wing: WingSurface, length: float, tip_chord: float,
                 delta_x: float = 0.0, delta_z: float = 0.0,
                 camber_scale: float = 1.0, n_stations: int = 9,
                 blends: dict | None = None):
        self.tip_chord = float(tip_chord)
        self.delta_x = float(delta_x)
        self.delta_z = float(delta_z)
        self.camber_scale = float(camber_scale)
        spec = {**self.DEFAULT_BLENDS, **(blends or {})}
        if set(spec) != set(self.DEFAULT_BLENDS):
            unknown = set(spec) - set(self.DEFAULT_BLENDS)
            raise ValueError(f"unknown blend keys {sorted(unknown)}; "
                             f"valid: {sorted(self.DEFAULT_BLENDS)}")
        self._f = {key: blend_function(law) for key, law in spec.items()}
        super().__init__(wing, length, n_stations)

    def _station_section(self, v: float):
        shape = self.wing.wing_shape
        eta_v = self.eta_cut + v * (1.0 - self.eta_cut)
        a = shape.airfoil_distribution(eta_v)
        f = self._f
        tau = 1.0 - f["thickness"](v)                             # thickness fraction
        gam = 1.0 + (self.camber_scale - 1.0) * f["camber"](v)    # camber multiplier
        chord = self._root_chord + f["chord"](v) * (self.tip_chord - self._root_chord)
        twist_rad = (1.0 - f["twist"](v)) * self._root_twist      # zero at the tip

        def camber(x: float) -> float:
            return 0.5 * (float(a.upper.sample(x)) + float(a.lower.sample(x)))

        def upper(x: float, tau=tau, gam=gam) -> float:
            return gam * camber(x) + tau * (float(a.upper.sample(x)) - camber(x))

        def lower(x: float, tau=tau, gam=gam) -> float:
            return gam * camber(x) + tau * (float(a.lower.sample(x)) - camber(x))

        fo = f["offset"](v)
        return upper, lower, chord, twist_rad, self.delta_x * fo, self.delta_z * fo
