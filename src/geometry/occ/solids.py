"""Build capped solids by lofting section curves, and boolean-fuse them (gmsh OCC).

The robust route to a watertight, boolean-able part is: turn each cross-section
into a closed wire, loft them with ``addThruSections(makeSolid=True)`` (which
also caps the first/last sections), then ``fuse`` the parts so interior faces
are removed. The section curves come from our own geometry (e.g. WingSurface
airfoil loops), so the parametric design is preserved; OCC handles the skin and
the solid topology.

All functions take the ``gmsh`` module and run inside a
:class:`~geometry.occ.GmshSession`.
"""
from __future__ import annotations

from typing import Sequence

from geometry.primitives import Point


def closed_section_wire(gmsh, points: Sequence[Point], degree: int = 3) -> int:
    """Create a closed spline wire through an ordered loop of points.

    Uses ``gmsh.model.occ.addSpline``, which builds a curve that *interpolates*
    the given points (faithful to the sampled section); the loop is closed by
    repeating the first point.

    Args:
        points: loop points in order, NOT repeating the first point.
        degree: reserved (addSpline picks its own degree); kept for API stability.

    Returns:
        The wire tag.
    """
    occ = gmsh.model.occ
    pt_tags = [occ.addPoint(float(p.x), float(p.y), float(p.z)) for p in points]
    # Close the curve by repeating the first point.
    curve = occ.addSpline(pt_tags + [pt_tags[0]])
    return occ.addWire([curve])


def two_edge_section_wire(gmsh, upper: Sequence[Point], lower: Sequence[Point]) -> int:
    """Closed wire made of two edges (upper + lower), sharing LE and TE vertices.

    Lofting through such wires produces separate upper/lower faces (each 4-sided),
    which can then be meshed structured. ``upper`` runs TE->LE, ``lower`` runs
    LE->TE (as returned by ``WingSurface.section_edges``).
    """
    occ = gmsh.model.occ
    up = [occ.addPoint(float(p.x), float(p.y), float(p.z)) for p in upper]
    te, le = up[0], up[-1]  # upper: TE ... LE
    # lower interior points only; reuse LE and TE vertices so the wire closes.
    lo_interior = [occ.addPoint(float(p.x), float(p.y), float(p.z)) for p in lower[1:-1]]
    lower_tags = [le] + lo_interior + [te]
    e_upper = occ.addSpline(up)
    e_lower = occ.addSpline(lower_tags)
    return occ.addWire([e_upper, e_lower])


def loft_split_solid(gmsh, sections: Sequence[tuple[Sequence[Point], Sequence[Point]]],
                     make_solid: bool = True) -> int:
    """Loft a solid through 2-edge (upper/lower) sections; skin splits into faces.

    ``sections`` is a list of ``(upper, lower)`` point tuples. The resulting solid
    has separate upper and lower side faces (4-sided), suitable for structured
    meshing, plus end caps.
    """
    wires = [two_edge_section_wire(gmsh, up, lo) for (up, lo) in sections]
    return loft_solid(gmsh, wires, make_solid=make_solid)


def loft_solid(gmsh, wires: Sequence[int], make_solid: bool = True, ruled: bool = False) -> int:
    """Loft through closed wires into a (capped) solid; return the volume tag.

    Uses ``addThruSections``: with ``make_solid=True`` the first and last
    sections are capped, giving a closed solid.
    """
    out = gmsh.model.occ.addThruSections(
        list(wires), makeSolid=make_solid, makeRuled=ruled
    )
    # addThruSections returns the created entities; the volume is the dim-3 one.
    vols = [tag for (dim, tag) in out if dim == 3]
    if not vols:
        raise RuntimeError(f"addThruSections did not produce a solid: {out}")
    return vols[0]


def solid_from_section_loops(
    gmsh, loops: Sequence[Sequence[Point]], degree: int = 3, make_solid: bool = True
) -> int:
    """Convenience: build wires from point-loops and loft them into a solid."""
    wires = [closed_section_wire(gmsh, loop, degree=degree) for loop in loops]
    return loft_solid(gmsh, wires, make_solid=make_solid)


def fuse_solids(gmsh, solid_tags: Sequence[int]) -> int:
    """Boolean-union solids into one; interior faces are removed. Returns its tag."""
    tags = list(solid_tags)
    if len(tags) < 2:  # noqa: PLR2004
        raise ValueError("fuse_solids needs at least two solids.")
    current = [(3, tags[0])]
    for t in tags[1:]:
        out, _ = gmsh.model.occ.fuse(current, [(3, t)])
        current = out
    gmsh.model.occ.synchronize()
    return current[0][1]


def wing_section_loops(wing_surface, n_loop: int = 80) -> list[list[Point]]:
    """Sample a WingSurface's airfoil sections into closed point-loops.

    Each profile (the closed TE->LE->TE airfoil curve) is sampled at ``n_loop``
    points with the trailing-edge endpoint dropped (so the loop does not repeat
    a point and can be closed cleanly).
    """
    import numpy as np

    us = np.linspace(0.0, 1.0, n_loop, endpoint=False)
    loops: list[list[Point]] = []
    for profile in wing_surface.profiles:
        loops.append([profile.point_at_parameter(float(u)) for u in us])
    return loops
