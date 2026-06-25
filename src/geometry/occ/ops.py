"""CAD operations on OCC entities via gmsh: boolean/fragment, curves, STEP I/O.

All functions take the ``gmsh`` module and must run inside an active
:class:`~geometry.occ.GmshSession`.
"""
from __future__ import annotations

import numpy as np

from geometry.curves import InterpolatedLine
from geometry.primitives import Point


def fragment(gmsh, dimtags_a: list[tuple[int, int]], dimtags_b: list[tuple[int, int]]):
    """Boolean-fragment two sets of entities (e.g. ``[(2, wing)]``, ``[(2, fuse)]``).

    Fragmenting intersects all inputs and splits them along their intersection
    curves into a single coherent (conformal) model: shared curves become shared
    topology, so a later surface mesh is conformal across the junction.

    Returns the gmsh ``(out_dimtags, mapping)`` pair and synchronizes the model.
    """
    out, mapping = gmsh.model.occ.fragment(dimtags_a, dimtags_b)
    gmsh.model.occ.synchronize()
    return out, mapping


def export_step(gmsh, path: str) -> None:
    """Write the current model to STEP (use a .step/.stp extension)."""
    gmsh.write(str(path))


def import_step(gmsh, path: str):
    """Import a STEP/IGES/BREP file and synchronize; returns imported dimtags."""
    tags = gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    return tags


def curve_tags(gmsh) -> list[int]:
    """Tags of all curve (dim-1) entities currently in the model."""
    return [tag for (dim, tag) in gmsh.model.getEntities(1)]


def sample_curve(gmsh, tag: int, n: int = 60) -> InterpolatedLine:
    """Sample a model curve into an :class:`InterpolatedLine`."""
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    ts = np.linspace(lo[0], hi[0], n)
    pts = [Point(*gmsh.model.getValue(1, tag, [float(t)])) for t in ts]
    return InterpolatedLine(pts)


def intersection_curves(gmsh, face_a: int, face_b: int, n: int = 60) -> list[InterpolatedLine]:
    """Return the curves shared by two faces (their intersection after fragment).

    After :func:`fragment`, the intersection shows up as curves that appear on
    the boundary of *both* faces. This finds those shared curves and samples
    them. Call after fragmenting and synchronizing.
    """
    def boundary_curves(face: int) -> set[int]:
        b = gmsh.model.getBoundary([(2, face)], oriented=False, recursive=False)
        return {abs(tag) for (dim, tag) in b if dim == 1}

    shared = boundary_curves(face_a) & boundary_curves(face_b)
    return [sample_curve(gmsh, tag, n) for tag in sorted(shared)]
