"""CAD operations on OCC entities via gmsh: boolean/fragment, curves, STEP I/O.

All functions take the ``gmsh`` module and must run inside an active
:class:`~geometry.occ.GmshSession`.
"""
from __future__ import annotations

import numpy as np

from geometry.curves import InterpolatedLine
from geometry.primitives import Point


def apply_position(gmsh, dimtags: list[tuple[int, int]], position) -> None:
    """Rigidly place entities using a geometry :class:`~geometry.primitives.Position`.

    The Position's orientation (rotation, local->global) and origin (translation)
    are converted to an OCC affine transform, so you can place a wing on a
    fuselage with the same Position/Orientation API you use everywhere else::

        from geometry import Orientation, Point, Position, Vector
        ident = Orientation(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))
        # span Y -> +X, root at the origin:
        pos = Position(Point(0, 0, 0), ident.rotate(-math.pi / 2, Vector(0, 0, 1)))
        apply_position(gmsh, [(3, wing_solid)], pos)

    Applies ``global = R @ local + origin`` (matching ``Position.point_in_global``).
    """
    r = position.orientation.as_matrix()
    o = position.origin.as_array()
    affine = [
        r[0, 0], r[0, 1], r[0, 2], o[0],
        r[1, 0], r[1, 1], r[1, 2], o[1],
        r[2, 0], r[2, 1], r[2, 2], o[2],
    ]
    gmsh.model.occ.affineTransform(list(dimtags), [float(v) for v in affine])


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


def export_mesh(gmsh, path: str, binary: bool = False) -> None:
    """Write the current mesh to ``path``; the format is chosen by extension.

    gmsh writes many mesh formats by extension, e.g.:
        .stl   triangulated surface  -> FlightStream / OpenVSP import
               (quads are split to triangles; use ASCII for OpenVSP)
        .tri   Cart3D triangulation  -> OpenVSP
        .vtk   ParaView / FlightStream results viewing (preserves quads)
        .bdf   Nastran (preserves quads), .msh native gmsh, .su2, .unv, ...

    Args:
        path: output filename (extension selects the format).
        binary: write binary where supported. Keep ``False`` (ASCII) for STL
            going into OpenVSP, which is more reliable per its docs.
    """
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    gmsh.write(str(path))


def import_step(gmsh, path: str):
    """Import a STEP/IGES/BREP file and synchronize; returns imported dimtags."""
    tags = gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    return tags


def curve_tags(gmsh) -> list[int]:
    """Tags of all curve (dim-1) entities currently in the model."""
    return [tag for (dim, tag) in gmsh.model.getEntities(1)]


def solid_tags(gmsh) -> list[int]:
    """Tags of all volume (dim-3) entities."""
    return [tag for (dim, tag) in gmsh.model.getEntities(3)]


def face_tags(gmsh) -> list[int]:
    """Tags of all surface (dim-2) entities."""
    return [tag for (dim, tag) in gmsh.model.getEntities(2)]


def face_centroid(gmsh, tag: int) -> np.ndarray:
    """Center of mass of a face (after synchronize)."""
    return np.array(gmsh.model.occ.getCenterOfMass(2, tag), dtype=float)


def n_boundary_curves(gmsh, face_tag: int) -> int:
    """Number of curves bounding a face (4 for a clean quad patch)."""
    b = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    return sum(1 for (d, t) in b if d == 1)


def describe_faces(gmsh, span_axis: int = 0, outboard_min: float | None = None) -> list[dict]:
    """Print a summary of all faces (centroid, #curves, bbox) for debugging.

    Returns a list of per-face dicts. ``outboard_min`` (if given) reports how
    many faces lie beyond it along ``span_axis`` and how many of those are
    4-sided (i.e. eligible for structured meshing).
    """
    import numpy as np

    info = []
    for tag in face_tags(gmsh):
        cen = face_centroid(gmsh, tag)
        nbc = n_boundary_curves(gmsh, tag)
        bb = gmsh.model.getBoundingBox(2, tag)
        info.append({"tag": tag, "centroid": cen, "n_curves": nbc, "bbox": bb})

    hist: dict[int, int] = {}
    for f in info:
        hist[f["n_curves"]] = hist.get(f["n_curves"], 0) + 1
    cen_axis = np.array([f["centroid"][span_axis] for f in info])
    print(f"[describe_faces] {len(info)} faces; #curves histogram: "
          f"{dict(sorted(hist.items()))}")
    print(f"[describe_faces] centroid[axis {span_axis}] range: "
          f"{cen_axis.min():.1f} .. {cen_axis.max():.1f}")
    if outboard_min is not None:
        beyond = [f for f in info if f["centroid"][span_axis] > outboard_min]
        quad = [f for f in beyond if f["n_curves"] == 4]  # noqa: PLR2004
        print(f"[describe_faces] beyond {outboard_min} on axis {span_axis}: "
              f"{len(beyond)} faces, of which 4-sided: {len(quad)}")
        for f in beyond[:12]:
            print(f"    tag={f['tag']} n_curves={f['n_curves']} "
                  f"centroid={np.round(f['centroid'], 1)}")
    return info


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
