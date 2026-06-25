"""Unstructured surface meshing via gmsh, returning a geometry.mesh.Mesh.

Run inside an active :class:`~geometry.occ.GmshSession`, after the model
geometry has been built and synchronized.
"""
from __future__ import annotations

import numpy as np

from geometry.mesh import Element, Mesh, Node

# gmsh element type id -> nodes per element (surface element types we keep).
_NODES_PER_ELEMENT = {2: 3, 3: 4, 9: 6, 10: 9, 16: 8}


def set_mesh_size(gmsh, size_max: float | None = None, size_min: float | None = None) -> None:
    """Set global min/max element size."""
    if size_max is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))
    if size_min is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))


def refine_near_curves(gmsh, curve_tags: list[int], size: float, distance: float) -> int:
    """Add a Distance+Threshold size field that refines near the given curves.

    Returns the threshold field id (set as background field). Useful to cluster
    elements around an intersection seam.
    """
    df = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(df, "CurvesList", list(curve_tags))
    gmsh.model.mesh.field.setNumber(df, "Sampling", 200)
    tf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(tf, "InField", df)
    gmsh.model.mesh.field.setNumber(tf, "SizeMin", size)
    gmsh.model.mesh.field.setNumber(tf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(tf, "DistMax", distance)
    gmsh.model.mesh.field.setAsBackgroundMesh(tf)
    return tf


def set_surface_algorithm(gmsh, face_tag: int, algorithm: int) -> None:
    """Set the 2-D meshing algorithm for a single face.

    Common values: 1=MeshAdapt, 5=Delaunay, 6=Frontal-Delaunay (default),
    8=Frontal-Delaunay for quads, 11=Quasi-structured quad.
    """
    gmsh.model.mesh.setAlgorithm(2, face_tag, algorithm)


def make_structured_quads(gmsh, face_tag: int, n_per_side: int, recombine: bool = True) -> None:
    """Mark a (4-sided) face for a structured (transfinite) mesh.

    Sets every bounding curve transfinite with ``n_per_side`` nodes, makes the
    face transfinite, and (optionally) recombines triangles into quads. The face
    must have 4 corner vertices — after a boolean cut a face usually won't, which
    is why the junction region is meshed unstructured instead.
    """
    boundary = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    for (dim, tag) in boundary:
        if dim == 1:
            gmsh.model.mesh.setTransfiniteCurve(abs(tag), n_per_side)
    gmsh.model.mesh.setTransfiniteSurface(face_tag)
    if recombine:
        gmsh.model.mesh.setRecombine(2, face_tag)


def set_transfinite_curve(gmsh, curve_tag: int, n_nodes: int,
                          mesh_type: str = "Progression", coef: float = 1.0) -> None:
    """Prescribe the node count/grading along a single curve."""
    gmsh.model.mesh.setTransfiniteCurve(curve_tag, n_nodes, meshType=mesh_type, coef=coef)


def set_curvature_sizing(gmsh, n_per_2pi: float = 20.0,
                         size_min: float | None = None, size_max: float | None = None,
                         from_points: bool = False, extend_from_boundary: bool = False) -> None:
    """Enable curvature-adaptive element sizing for unstructured faces.

    ``n_per_2pi`` is roughly the number of elements per full turn of curvature,
    so high-curvature regions (leading edges, nose) get finer elements. Has no
    effect on transfinite (structured) faces.

    Notes on the competing size sources:
        - ``from_points`` (``Mesh.MeshSizeFromPoints``): STEP files carry
          per-vertex characteristic lengths that fight the curvature field and
          cause patchy refinement. Keep this **off** for imported geometry.
        - ``extend_from_boundary`` (``Mesh.MeshSizeExtendFromBoundary``):
          interpolates sizes across each face, which *smooths* size transitions.
          This is gmsh's de-facto gradation control (it has no scalar
          "growth rate"); keep it **on** for smooth growth. If you still see
          patchiness, turn it off.
    """
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(n_per_2pi))
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if from_points else 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1 if extend_from_boundary else 0)
    if size_min is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))
    if size_max is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))


def make_structured_quads_uv(gmsh, face_tag: int, n_chord: int, n_span: int,
                             span_axis: int = 0, recombine: bool = True) -> bool:
    """Structured quad mesh on a 4-sided face, with separate chord/span counts.

    The face's four curves are paired into opposite sides; the pair extending
    most along ``span_axis`` (0=x, 1=y, 2=z) gets ``n_span`` nodes, the other
    gets ``n_chord``. Returns True on success, False if the face isn't 4-sided
    (so the caller can leave it unstructured).
    """
    boundary = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    curves = [abs(t) for (d, t) in boundary if d == 1]
    if len(curves) != 4:  # noqa: PLR2004
        return False

    def endpoints(c: int) -> set[int]:
        return {abs(t) for (d, t) in gmsh.model.getBoundary([(1, c)], oriented=False) if d == 0}

    ep = {c: endpoints(c) for c in curves}
    c0 = curves[0]
    opposite = [c for c in curves[1:] if ep[c].isdisjoint(ep[c0])]
    if len(opposite) != 1:
        return False  # not a clean quad topology
    pair_a = [c0, opposite[0]]
    pair_b = [c for c in curves if c not in pair_a]

    def span_extent(c: int) -> float:
        bb = gmsh.model.getBoundingBox(1, c)  # xmin,ymin,zmin,xmax,ymax,zmax
        return bb[3 + span_axis] - bb[span_axis]

    ext_a = sum(span_extent(c) for c in pair_a)
    ext_b = sum(span_extent(c) for c in pair_b)
    span_pair, chord_pair = (pair_a, pair_b) if ext_a >= ext_b else (pair_b, pair_a)

    for c in span_pair:
        gmsh.model.mesh.setTransfiniteCurve(c, n_span)
    for c in chord_pair:
        gmsh.model.mesh.setTransfiniteCurve(c, n_chord)
    gmsh.model.mesh.setTransfiniteSurface(face_tag)
    if recombine:
        gmsh.model.mesh.setRecombine(2, face_tag)
    return True


def generate_surface_mesh(
    gmsh, size_max: float | None = None, size_min: float | None = None
) -> Mesh:
    """Generate a 2-D mesh of the model and return it as a geometry Mesh."""
    set_mesh_size(gmsh, size_max, size_min)
    gmsh.model.mesh.generate(2)
    return extract_mesh(gmsh)


def extract_mesh(gmsh) -> Mesh:
    """Convert the current gmsh 2-D mesh into a geometry.mesh.Mesh."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_node: dict[int, Node] = {}
    nodes: list[Node] = []
    for tag, c in zip(node_tags, coords):
        node = Node(float(c[0]), float(c[1]), float(c[2]))
        tag_to_node[int(tag)] = node
        nodes.append(node)

    elements: list[Element] = []
    etypes, _etags, enodes = gmsh.model.mesh.getElements(dim=2)
    for etype, conn in zip(etypes, enodes):
        npe = _NODES_PER_ELEMENT.get(int(etype))
        if npe is None:
            continue
        conn = np.asarray(conn, dtype=int).reshape(-1, npe)
        for row in conn:
            elements.append(Element([tag_to_node[int(t)] for t in row]))
    return Mesh(nodes, elements)
