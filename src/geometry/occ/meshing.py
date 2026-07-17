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
          interpolates sizes inward from each face's boundary edges. It can
          smooth transitions, but because it pulls the *finest* boundary size
          across the whole face it tends to over-refine flat faces (they never
          reach ``size_max``). Keep it **off** so curvature drives size per face
          (flat -> coarse, curved -> fine). For true bounded gradation, use a
          size field instead.
    """
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(n_per_2pi))
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if from_points else 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1 if extend_from_boundary else 0)
    if size_min is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(size_min))
    if size_max is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_max))


def make_structured_quads_uv(gmsh, face_tag: int, n_chord: int, n_span: int,
                             span_axis: int = 0, recombine: bool = True,
                             chord_law: str = "Progression", chord_coef: float = 1.0,
                             span_law: str = "Progression", span_coef: float = 1.0) -> bool:
    """Structured quad mesh on a 4-sided face, with per-direction node counts and laws.

    The four curves are paired into opposite sides; the pair extending most along
    ``span_axis`` (0=x, 1=y, 2=z) is the span direction (gets ``n_span`` nodes and
    ``span_law``/``span_coef``), the other is the chord direction.

    Distribution laws (gmsh transfinite ``meshType``):
        - "Progression": geometric; ``coef`` is the ratio between successive
          elements. coef>1 clusters toward one end, coef<1 toward the other
          (≈ one-sided exponential). coef=1 is uniform.
        - "Bump": clusters toward BOTH ends when ``coef`` < 1 (≈ dual-sided
          exponential / cosine); the smaller the coef the stronger the clustering.
        - "Beta": smooth two-sided clustering controlled by ``coef``.

    For an airfoil chord you typically want "Bump" with coef<1 (refine LE & TE);
    for the span, "Progression" with coef<1 to refine toward the root, or "Bump"
    to refine root and tip. Returns False if the face isn't a clean 4-sided patch.
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
        gmsh.model.mesh.setTransfiniteCurve(c, n_span, meshType=span_law, coef=span_coef)
    for c in chord_pair:
        gmsh.model.mesh.setTransfiniteCurve(c, n_chord, meshType=chord_law, coef=chord_coef)
    gmsh.model.mesh.setTransfiniteSurface(face_tag)
    if recombine:
        gmsh.model.mesh.setRecombine(2, face_tag)
    return True


def transfinite_positions(n: int, law: str = "Progression", coef: float = 1.0):
    """Return the [0,1] node positions a gmsh transfinite curve would produce.

    Implemented exactly for "Progression" (geometric) and uniform; provided so
    you can plot and tune a distribution before meshing. For "Bump"/"Beta",
    compare against your own ``geometry.mesh.spacing`` functions (e.g.
    ``nodes_exponential_dual``) which give the dual-sided clustering Bump mimics.
    """
    import numpy as np

    if law == "Progression" and abs(coef - 1.0) > 1e-12:  # noqa: PLR2004
        i = np.arange(n)
        return (coef**i - 1.0) / (coef ** (n - 1) - 1.0)
    return np.linspace(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Size fields (gmsh's mechanism for controlled, smooth size gradation)
# ---------------------------------------------------------------------------

def field_distance(gmsh, curves: list[int] | None = None,
                   surfaces: list[int] | None = None, sampling: int = 200) -> int:
    """A Distance field measuring distance to the given curves/surfaces."""
    f = gmsh.model.mesh.field.add("Distance")
    if curves:
        gmsh.model.mesh.field.setNumbers(f, "CurvesList", list(curves))
    if surfaces:
        gmsh.model.mesh.field.setNumbers(f, "SurfacesList", list(surfaces))
    gmsh.model.mesh.field.setNumber(f, "Sampling", sampling)
    return f


def field_threshold(gmsh, in_field: int, size_min: float, size_max: float,
                    dist_min: float, dist_max: float) -> int:
    """A Threshold field: size_min within dist_min, ramping to size_max by dist_max.

    The ramp slope (size_max - size_min)/(dist_max - dist_min) is the effective
    growth rate, so widening dist_max gives smoother, slower growth.
    """
    f = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f, "InField", in_field)
    gmsh.model.mesh.field.setNumber(f, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(f, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(f, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(f, "DistMax", dist_max)
    return f


def field_min(gmsh, fields: list[int]) -> int:
    """A Min field combining several fields (the finest size wins)."""
    f = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f, "FieldsList", list(fields))
    return f


def set_background_field(gmsh, field_id: int, disable_other_sources: bool = True) -> None:
    """Use ``field_id`` as the background mesh size field.

    With ``disable_other_sources`` the point sizes and boundary extension are
    turned off so the field is authoritative (curvature sizing, if enabled, is
    still combined by taking the minimum).
    """
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
    if disable_other_sources:
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


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


# ---------------------------------------------------------------------------
# Volume meshing with entity-linked mesh regions ("subgrids")
# ---------------------------------------------------------------------------

#: gmsh volume element type -> nodes per element (linear tets only for now).
_VOLUME_NODES_PER_ELEMENT = {4: 4}


class RegionMesh:
    """The part of a mesh lying on one or more geometry entities (a *subgrid*).

    Both ``nodes`` and ``facets`` reference the very same :class:`Node` objects
    as the parent mesh -- a facet is a new :class:`Element` over existing
    nodes. That is what makes layered modelling possible: a composite shell
    coating on a solid's face can be built directly on ``facets`` and shares
    its DOFs with the solid elements automatically (no coupling needed).

    Attributes:
        name: region label.
        nodes: mesh nodes on the entities (shared with the parent mesh).
        facets: boundary elements on the entities -- triangles/quads for faces,
            2-node segments for edges -- over the shared nodes.
    """

    def __init__(self, name: str, nodes: list[Node], facets: list[Element]):
        self.name = name
        self.nodes = nodes
        self.facets = facets

    def __repr__(self) -> str:
        return f"RegionMesh({self.name!r}, n_nodes={len(self.nodes)}, n_facets={len(self.facets)})"


class MeshedModel:
    """A mesh extracted from gmsh plus the tag bookkeeping to query regions.

    Returned by :func:`generate_volume_mesh`. ``mesh`` holds the volume
    elements; :meth:`region` resolves geometry entities (``geometry.occ.shapes``
    Faces/Edges/Solids or raw ``(dim, tag)`` pairs) to the mesh nodes and
    boundary facets that lie on them, sharing Node objects with ``mesh``.
    """

    def __init__(self, gmsh, mesh: Mesh, tag_to_node: dict[int, Node]):
        self._gmsh = gmsh
        self.mesh = mesh
        self._tag_to_node = tag_to_node

    def region(self, *entities, name: str = "region") -> RegionMesh:
        """The mesh nodes + facets on the given faces/edges (shared Node objects).

        Args:
            entities: ``geometry.occ.shapes`` Shape objects (Face, Edge, Solid)
                or raw gmsh ``(dim, tag)`` tuples.
            name: label for the region.
        """
        gmsh = self._gmsh
        dimtags = [e if isinstance(e, tuple) else e.dimtag for e in entities]

        node_ids: dict[int, Node] = {}
        facets: list[Element] = []
        for dim, tag in dimtags:
            tags, _, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True,
                                                  returnParametricCoord=False)
            for t in tags:
                node_ids.setdefault(int(t), self._tag_to_node[int(t)])
            if dim in (1, 2):
                per_type = {1: 2, 2: 3, 3: 4}  # segment, triangle, quad node counts
                etypes, _, enodes = gmsh.model.mesh.getElements(dim, tag)
                for etype, raw in zip(etypes, enodes):
                    npe = per_type.get(int(etype))
                    if npe is None:
                        continue
                    table = np.asarray(raw, dtype=int).reshape(-1, npe)
                    for row in table:
                        facets.append(Element([self._tag_to_node[int(t)] for t in row]))
        return RegionMesh(name, list(node_ids.values()), facets)


def generate_volume_mesh(
    gmsh, size_max: float | None = None, size_min: float | None = None
) -> MeshedModel:
    r"""Generate a 3-D (tet) mesh of the model and return it with region access.

    The returned :class:`MeshedModel` contains the volume mesh (linear
    tetrahedra as 4-node :class:`Element`\ s) and can resolve per-entity
    regions: gmsh meshes every face before filling the volume, so each face's
    triangles exist and conform to the tets -- ``model.region(face)`` returns
    them as facets over the *same* node objects.
    """
    set_mesh_size(gmsh, size_max, size_min)
    gmsh.model.mesh.generate(3)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_node: dict[int, Node] = {}
    nodes: list[Node] = []
    for tag, c in zip(node_tags, coords):
        node = Node(float(c[0]), float(c[1]), float(c[2]))
        tag_to_node[int(tag)] = node
        nodes.append(node)

    elements: list[Element] = []
    etypes, _etags, enodes = gmsh.model.mesh.getElements(dim=3)
    for etype, raw in zip(etypes, enodes):
        npe = _VOLUME_NODES_PER_ELEMENT.get(int(etype))
        if npe is None:
            continue
        table = np.asarray(raw, dtype=int).reshape(-1, npe)
        for row in table:
            elements.append(Element([tag_to_node[int(t)] for t in row]))
    if not elements:
        raise RuntimeError("volume meshing produced no tetrahedra")
    return MeshedModel(gmsh, Mesh(nodes, elements), tag_to_node)
