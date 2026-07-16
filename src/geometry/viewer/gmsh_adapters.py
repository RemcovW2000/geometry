"""Viewer adapters for the gmsh OCC shape API (Solid / Face / Edge / Vertex).

Shapes are displayed through gmsh's own mesher: once per scene build a
curvature-adaptive *display* mesh is generated for the whole model, and each
face/edge extracts its triangles/segments from it. The tree mirrors the CAD
topology::

    Solid
    ├─ faces
    │   ├─ Face  (triangles)
    │   │   └─ edges → Edge (polyline) → vertices → Vertex (point)
    │   └─ ...
    └─ ...

so clicking an edge in the browser tells you exactly which face/solid it
belongs to -- no more raw (dim, tag) bookkeeping.
"""
from __future__ import annotations

import numpy as np

from geometry.occ.session import ensure_session
from geometry.occ.shapes import Edge, Face, Shape, Solid, Vertex
from geometry.viewer.base import (
    ViewNode,
    mesh_payload,
    points_payload,
    polyline_payload,
    register_adapter,
    register_scene_hook,
    segments_payload,
)

FACE_COLOR = "#8fa8bf"
EDGE_COLOR = "#e8b34b"
VERTEX_COLOR = "#e06666"

_tessellated = False


def _reset_tessellation() -> None:
    global _tessellated  # noqa: PLW0603  (per-scene cache flag)
    _tessellated = False


register_scene_hook(_reset_tessellation)


def _ensure_display_mesh() -> None:
    """Generate a curvature-adaptive 2D mesh of the whole model, once per scene."""
    global _tessellated  # noqa: PLW0603  (per-scene cache flag)
    if _tessellated:
        return
    gmsh = ensure_session()
    gmsh.model.occ.synchronize()
    # Size the display mesh from the model extent + curvature.
    bounds = np.array([gmsh.model.getBoundingBox(dim, tag)
                       for dim, tag in gmsh.model.getEntities(3)]
                      or [gmsh.model.getBoundingBox(dim, tag)
                          for dim, tag in gmsh.model.getEntities(2)] or [[0, 0, 0, 1, 1, 1]])
    mins, maxs = bounds[:, :3].min(axis=0), bounds[:, 3:].max(axis=0)
    diag = float(np.linalg.norm(maxs - mins)) or 1.0
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 32)
    gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 25.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 400.0)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(2)
    _tessellated = True


def _entity_mesh(dim: int, tag: int, element_type: int, nodes_per_element: int):
    """(coords (N,3), connectivity (M, npe)) of one entity's display mesh."""
    gmsh = ensure_session()
    node_tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True,
                                                    returnParametricCoord=False)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}
    types, _, connectivity = gmsh.model.mesh.getElements(dim, tag)
    for etype, conn in zip(types, connectivity):
        if etype == element_type:
            local = np.array([index[int(t)] for t in conn], dtype=int)
            return coords, local.reshape(-1, nodes_per_element)
    return coords, np.zeros((0, nodes_per_element), dtype=int)


def _face_geoms(face: Face) -> list[dict]:
    _ensure_display_mesh()
    try:
        coords, tris = _entity_mesh(2, face.tag, element_type=2, nodes_per_element=3)
    except Exception:  # noqa: BLE001  (unmeshable face: show nothing rather than crash)
        return []
    if len(tris) == 0:
        return []
    return [mesh_payload(coords, tris, FACE_COLOR)]


def _edge_geoms(edge: Edge) -> list[dict]:
    _ensure_display_mesh()
    try:
        coords, segs = _entity_mesh(1, edge.tag, element_type=1, nodes_per_element=2)
        if len(segs):
            return [segments_payload(coords, segs, EDGE_COLOR)]
    except Exception:  # noqa: BLE001
        pass
    try:
        return [polyline_payload(edge.sample(50), EDGE_COLOR)]
    except Exception:  # noqa: BLE001
        return []


def _vertex_node(vertex: Vertex) -> ViewNode:
    p = vertex.point
    return ViewNode(
        label=f"{vertex.name} ({p.x:.3g}, {p.y:.3g}, {p.z:.3g})", kind="Vertex",
        geoms=[points_payload([p], VERTEX_COLOR)],
        meta={"tag": vertex.tag},
    )


def _edge_node(edge: Edge) -> ViewNode:
    return ViewNode(
        label=edge.name, kind="Edge",
        geoms=_edge_geoms(edge),
        meta={"tag": edge.tag, "type": edge.kind, "length": round(edge.length, 6)},
        children=[ViewNode(label="vertices", kind="group",
                           children=[_vertex_node(v) for v in edge.vertices])],
    )


def _face_node(face: Face) -> ViewNode:
    return ViewNode(
        label=face.name, kind="Face",
        geoms=_face_geoms(face),
        meta={"tag": face.tag, "type": face.kind, "area": round(face.area, 6)},
        children=[ViewNode(label="edges", kind="group",
                           children=[_edge_node(e) for e in face.edges])],
    )


def _solid_node(solid: Solid) -> ViewNode:
    return ViewNode(
        label=solid.name, kind="Solid",
        meta={"tag": solid.tag, "volume": round(solid.volume, 6)},
        children=[ViewNode(label="faces", kind="group",
                           children=[_face_node(f) for f in solid.faces])],
    )


# Later registrations win, so the generic fallback goes first.
register_adapter(Shape, _solid_node)
register_adapter(Vertex, _vertex_node)
register_adapter(Edge, _edge_node)
register_adapter(Face, _face_node)
register_adapter(Solid, _solid_node)
