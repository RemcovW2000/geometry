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
