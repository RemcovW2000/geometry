"""Couple a geometry mesh to the viewer's hierarchy, picking and rendering.

A :class:`Scene` builds the hierarchy, precomputes pickable geometry, resolves a
click to a hierarchy node, and (lazily, only if PyVista is installed) builds the
render primitives.

The split matters: everything except :meth:`Scene.build_polydata` is pure
NumPy/Python and unit-testable without a display or PyVista. That is what keeps
the pick-to-hierarchy contract cheap to verify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.mesh.mesh import Mesh
from geometry.viewer.hierarchy import Hierarchy, HierarchyNode
from geometry.viewer.ids import SceneId


@dataclass
class RenderData:
    """PyVista primitives plus the id lists that align with their cells/points.

    Attributes:
        faces: PolyData of element faces; cell ``i`` <-> ``face_ids[i]``.
        edges: PolyData of element edges; line cell ``i`` <-> ``edge_ids[i]``.
        nodes: PolyData of nodes; point ``i`` <-> ``node_ids[i]``.
    """

    faces: object
    edges: object
    nodes: object
    face_ids: list[SceneId]
    edge_ids: list[SceneId]
    node_ids: list[SceneId]


class Scene:
    """Bind a mesh to a pickable hierarchy and (optionally) a 3D rendering.

    Args:
        mesh: The mesh to view.
        name: Label for the mesh node in the hierarchy.
        surface: Optional parametric surface the mesh came from; if given it
            becomes the root and the mesh hangs beneath it.
        surface_name: Label for the surface node when ``surface`` is provided.
    """

    def __init__(
        self,
        mesh: Mesh,
        name: str = "mesh",
        surface: object = None,
        surface_name: str = "surface",
    ):
        self.mesh = mesh
        self.name = name
        self.surface = surface
        self.surface_name = surface_name

        self._node_index = {id(node): i for i, node in enumerate(mesh.nodes)}
        self.hierarchy = self._build_hierarchy()
        self._precompute_pick_geometry()

    # -- hierarchy ---------------------------------------------------------
    def _build_hierarchy(self) -> Hierarchy:
        mesh_id = SceneId("mesh", (self.name,))
        mesh_node = HierarchyNode(id=mesh_id, label=self.name, obj=self.mesh)

        if self.surface is not None:
            root = HierarchyNode(
                id=SceneId("surface", (self.surface_name,)),
                label=self.surface_name,
                obj=self.surface,
            )
            mesh_node.parent = root
            root.children.append(mesh_node)
        else:
            root = mesh_node

        for e_idx, element in enumerate(self.mesh.elements):
            elem_node = mesh_node.add_child(
                id=SceneId("element", (e_idx,)),
                label=f"Element {e_idx}",
                obj=element,
            )
            n = len(element.nodes)
            for local in range(n):
                a = element.nodes[local]
                b = element.nodes[(local + 1) % n]
                ai = self._node_index[id(a)]
                bi = self._node_index[id(b)]
                elem_node.add_child(
                    id=SceneId("edge", (e_idx, local)),
                    label=f"Edge n{ai}–n{bi}",
                    obj=(a, b),
                    metadata={"node_indices": (ai, bi)},
                )

        nodes_group = mesh_node.add_child(
            id=SceneId("group", (self.name, "nodes")),
            label="Nodes",
        )
        for i, node in enumerate(self.mesh.nodes):
            nodes_group.add_child(
                id=SceneId("node", (i,)),
                label=f"Node {i}",
                obj=node,
            )

        return Hierarchy(root)

    # -- pick geometry (pure numpy) ----------------------------------------
    def _precompute_pick_geometry(self) -> None:
        coords = np.array([n.as_array() for n in self.mesh.nodes], dtype=float)
        self._node_coords = coords
        self._node_ids = [SceneId("node", (i,)) for i in range(len(coords))]

        edge_mids: list[np.ndarray] = []
        edge_ids: list[SceneId] = []
        edge_lengths: list[float] = []
        elem_centroids: list[np.ndarray] = []
        elem_ids: list[SceneId] = []
        for e_idx, element in enumerate(self.mesh.elements):
            pts = element.node_coordinates()
            elem_centroids.append(pts.mean(axis=0))
            elem_ids.append(SceneId("element", (e_idx,)))
            n = len(element.nodes)
            for local in range(n):
                a = pts[local]
                b = pts[(local + 1) % n]
                edge_mids.append(0.5 * (a + b))
                edge_lengths.append(float(np.linalg.norm(b - a)))
                edge_ids.append(SceneId("edge", (e_idx, local)))

        self._edge_mids = np.array(edge_mids, dtype=float).reshape(-1, 3)
        self._edge_ids = edge_ids
        self._elem_centroids = np.array(elem_centroids, dtype=float).reshape(-1, 3)
        self._elem_ids = elem_ids
        # Characteristic length used to decide node vs edge vs element on a click.
        self._h = float(np.median(edge_lengths)) if edge_lengths else 1.0

    def resolve_click(
        self, point: np.ndarray | tuple[float, float, float], snap: float = 0.25
    ) -> HierarchyNode:
        """Resolve a picked 3D point to the nearest pickable hierarchy node.

        A click within ``snap`` characteristic lengths of a node snaps to that
        node; otherwise within ``snap`` of an edge midpoint snaps to that edge;
        otherwise it falls to the element with the nearest centroid.

        Args:
            point: The picked world-space coordinate.
            snap: Snap radius as a fraction of the median edge length.

        Returns:
            The resolved :class:`HierarchyNode`.
        """
        p = np.asarray(point, dtype=float).reshape(3)
        tol = snap * self._h

        d_node, i_node = self._nearest(self._node_coords, p)
        if d_node <= tol:
            return self.hierarchy.node_for_id(self._node_ids[i_node])

        d_edge, i_edge = self._nearest(self._edge_mids, p)
        if d_edge <= tol:
            return self.hierarchy.node_for_id(self._edge_ids[i_edge])

        _, i_elem = self._nearest(self._elem_centroids, p)
        return self.hierarchy.node_for_id(self._elem_ids[i_elem])

    @staticmethod
    def _nearest(coords: np.ndarray, p: np.ndarray) -> tuple[float, int]:
        if coords.size == 0:
            return float("inf"), -1
        d = np.linalg.norm(coords - p, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), i

    def describe(self, node: HierarchyNode) -> str:
        """Return a human-readable one-line description of a resolved node."""
        return f"{node.label}  [{self.hierarchy.path_str(node.id)}]"

    # -- rendering (needs PyVista) -----------------------------------------
    def build_polydata(self) -> RenderData:
        """Build PyVista primitives for faces, edges and nodes.

        Raises:
            ImportError: If PyVista is not installed (``pip install
                'geometry[viewer]'``).
        """
        try:
            import pyvista as pv  # noqa: PLC0415  (optional 'viewer' extra)
        except ImportError as exc:  # pragma: no cover - exercised via extra
            raise ImportError(
                "The viewer needs PyVista. Install with: pip install 'geometry[viewer]'"
            ) from exc

        coords = self._node_coords

        # Faces: one polygon cell per element (VTK face format: [n, i0, i1, ...]).
        face_cells: list[int] = []
        face_ids: list[SceneId] = []
        for e_idx, element in enumerate(self.mesh.elements):
            idxs = [self._node_index[id(n)] for n in element.nodes]
            face_cells.append(len(idxs))
            face_cells.extend(idxs)
            face_ids.append(SceneId("element", (e_idx,)))
        faces = pv.PolyData(coords, faces=np.array(face_cells))
        faces.cell_data["pick"] = np.arange(len(face_ids), dtype=np.int64)

        # Edges: one line cell per element edge (VTK line format: [2, a, b]).
        line_cells: list[int] = []
        edge_ids: list[SceneId] = []
        for e_idx, element in enumerate(self.mesh.elements):
            idxs = [self._node_index[id(n)] for n in element.nodes]
            n = len(idxs)
            for local in range(n):
                line_cells.extend((2, idxs[local], idxs[(local + 1) % n]))
                edge_ids.append(SceneId("edge", (e_idx, local)))
        edges = pv.PolyData(coords, lines=np.array(line_cells))
        edges.cell_data["pick"] = np.arange(len(edge_ids), dtype=np.int64)

        # Nodes: a point cloud.
        nodes = pv.PolyData(coords)
        nodes.point_data["pick"] = np.arange(len(coords), dtype=np.int64)

        return RenderData(
            faces=faces,
            edges=edges,
            nodes=nodes,
            face_ids=face_ids,
            edge_ids=edge_ids,
            node_ids=list(self._node_ids),
        )
