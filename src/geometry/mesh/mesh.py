"""Pure-geometry mesh: a collection of nodes and elements.

This is the abstract, physics-free notion of a mesh. It knows which nodes
exist and how they are connected into elements, and offers basic spatial
queries and visualisation. Analysis layers (e.g. an FEM model) build on top
of this by subclassing or composing it.
"""
from __future__ import annotations

import numpy as np

from geometry.mesh.element import Element
from geometry.mesh.node import Node
from geometry.primitives import Point


class Mesh:
    """A collection of nodes and the elements that connect them."""

    def __init__(self, nodes: list[Node], elements: list[Element]):
        self.nodes = nodes
        self.elements = elements
        self.node_id_map = {node: i for i, node in enumerate(nodes)}

    def __repr__(self) -> str:
        return f"Mesh(n_nodes={len(self.nodes)}, n_elements={len(self.elements)})"

    def node_index(self, node: Node) -> int:
        """Return the global index of a node in this mesh."""
        return self.node_id_map[node]

    def find_node_at_point(self, point: Point, tol: float = 1e-6) -> Node:
        """Find the node located at the given point coordinates."""
        return next(node for node in self.nodes if node.is_at_point(point, tol))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mins, maxs) coordinate arrays spanning all nodes."""
        pts = np.array([node.as_array() for node in self.nodes], dtype=float)
        return pts.min(axis=0), pts.max(axis=0)

    def visualize(self) -> None:
        """Visualize the (undeformed) mesh by plotting element outlines."""
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        all_points = []
        for element in self.elements:
            xyz = np.array([n.as_array() for n in element.nodes])
            all_points.append(xyz)
            closed = np.vstack((xyz, xyz[0]))
            ax.plot(*closed.T, "k--", alpha=0.4)

        pts = np.vstack(all_points)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        centers = 0.5 * (mins + maxs)
        max_half = 0.5 * (maxs - mins).max()
        pad = max_half * 0.05 if max_half > 0 else 1.0
        cx, cy, cz = centers
        r = max_half + pad
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Mesh")
        plt.show()
