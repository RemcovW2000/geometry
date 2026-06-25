"""Pure-geometry mesh element.

An element describes the connectedness of a set of nodes and nothing more.
It carries no physics: no stiffness, no material, no degrees of freedom.
Analysis layers (e.g. FEM) compose an :class:`Element` to describe an
element's geometry and add their own behaviour on top.
"""
from __future__ import annotations

import numpy as np

from geometry.mesh.node import Node
from geometry.primitives import Point


class Element:
    """The connectedness of an ordered set of nodes."""

    def __init__(self, nodes: list[Node]):
        if len(nodes) == 0:
            raise ValueError("An element must reference at least one node.")
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"Element(n_nodes={len(self.nodes)})"

    @property
    def center_point(self) -> Point:
        """Return the centroid of the element's nodes."""
        n = len(self.nodes)
        return Point(
            sum(node.x for node in self.nodes) / n,
            sum(node.y for node in self.nodes) / n,
            sum(node.z for node in self.nodes) / n,
        )

    def node_coordinates(self) -> np.ndarray:
        """Return an (n_nodes, 3) array of the element's node coordinates."""
        return np.array([node.as_array() for node in self.nodes], dtype=float)
