"""Pure-geometry meshing: nodes, elements (connectedness), and spacing helpers."""
from geometry.mesh.element import Element
from geometry.mesh.mesh import Mesh
from geometry.mesh.node import Node
from geometry.mesh.planar import are_points_planar
from geometry.mesh.spacing import (
    nodes_cosine,
    nodes_exponential,
    nodes_exponential_dual,
    nodes_exponential_from_first,
    nodes_first_last_spacing,
    nodes_linspace,
    nodes_spacing_start_end_refinement,
)

__all__ = [
    "Node",
    "Element",
    "Mesh",
    "are_points_planar",
    "nodes_linspace",
    "nodes_cosine",
    "nodes_exponential",
    "nodes_exponential_dual",
    "nodes_first_last_spacing",
    "nodes_exponential_from_first",
    "nodes_spacing_start_end_refinement",
]
