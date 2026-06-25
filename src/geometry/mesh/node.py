"""Node primitive for meshes."""
from geometry.primitives import Point


class Node(Point):
    """Point used to represent a node in a mesh.

    A node is geometrically just a point in 3D space. Any analysis-specific
    state (degrees of freedom, loads, displacements) lives in the consuming
    layer (e.g. an FEM model), not here.
    """

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)
