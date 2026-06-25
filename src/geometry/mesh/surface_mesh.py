"""Tessellate a parametric surface into a quad mesh.

Any object exposing ``point_at_parameter(u, v) -> Point`` (e.g.
:class:`~geometry.surfaces.GordonSurface` or
:class:`~geometry.aero.wing_surface.WingSurface`) can be sampled on a grid of
``u``/``v`` stations into a pure-geometry :class:`~geometry.mesh.mesh.Mesh` of
quadrilateral elements.
"""
from __future__ import annotations

from typing import Protocol

from geometry.mesh.element import Element
from geometry.mesh.mesh import Mesh
from geometry.mesh.node import Node
from geometry.primitives import Point


class _SurfaceLike(Protocol):
    def point_at_parameter(self, u: float, v: float) -> Point: ...


def mesh_surface(
    surface: _SurfaceLike,
    u_params: list[float],
    v_params: list[float],
    wrap_u: bool = False,
    wrap_v: bool = False,
) -> Mesh:
    """Sample a parametric surface into a quad mesh.

    Args:
        surface: Anything with ``point_at_parameter(u, v) -> Point``.
        u_params: Chordwise/first-direction parameter stations (in [0, 1]).
        v_params: Spanwise/second-direction parameter stations (in [0, 1]).
        wrap_u: If True, connect the last u-station back to the first (closed
            loop in u) without creating duplicate nodes.
        wrap_v: If True, connect the last v-station back to the first.

    Returns:
        A :class:`~geometry.mesh.mesh.Mesh` whose elements are 4-node quads,
        each ordered counter-clockwise as (u, v), (u+1, v), (u+1, v+1), (u, v+1).
    """
    us = list(u_params)
    vs = list(v_params)
    n_u, n_v = len(us), len(vs)
    if n_u < 2 or n_v < 2:  # noqa: PLR2004
        raise ValueError("Need at least 2 stations in each direction to form elements.")

    node_grid: list[list[Node]] = [
        [Node(*(float(c) for c in surface.point_at_parameter(u, v).as_array())) for v in vs]
        for u in us
    ]
    nodes = [node_grid[i][j] for i in range(n_u) for j in range(n_v)]

    i_max = n_u if wrap_u else n_u - 1
    j_max = n_v if wrap_v else n_v - 1

    elements: list[Element] = []
    for i in range(i_max):
        i2 = (i + 1) % n_u
        for j in range(j_max):
            j2 = (j + 1) % n_v
            elements.append(
                Element(
                    [
                        node_grid[i][j],
                        node_grid[i2][j],
                        node_grid[i2][j2],
                        node_grid[i][j2],
                    ]
                )
            )
    return Mesh(nodes, elements)
