"""Built-in viewer adapters for the pure-Python geometry types.

Each adapter maps one geometry class to a :class:`~geometry.viewer.base.ViewNode`
subtree: what it looks like (render payloads) and what it contains (children).
The hierarchy follows the geometry's own structure, e.g. a GordonSurface node
contains its profile and guide curves, and each curve contains its defining
points.
"""
from __future__ import annotations

import numpy as np

from geometry.curves import Curve, InterpolatedLine
from geometry.mesh.mesh import Mesh
from geometry.primitives import Point
from geometry.surfaces import GordonPatch, GordonSurface
from geometry.viewer.base import (
    ViewNode,
    grid_payload,
    points_payload,
    polyline_payload,
    register_adapter,
    segments_payload,
)

SURFACE_COLOR = "#7aa2c9"
CURVE_COLOR = "#e8b34b"
POINT_COLOR = "#e06666"
MESH_COLOR = "#9dbd8f"


def _point_node(point: Point) -> ViewNode:
    label = f"Point({point.x:.3g}, {point.y:.3g}, {point.z:.3g})"
    return ViewNode(label=label, kind="Point", geoms=[points_payload([point], POINT_COLOR)])


def _curve_node(curve: Curve, n: int = 100) -> ViewNode:
    us = np.linspace(0.0, 1.0, n)
    pts = [curve.point_at_parameter(float(u)) for u in us]
    children = []
    defining = getattr(curve, "points", None)
    if defining:
        children.append(ViewNode(
            label="points", kind="group",
            children=[_point_node(p) for p in defining],
        ))
    return ViewNode(
        label=type(curve).__name__, kind=type(curve).__name__,
        geoms=[polyline_payload(pts, CURVE_COLOR)],
        children=children,
    )


def _gordon_patch_node(patch: GordonPatch) -> ViewNode:
    children = [
        ViewNode(label="u_lines", kind="group",
                 children=[_curve_node(c) for c in patch.u_lines]),
        ViewNode(label="v_lines", kind="group",
                 children=[_curve_node(c) for c in patch.v_lines]),
    ]
    return ViewNode(
        label="GordonPatch", kind="GordonPatch",
        geoms=[grid_payload(patch.sample_grid(30, 30), SURFACE_COLOR)],
        children=children,
    )


def _gordon_surface_node(surface: GordonSurface) -> ViewNode:
    children = [
        ViewNode(label="profiles", kind="group",
                 children=[_curve_node(c) for c in surface.profiles]),
        ViewNode(label="guides", kind="group",
                 children=[_curve_node(c) for c in surface.guides]),
    ]
    return ViewNode(
        label="GordonSurface", kind="GordonSurface",
        geoms=[grid_payload(surface.sample_grid(40, 40), SURFACE_COLOR)],
        meta={"n_profiles": surface.n_profiles, "n_guides": surface.n_guides},
        children=children,
    )


def _mesh_node(mesh: Mesh) -> ViewNode:
    coords = np.array([node.as_array() for node in mesh.nodes], dtype=float)
    triangles: list[tuple[int, int, int]] = []
    segments: list[tuple[int, int]] = []
    for element in mesh.elements:
        idx = [mesh.node_index(n) for n in element.nodes]
        for k in range(1, len(idx) - 1):  # fan-triangulate 3/4+-node elements
            triangles.append((idx[0], idx[k], idx[k + 1]))
        for k in range(len(idx)):
            segments.append((idx[k], idx[(k + 1) % len(idx)]))
    geoms = []
    if triangles:
        from geometry.viewer.base import mesh_payload  # noqa: PLC0415

        geoms.append(mesh_payload(coords, np.array(triangles, dtype=int), MESH_COLOR))
    if segments:
        geoms.append(segments_payload(coords, np.array(segments, dtype=int), "#3c4753"))
    node = ViewNode(
        label=f"Mesh ({len(mesh.nodes)} nodes, {len(mesh.elements)} elements)",
        kind="Mesh",
        geoms=geoms,
        meta={"n_nodes": len(mesh.nodes), "n_elements": len(mesh.elements)},
        children=[ViewNode(label="nodes", kind="group",
                           geoms=[points_payload(coords, POINT_COLOR)])],
    )
    return node


def _wing_surface_node(wing) -> ViewNode:
    children = []
    profiles = getattr(wing, "profiles", None)
    if profiles:
        children.append(ViewNode(label="profiles", kind="group",
                                 children=[_curve_node(c) for c in profiles]))
    return ViewNode(
        label="WingSurface", kind="WingSurface",
        geoms=[grid_payload(wing.sample_grid(60, 30), SURFACE_COLOR)],
        children=children,
    )


def _is_wing_surface(obj) -> bool:
    try:
        from geometry.aero.wing_surface import WingSurface  # noqa: PLC0415  (optional)

        return isinstance(obj, WingSurface)
    except ImportError:  # pragma: no cover
        return False


register_adapter(Point, _point_node)
register_adapter(Curve, _curve_node)
register_adapter(InterpolatedLine, _curve_node)
register_adapter(GordonPatch, _gordon_patch_node)
register_adapter(GordonSurface, _gordon_surface_node)
register_adapter(Mesh, _mesh_node)
register_adapter(_is_wing_surface, _wing_surface_node)
