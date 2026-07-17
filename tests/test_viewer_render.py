"""Render-layer tests. Skipped entirely when PyVista (the ``viewer`` extra) is absent."""

from __future__ import annotations

import math

import pytest

from geometry.mesh.surface_mesh import mesh_surface
from geometry.primitives import Point
from geometry.viewer.scene import Scene

pv = pytest.importorskip("pyvista")


class _Saddle:
    def point_at_parameter(self, u: float, v: float) -> Point:
        return Point(u, v, 0.4 * math.sin(math.pi * u) * math.sin(math.pi * v))


@pytest.fixture()
def scene() -> Scene:
    stations = [i / 4 for i in range(5)]
    return Scene(mesh_surface(_Saddle(), stations, stations), name="mesh")


def test_build_polydata_shapes(scene: Scene) -> None:
    render = scene.build_polydata()
    assert render.faces.n_cells == 16
    assert render.faces.n_points == 25
    assert render.edges.n_cells == 16 * 4
    assert render.nodes.n_points == 25
    # Id lists align 1:1 with the cells/points they annotate.
    assert len(render.face_ids) == render.faces.n_cells
    assert len(render.edge_ids) == render.edges.n_cells
    assert len(render.node_ids) == render.nodes.n_points


def test_face_cell_id_matches_hierarchy(scene: Scene) -> None:
    render = scene.build_polydata()
    # Cell 7's id must resolve in the hierarchy to "Element 7".
    fid = render.face_ids[7]
    assert scene.hierarchy.node_for_id(fid).label == "Element 7"


def test_offscreen_launch_smoke(scene: Scene) -> None:
    pv.OFF_SCREEN = True
    from geometry.viewer.frontends.pyvista_view import launch  # noqa: PLC0415

    picked = []
    pl = launch(scene, off_screen=True, on_pick=picked.append)

    # Regression: the faces mesh must be pickable, otherwise clicks never
    # register and the viewer feels "dead" (nodes/edges are non-pickable so as
    # not to steal clicks from the surface picker).
    assert bool(pl.renderer.actors["faces"].GetPickable()) is True
    assert bool(pl.renderer.actors["nodes"].GetPickable()) is False

    # Drive the pick path directly (no interactive window in a test).
    node = scene.resolve_click(scene.mesh.nodes[0].as_array())
    assert node.label == "Node 0"
    pl.close()
