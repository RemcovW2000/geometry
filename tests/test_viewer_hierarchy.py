"""Tests for the pure (GUI-free) viewer core: hierarchy and click resolution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.mesh.surface_mesh import mesh_surface
from geometry.primitives import Point
from geometry.viewer.ids import SceneId
from geometry.viewer.scene import Scene


class _Saddle:
    def point_at_parameter(self, u: float, v: float) -> Point:
        return Point(u, v, 0.4 * math.sin(math.pi * u) * math.sin(math.pi * v))


@pytest.fixture()
def scene() -> Scene:
    stations = [i / 4 for i in range(5)]  # 4x4 = 16 quads, 25 nodes
    mesh = mesh_surface(_Saddle(), stations, stations)
    return Scene(mesh, name="mesh", surface=_Saddle(), surface_name="saddle")


def test_scene_id_str_and_validation() -> None:
    assert str(SceneId("element", (5,))) == "element(5)"
    assert str(SceneId("edge", (5, 2))) == "edge(5, 2)"
    with pytest.raises(ValueError):
        SceneId("banana", (0,))


def test_hierarchy_counts(scene: Scene) -> None:
    # 16 elements, each with 4 edges, 25 nodes.
    assert len(scene.mesh.elements) == 16
    assert len(scene.mesh.nodes) == 25
    elem_nodes = [n for n in scene.hierarchy if n.id.kind == "element"]
    edge_nodes = [n for n in scene.hierarchy if n.id.kind == "edge"]
    node_nodes = [n for n in scene.hierarchy if n.id.kind == "node"]
    assert len(elem_nodes) == 16
    assert len(edge_nodes) == 16 * 4
    assert len(node_nodes) == 25


def test_path_from_surface_root(scene: Scene) -> None:
    assert scene.hierarchy.path(SceneId("element", (0,))) == [
        "saddle",
        "mesh",
        "Element 0",
    ]
    assert scene.hierarchy.path(SceneId("edge", (0, 0)))[:3] == [
        "saddle",
        "mesh",
        "Element 0",
    ]


def test_ids_are_unique(scene: Scene) -> None:
    ids = [n.id for n in scene.hierarchy]
    assert len(ids) == len(set(ids))


def test_resolve_click_snaps_to_node(scene: Scene) -> None:
    target = scene.mesh.nodes[7]
    node = scene.resolve_click(target.as_array())
    assert node.id == SceneId("node", (7,))


def test_resolve_click_snaps_to_edge(scene: Scene) -> None:
    # Midpoint of edge 0 of element 0 should resolve to that edge.
    elem = scene.mesh.elements[0]
    mid = 0.5 * (elem.nodes[0].as_array() + elem.nodes[1].as_array())
    node = scene.resolve_click(mid)
    assert node.id == SceneId("edge", (0, 0))


def test_resolve_click_falls_to_element(scene: Scene) -> None:
    # The centroid of an element (away from any node/edge) resolves to it.
    centroid = scene.mesh.elements[3].node_coordinates().mean(axis=0)
    node = scene.resolve_click(centroid)
    assert node.id == SceneId("element", (3,))


def test_resolve_click_roundtrips_to_object(scene: Scene) -> None:
    # The core guarantee: pick -> id -> hierarchy node -> original object.
    for i, mesh_node in enumerate(scene.mesh.nodes):
        resolved = scene.resolve_click(mesh_node.as_array())
        assert resolved.obj is mesh_node
        assert resolved.id == SceneId("node", (i,))


def test_describe_contains_path(scene: Scene) -> None:
    node = scene.hierarchy.node_for_id(SceneId("edge", (5, 2)))
    text = scene.describe(node)
    assert "Element 5" in text and "Edge" in text


def test_node_for_missing_id_raises(scene: Scene) -> None:
    with pytest.raises(KeyError):
        scene.hierarchy.node_for_id(SceneId("element", (999,)))
    assert scene.hierarchy.get(SceneId("element", (999,))) is None


def test_pick_geometry_shapes(scene: Scene) -> None:
    assert scene._node_coords.shape == (25, 3)
    assert scene._edge_mids.shape == (16 * 4, 3)
    assert scene._elem_centroids.shape == (16, 3)
    assert np.isfinite(scene._h) and scene._h > 0
