"""Tests for gmsh volume meshing and entity-linked mesh regions (skip without gmsh)."""
import numpy as np
import pytest

gmsh_module = pytest.importorskip("gmsh")

from geometry.occ.meshing import generate_volume_mesh  # noqa: E402
from geometry.occ.session import clear_model, ensure_session  # noqa: E402
from geometry.occ.shapes import Solid  # noqa: E402
from geometry.primitives import Point  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_model():
    ensure_session()
    clear_model()
    yield
    clear_model()


def tet_volume(nodes) -> float:
    p = np.array([n.as_array() for n in nodes], dtype=float)
    return abs(np.linalg.det(p[1:] - p[0])) / 6.0


def test_cube_tet_mesh_volume_and_counts():
    Solid.box(2.0, 2.0, 2.0)
    meshed = generate_volume_mesh(ensure_session(), size_max=0.8)
    assert len(meshed.mesh.elements) > 20
    assert all(len(tet.nodes) == 4 for tet in meshed.mesh.elements)
    total = sum(tet_volume(tet.nodes) for tet in meshed.mesh.elements)
    assert total == pytest.approx(8.0, rel=1e-6)  # tets exactly fill the cube


def test_face_region_shares_node_objects_with_volume_mesh():
    cube = Solid.box(2.0, 2.0, 2.0)
    top_face = cube.face_nearest(Point(1.0, 1.0, 3.0))
    meshed = generate_volume_mesh(ensure_session(), size_max=0.8)
    region = meshed.region(top_face, name="top")

    assert region.nodes and region.facets
    assert all(abs(node.z - 2.0) < 1e-9 for node in region.nodes)
    # The critical property: facets are built over the SAME Node objects as the
    # volume mesh (identity, not just equal coordinates) -> layered elements
    # (e.g. composite coatings) share DOFs with the solid automatically.
    mesh_node_ids = {id(n) for n in meshed.mesh.nodes}
    for facet in region.facets:
        assert all(id(n) in mesh_node_ids for n in facet.nodes)
    # Facet triangles conform: their nodes lie on the face too.
    region_ids = {id(n) for n in region.nodes}
    assert all(id(n) in region_ids for facet in region.facets for n in facet.nodes)


def test_edge_region_gives_segments():
    cube = Solid.box(2.0, 2.0, 2.0)
    edge = cube.edge_nearest(Point(1.0, 0.0, 2.0))
    meshed = generate_volume_mesh(ensure_session(), size_max=0.8)
    region = meshed.region(edge, name="edge")
    assert len(region.nodes) >= 2
    assert all(len(segment.nodes) == 2 for segment in region.facets)
    assert all(abs(n.y) < 1e-9 and abs(n.z - 2.0) < 1e-9 for n in region.nodes)
