"""Tests for the object-oriented topology API over gmsh OCC (skipped without gmsh)."""
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from geometry.occ.session import clear_model, ensure_session  # noqa: E402
from geometry.occ.shapes import (  # noqa: E402
    Edge,
    Face,
    Solid,
    Vertex,
    common,
    cut,
    fragment,
    fuse,
    import_step,
    nearest,
)
from geometry.primitives import Point, Vector  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_model():
    """Every test starts from an empty model in the shared session."""
    ensure_session()
    clear_model()
    yield
    clear_model()


def test_box_topology_counts():
    box = Solid.box(2.0, 1.0, 0.5)
    assert len(box.faces) == 6
    assert len(box.edges) == 12
    assert len(box.vertices) == 8
    assert all(isinstance(f, Face) for f in box.faces)
    assert all(isinstance(e, Edge) for e in box.faces[0].edges)
    assert all(isinstance(v, Vertex) for v in box.faces[0].edges[0].vertices)


def test_box_geometry_properties():
    box = Solid.box(2.0, 1.0, 0.5)
    assert box.volume == pytest.approx(1.0)
    com = box.center_of_mass
    assert (com.x, com.y, com.z) == pytest.approx((1.0, 0.5, 0.25))
    lo, hi = box.bounding_box
    assert (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z) == pytest.approx((2.0, 1.0, 0.5))
    total_area = sum(f.area for f in box.faces)
    assert total_area == pytest.approx(2 * (2 * 1 + 2 * 0.5 + 1 * 0.5))


def test_edge_length_and_sampling():
    box = Solid.box(1.0, 1.0, 1.0)
    lengths = sorted(e.length for e in box.edges)
    assert lengths[0] == pytest.approx(1.0)
    pts = box.edges[0].sample(5)
    assert len(pts) == 5


def test_face_selection_by_point():
    box = Solid.box(1.0, 1.0, 1.0)
    top = box.face_nearest(Point(0.5, 0.5, 2.0))
    assert top.center_of_mass.z == pytest.approx(1.0)
    planes = box.faces_where(lambda f: f.kind == "Plane")
    assert len(planes) == 6


def test_fuse_removes_interior():
    from geometry.occ.shapes import all_shapes

    a = Solid.box(1.0, 1.0, 1.0)
    b = Solid.box(1.0, 1.0, 1.0, corner=Point(0.5, 0.0, 0.0))
    fused = fuse(a, b)
    assert isinstance(fused, Solid)
    assert fused.volume == pytest.approx(1.5)
    # Inputs are consumed: one volume remains (the result may reuse an input's
    # tag, so `a.exists` alone cannot detect consumption -- see module docs).
    assert len(all_shapes(3)) == 1
    assert len(fused.faces) == 6  # coplanar faces merged into a clean box skin


def test_cut_and_common_and_operators():
    a = Solid.box(1.0, 1.0, 1.0)
    tool = Solid.cylinder(Point(0.5, 0.5, -0.1), Vector(0, 0, 1), radius=0.2, height=1.2)
    holed = cut(a, tool)
    assert holed.volume == pytest.approx(1.0 - np.pi * 0.2**2 * 1.0, rel=1e-3)

    b = Solid.box(1.0, 1.0, 1.0)
    c = Solid.box(1.0, 1.0, 1.0, corner=Point(0.5, 0.0, 0.0))
    overlap = common(b, c)
    assert overlap.volume == pytest.approx(0.5)

    d = Solid.box(1.0, 1.0, 1.0)
    e = Solid.sphere(Point(1.0, 0.5, 0.5), 0.3)
    merged = d + e
    assert isinstance(merged, Solid)


def test_fragment_keeps_conformal_pieces():
    a = Solid.box(1.0, 1.0, 1.0)
    b = Solid.box(1.0, 1.0, 1.0, corner=Point(0.5, 0.0, 0.0))
    pieces = fragment(a, b)
    assert len(pieces) == 3  # left-only, overlap, right-only
    assert sum(p.volume for p in pieces) == pytest.approx(1.5)


def test_transforms():
    box = Solid.box(1.0, 1.0, 1.0)
    box.translate(Vector(10.0, 0.0, 0.0))
    assert box.center_of_mass.x == pytest.approx(10.5)
    mirrored = box.mirrored(1.0, 0.0, 0.0)  # about x = 0
    assert mirrored.center_of_mass.x == pytest.approx(-10.5)
    assert box.exists and mirrored.exists


def test_loft_produces_capped_solid():
    def square(z: float, half: float) -> list[Point]:
        return [Point(-half, -half, z), Point(half, -half, z),
                Point(half, half, z), Point(-half, half, z)]

    solid = Solid.loft([square(0.0, 0.5), square(1.0, 0.3)])
    assert isinstance(solid, Solid)
    assert solid.volume > 0
    lo, hi = solid.bounding_box
    assert hi.z - lo.z == pytest.approx(1.0, rel=1e-6)


def test_step_roundtrip(tmp_path):
    Solid.box(1.0, 2.0, 3.0)
    path = tmp_path / "box.step"
    from geometry.occ.shapes import export_step

    export_step(str(path))
    clear_model()
    shapes = import_step(str(path))
    assert len(shapes) == 1
    assert isinstance(shapes[0], Solid)
    assert shapes[0].volume == pytest.approx(6.0, rel=1e-6)


def test_nearest_and_distance():
    a = Solid.box(1.0, 1.0, 1.0)
    b = Solid.box(1.0, 1.0, 1.0, corner=Point(5.0, 0.0, 0.0))
    assert nearest([a, b], Point(5.5, 0.5, 0.5)) is b
    assert a.distance_to(Point(2.0, 0.5, 0.5)) == pytest.approx(1.0)


def test_provenance_records_build_history():
    a = Solid.box(1.0, 1.0, 1.0, name="a")
    b = Solid.cylinder(Point(0.5, 0.5, 0.5), Vector(0, 0, 1), radius=0.2, height=1.0, name="b")
    fused = fuse(a, b, name="part")
    assert fused.provenance.name == "fuse"
    assert [inp.name for inp in fused.provenance.inputs] == ["a", "b"]
    assert a.provenance.name == "box" and a.provenance.params["dx"] == 1.0
    assert b.provenance.params["radius"] == 0.2


def test_loft_provenance_keeps_sections():
    def square(z, half):
        return [Point(-half, -half, z), Point(half, -half, z),
                Point(half, half, z), Point(-half, half, z)]

    solid = Solid.loft([square(0.0, 0.5), square(1.0, 0.3)])
    assert solid.provenance.name == "loft"
    assert len(solid.provenance.inputs) == 2
    assert len(solid.provenance.inputs[0]) == 4  # the section point loops survive


def test_viewer_shows_build_history():
    from geometry.viewer.base import build_scene

    a = Solid.box(1.0, 1.0, 1.0, name="base")
    b = Solid.cylinder(Point(0.5, 0.5, 1.0), Vector(0, 0, 1), radius=0.2, height=0.5,
                       name="boss")
    part = fuse(a, b, name="part")
    scene = build_scene([part])
    solid_node = scene["root"]["children"][0]
    assert solid_node["kind"] == "Solid · fuse"
    history = solid_node["children"][1]
    assert history["label"] == "history: fuse"
    assert history["visible"] is False
    labels = [child["label"] for child in history["children"]]
    assert labels == ["base", "boss"]
    # the consumed inputs still expose their own provenance chains
    assert history["children"][0]["kind"] == "Solid · box"
    assert history["children"][1]["meta"]["radius"] == 0.2


def test_viewer_shows_loft_sections_in_history():
    from geometry.viewer.base import build_scene

    def square(z, half):
        return [Point(-half, -half, z), Point(half, -half, z),
                Point(half, half, z), Point(-half, half, z)]

    solid = Solid.loft([square(0.0, 0.5), square(1.0, 0.3)])
    scene = build_scene([solid])
    history = scene["root"]["children"][0]["children"][1]
    sections = history["children"]
    assert [s["label"] for s in sections] == ["section0", "section1"]
    assert sections[0]["geoms"][0]["kind"] == "lines"  # the input curves are drawn


def test_viewer_adapter_builds_topology_tree():
    from geometry.viewer.base import build_scene

    box = Solid.box(1.0, 1.0, 1.0).named("box")
    scene = build_scene([box])
    root = scene["root"]
    solid_node = root["children"][0]
    assert solid_node["label"] == "box"
    faces_group = solid_node["children"][0]
    assert faces_group["label"] == "faces"
    assert len(faces_group["children"]) == 6
    face_node = faces_group["children"][0]
    assert face_node["geoms"] and face_node["geoms"][0]["kind"] == "mesh"
    assert len(face_node["geoms"][0]["indices"]) % 3 == 0
    edges_group = face_node["children"][0]
    assert len(edges_group["children"]) == 4
