"""Tests for the viewable object model and the browser-viewer scene pipeline.

Pure Python: no gmsh, no browser, no display needed.
"""
import json
import textwrap
import threading
import urllib.request

import numpy as np
import pytest

from geometry import InterpolatedLine, Point
from geometry.surfaces import GordonPatch
from geometry.viewer import Child, Viewable, as_viewable, build_scene, can_view
from geometry.viewer.base import grid_payload, points_payload, polyline_payload


def make_patch() -> GordonPatch:
    u0 = InterpolatedLine([Point(x, 0, 0) for x in np.linspace(0, 1, 5)])
    u1 = InterpolatedLine([Point(x, 1, 0.2) for x in np.linspace(0, 1, 5)])
    v0 = InterpolatedLine([Point(0, y, 0.2 * y) for y in np.linspace(0, 1, 5)])
    v1 = InterpolatedLine([Point(1, y, 0.2 * y) for y in np.linspace(0, 1, 5)])
    return GordonPatch(u_lines=[u0, u1], v_lines=[v0, v1])


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #

def test_grid_payload_counts():
    grid = np.zeros((4, 3, 3))
    payload = grid_payload(grid)
    assert payload["kind"] == "mesh"
    assert len(payload["positions"]) == 4 * 3 * 3
    assert len(payload["indices"]) == 3 * 2 * 2 * 3  # (nu-1)(nv-1) quads * 2 tris * 3


def test_polyline_and_points_payloads():
    pts = [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0)]
    line = polyline_payload(pts)
    assert line["segments"] == [0, 1, 1, 2]
    cloud = points_payload(pts)
    assert len(cloud["positions"]) == 9


# --------------------------------------------------------------------------- #
# Viewable / Child
# --------------------------------------------------------------------------- #

class Assembly(Viewable):
    surface = Child()
    curve = Child()


def test_child_declaration_order_and_children():
    assembly = Assembly()
    assembly.surface = make_patch()
    assembly.curve = InterpolatedLine([Point(0, 0, 0), Point(1, 1, 1)])
    names = [name for name, _ in assembly.children()]
    assert names == ["surface", "curve"]


def test_child_rejects_unviewable_values():
    assembly = Assembly()
    with pytest.raises(TypeError, match="must be viewable"):
        assembly.surface = object()


def test_unassigned_child_is_skipped_and_raises_on_access():
    assembly = Assembly()
    assembly.curve = InterpolatedLine([Point(0, 0, 0), Point(1, 1, 1)])
    assert [name for name, _ in assembly.children()] == ["curve"]
    with pytest.raises(AttributeError):
        _ = assembly.surface


def test_can_view_covers_adapted_types_and_lists():
    assert can_view(Point(0, 0, 0))
    assert can_view(make_patch())
    assert can_view([Point(0, 0, 0), make_patch()])
    assert not can_view(object())


# --------------------------------------------------------------------------- #
# Scene building
# --------------------------------------------------------------------------- #

def test_build_scene_tree_and_ids():
    assembly = Assembly()
    assembly.surface = make_patch()
    assembly.curve = InterpolatedLine([Point(0, 0, 0), Point(1, 1, 1)])
    scene = build_scene([assembly], name="demo")

    root = scene["root"]
    assert root["label"] == "demo"
    node = root["children"][0]
    assert node["label"] == "Assembly"
    labels = [child["label"] for child in node["children"]]
    assert labels == ["surface", "curve"]
    # hierarchical ids
    assert node["id"] == "0/0"
    assert node["children"][0]["id"] == "0/0/0"
    # the patch node carries a mesh and contains its boundary curves
    patch_node = node["children"][0]
    assert patch_node["geoms"][0]["kind"] == "mesh"
    groups = [c["label"] for c in patch_node["children"]]
    assert groups == ["u_lines", "v_lines"]
    # curves contain their defining points
    curve_node = patch_node["children"][0]["children"][0]
    assert curve_node["children"][0]["label"] == "points"
    # whole document is JSON-serializable
    json.dumps(scene)


def test_gordon_patch_mesh_interpolates_boundary():
    patch = make_patch()
    node = as_viewable(patch)
    positions = np.array(node.geoms[0]["positions"]).reshape(-1, 3)
    corner = patch.point_at_parameter(0.0, 0.0)
    assert np.min(np.linalg.norm(positions - [corner.x, corner.y, corner.z], axis=1)) < 1e-9


# --------------------------------------------------------------------------- #
# Runner + server round trip (no gmsh required)
# --------------------------------------------------------------------------- #

SCRIPT = textwrap.dedent(
    """
    from geometry import InterpolatedLine, Point

    def build():
        return [InterpolatedLine([Point(0, 0, 0), Point(1, 0, 1), Point(2, 0, 0)])]
    """
)


def test_runner_executes_script(tmp_path):
    from geometry.viewer._runner import run_script

    script = tmp_path / "scene_script.py"
    script.write_text(SCRIPT)
    document = run_script(str(script))
    assert document["root"]["children"][0]["kind"] == "InterpolatedLine"


def test_runner_reports_build_errors(tmp_path):
    from geometry.viewer._runner import main

    script = tmp_path / "broken.py"
    script.write_text("def build():\n    raise ValueError('boom')\n")
    out = tmp_path / "out.json"
    main([str(script), str(out)])
    document = json.loads(out.read_text())
    assert document["root"] is None
    assert "boom" in document["error"]


# --------------------------------------------------------------------------- #
# The explicit Viewer API
# --------------------------------------------------------------------------- #

def test_viewer_holds_explicit_objects_and_builds_scene():
    from geometry.viewer import Viewer

    curve = InterpolatedLine([Point(0, 0, 0), Point(1, 0, 1)])
    viewer = Viewer([make_patch()], name="explicit").add(curve)
    scene = viewer.scene()
    assert scene["root"]["label"] == "explicit"
    kinds = [child["kind"] for child in scene["root"]["children"]]
    assert kinds == ["GordonPatch", "InterpolatedLine"]


def test_viewer_show_captures_instead_of_serving(tmp_path, monkeypatch):
    from geometry.viewer import Viewer
    from geometry.viewer.viewer import CAPTURE_ENV

    out = tmp_path / "captured.json"
    monkeypatch.setenv(CAPTURE_ENV, str(out))
    viewer = Viewer([Point(1, 2, 3)], name="captured")
    with pytest.raises(SystemExit):
        viewer.show()
    document = json.loads(out.read_text())
    assert document["error"] is None
    assert document["root"]["label"] == "captured"
    assert document["root"]["children"][0]["kind"] == "Point"


def test_runner_supports_viewer_show_scripts(tmp_path):
    from geometry.viewer._runner import main

    script = tmp_path / "show_script.py"
    script.write_text(textwrap.dedent(
        """
        from geometry import InterpolatedLine, Point
        from geometry.viewer import Viewer

        line = InterpolatedLine([Point(0, 0, 0), Point(1, 1, 1)])
        Viewer([line], name="from_show").show()
        """
    ))
    out = tmp_path / "out.json"
    main([str(script), str(out)])
    document = json.loads(out.read_text())
    assert document["error"] is None
    assert document["root"]["label"] == "from_show"
    assert document["script"] == str(script)


def test_server_endpoints(tmp_path):
    from geometry.viewer.server import ViewerServer

    script = tmp_path / "scene_script.py"
    script.write_text(SCRIPT)
    server = ViewerServer(script, port=0)  # port 0: pick a free one
    http = __import__("http.server", fromlist=["ThreadingHTTPServer"])
    httpd = http.ThreadingHTTPServer(("127.0.0.1", 0), server._handler_class())
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        server.rebuild()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/scene", timeout=10) as response:
            scene = json.loads(response.read())
        assert scene["error"] is None
        assert scene["version"] == 1
        assert scene["root"]["children"][0]["kind"] == "InterpolatedLine"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=10) as response:
            status = json.loads(response.read())
        assert status["version"] == 1 and status["building"] is False

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            page = response.read().decode()
        assert "geometry viewer" in page
    finally:
        httpd.shutdown()
