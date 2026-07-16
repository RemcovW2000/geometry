"""A minimal PyVista desktop frontend that demonstrates the pick-to-hierarchy loop.

This is the prototype coupling: it renders a :class:`~geometry.viewer.scene.Scene`
(faces, edges, nodes), and on every left click resolves the picked point to a
hierarchy node and shows its full path both in an on-screen banner and on stdout.
Clicking a node snaps to that node, clicking an edge to that edge, and clicking a
face interior to that element.

The web (trame) frontend is intended to reuse :class:`Scene` unchanged; only the
picking transport differs. See ``docs/viewer_design.md``.
"""

from __future__ import annotations

from typing import Callable

from geometry.viewer.hierarchy import HierarchyNode
from geometry.viewer.scene import Scene

# Highlight colours per kind.
_COLOURS = {"node": "red", "edge": "orange", "element": "gold"}


def launch(
    scene: Scene,
    off_screen: bool = False,
    on_pick: Callable[[HierarchyNode], None] | None = None,
) -> object:
    """Open an interactive window for ``scene`` with click-to-inspect picking.

    Args:
        scene: The scene to display.
        off_screen: If True, build the plotter without opening a window (used
            for headless smoke tests / screenshots).
        on_pick: Optional extra callback invoked with the resolved node on each
            pick (in addition to the built-in banner + stdout report).

    Returns:
        The configured PyVista ``Plotter`` (already shown unless ``off_screen``).
    """
    import pyvista as pv  # noqa: PLC0415  (optional 'viewer' extra)

    render = scene.build_polydata()

    pl = pv.Plotter(off_screen=off_screen)
    # The faces mesh must stay pickable: the surface picker returns the world
    # point where the click hits it, which Scene.resolve_click then snaps to the
    # nearest node / edge / element.
    pl.add_mesh(
        render.faces,
        color="lightsteelblue",
        show_edges=True,
        edge_color="steelblue",
        opacity=1.0,
        pickable=True,
        name="faces",
    )
    # Node markers are visual only; they must not be pickable or they steal
    # clicks from the surface picker.
    pl.add_mesh(
        render.nodes,
        color="black",
        point_size=8,
        render_points_as_spheres=True,
        pickable=False,
        name="nodes",
    )

    pl.add_text("Click a face, edge or node…", font_size=10, name="banner")
    highlight_name = "__highlight__"

    def report(point, *_ignored) -> None:
        if point is None:
            return
        node = scene.resolve_click(point)
        line = scene.describe(node)
        print(line)
        pl.add_text(line, font_size=10, name="banner")
        _highlight(pl, scene, node, highlight_name)
        if on_pick is not None:
            on_pick(node)

    # A single left click on the surface drives element/edge/node selection.
    pl.enable_surface_point_picking(
        callback=report,
        show_point=False,
        show_message=False,
        left_clicking=True,
    )

    if not off_screen:
        pl.show()
    return pl


def _highlight(pl, scene: Scene, node: HierarchyNode, name: str) -> None:
    """Draw a highlight actor over the resolved primitive."""
    import numpy as np  # noqa: PLC0415  (optional 'viewer' extra)
    import pyvista as pv  # noqa: PLC0415

    kind = node.id.kind
    colour = _COLOURS.get(kind, "magenta")

    if kind == "node":
        (i,) = node.id.key
        pl.add_mesh(
            pv.PolyData(scene._node_coords[i : i + 1]),
            color=colour,
            point_size=16,
            render_points_as_spheres=True,
            name=name,
        )
    elif kind == "edge":
        a, b = node.obj
        seg = pv.lines_from_points(np.array([a.as_array(), b.as_array()]))
        pl.add_mesh(seg, color=colour, line_width=8, name=name)
    elif kind == "element":
        (e_idx,) = node.id.key
        pts = scene.mesh.elements[e_idx].node_coordinates()
        n = len(pts)
        face = np.hstack([[n], list(range(n))])
        pl.add_mesh(pv.PolyData(pts, faces=face), color=colour, opacity=0.6, name=name)
