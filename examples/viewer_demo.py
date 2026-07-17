"""Demo scene for the browser viewer.

Run:
    python examples/viewer_demo.py

Then edit this file (change a chord, move a point, ...) and save: the page
rebuilds and shows the new geometry. The tree on the left mirrors the object
hierarchy: the Gordon patch contains its boundary curves, each curve its
defining points; the gmsh solid (if gmsh is installed) contains faces, the
faces edges, and so on -- plus a hidden-by-default "history" node showing what
each solid was built from.

(The CLI form still works too: python -m geometry.viewer examples/viewer_demo.py --watch)
"""
import numpy as np

from geometry import InterpolatedLine, Point
from geometry.surfaces import GordonPatch
from geometry.viewer import Child, Viewable


class WingPanel(Viewable):
    """Example of a user-defined composite object with declared children."""

    surface = Child()
    spar_line = Child()

    def __init__(self, surface: GordonPatch, spar_line: InterpolatedLine):
        self.surface = surface
        self.spar_line = spar_line

    def viz_meta(self) -> dict:
        """Metadata shown in the inspector."""
        return {"description": "Gordon patch + a spar trace on it"}


def _gordon_panel() -> WingPanel:
    """A doubly-curved Gordon patch with a spar curve traced on it."""
    # Two u-lines (root and tip chords, cambered) ...
    root = InterpolatedLine([Point(x, 0.0, 0.12 * np.sin(np.pi * x)) for x in np.linspace(0, 1, 9)])
    tip = InterpolatedLine([Point(0.25 + 0.5 * x, 2.0, 0.06 * np.sin(np.pi * x) + 0.15)
                            for x in np.linspace(0, 1, 9)])
    # ... and two v-lines (leading and trailing edges) connecting them.
    leading = InterpolatedLine([Point(0.25 * v, 2.0 * v, 0.15 * v) for v in np.linspace(0, 1, 7)])
    trailing = InterpolatedLine([Point(1.0 - 0.25 * v, 2.0 * v, 0.15 * v) for v in np.linspace(0, 1, 7)])
    patch = GordonPatch(u_lines=[root, tip], v_lines=[leading, trailing])

    spar = InterpolatedLine([patch.point_at_parameter(0.3, v) for v in np.linspace(0, 1, 15)])
    return WingPanel(patch, spar)


def _gmsh_part():
    """A small CAD part through the topology API (skipped if gmsh is missing)."""
    from geometry import Vector
    from geometry.occ import available
    from geometry.occ.shapes import Solid, fuse

    if not available():
        return []
    box = Solid.box(1.2, 0.8, 0.4, corner=Point(2.0, -0.4, 0.0), name="base")
    boss = Solid.cylinder(Point(1, 0.0, 0.4), Vector(0, 0, 1), radius=0.18, height=0.5,
                          name="boss")
    part = fuse(box, boss, name="bracket")
    return [part]


def build() -> list:
    """The objects to display."""
    return [_gordon_panel(), *_gmsh_part()]


if __name__ == "__main__":
    from geometry.viewer import Viewer

    Viewer(build(), name="viewer_demo").show()
