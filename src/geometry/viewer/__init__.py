"""Interactive geometry viewer: type code, see it in the browser, click to inspect.

The explicit API: what you pass to the :class:`Viewer` is what is shown ::

    from geometry.viewer import Viewer

    Viewer([wing, pylon_mesh, my_curve]).show()     # python my_script.py

``show()`` serves a three.js page with a 3D view and an object tree (faces,
edges, curves, points, build history...). Hit save in your IDE (or the Rebuild
button) and the page updates: the script is re-run in a fresh subprocess and
its ``show()`` call feeds the new scene back instead of serving again.

The CLI form works too, for scripts that define ``build() -> list``::

    python -m geometry.viewer my_geometry.py --watch

Objects are displayable when they subclass
:class:`~geometry.viewer.base.Viewable` (with
:class:`~geometry.viewer.base.Child`-declared children) or have a registered
adapter -- built-ins cover ``Point``, curves, ``GordonSurface``/``GordonPatch``,
``WingSurface``, ``Mesh`` and the gmsh shape API (:mod:`geometry.occ.shapes`).

The older PyVista desktop frontend (:class:`Scene` + :func:`view`) is still
available and needs the ``viewer`` extra.
"""

from __future__ import annotations

from geometry.viewer.base import (
    Child,
    Viewable,
    ViewNode,
    as_viewable,
    build_scene,
    can_view,
    register_adapter,
)
from geometry.viewer.hierarchy import Hierarchy, HierarchyNode
from geometry.viewer.ids import SceneId
from geometry.viewer.scene import Scene
from geometry.viewer.viewer import Viewer


def view(scene: Scene, **kwargs) -> object:
    """Open the (PyVista) desktop frontend for ``scene``.

    Imported lazily so that ``import geometry.viewer`` works without PyVista.
    """
    from geometry.viewer.frontends.pyvista_view import launch  # noqa: PLC0415

    return launch(scene, **kwargs)


def serve(script: str, port: int = 8735, watch: bool = True, open_browser: bool = True) -> None:
    """Serve the browser viewer for a ``build()`` script (blocking)."""
    from geometry.viewer.server import ViewerServer  # noqa: PLC0415

    ViewerServer(script, port=port, watch=watch).serve_forever(open_browser=open_browser)


__all__ = [
    "Viewer",
    "Viewable",
    "Child",
    "ViewNode",
    "as_viewable",
    "build_scene",
    "can_view",
    "register_adapter",
    "serve",
    "Scene",
    "Hierarchy",
    "HierarchyNode",
    "SceneId",
    "view",
]
