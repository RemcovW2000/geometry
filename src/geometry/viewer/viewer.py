"""The explicit viewer API: ``Viewer([objects]).show()``.

What you pass to the :class:`Viewer` is exactly what the browser shows -- no
hidden discovery. A typical script::

    from geometry.viewer import Viewer

    wing = ...           # shapes, surfaces, curves, meshes, Viewable objects
    mesh = ...
    Viewer([wing, mesh]).show()      # or .launch(); blocks and serves

Auto-update still works: ``show()`` serves the scene it was given and watches
the calling script file; on save (or the Rebuild button) the whole script is
re-run in a fresh subprocess. In that subprocess ``show()`` does not serve --
it detects *capture mode* (an environment variable set by the runner), writes
the scene JSON for the server to pick up, and exits. Same script, two roles.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from geometry.viewer.base import build_scene

#: When set, ``Viewer.show()`` writes the scene to this path and exits instead
#: of serving -- the rebuild runner sets it before re-running the user script.
CAPTURE_ENV = "GEOMETRY_VIEWER_CAPTURE_OUT"


class Viewer:
    """An explicit collection of objects to display in the browser.

    Args:
        objects: the viewable items (shapes, surfaces, curves, meshes,
            ``Viewable`` subclasses, or lists of those).
        name: label of the scene root in the tree.
    """

    def __init__(self, objects: list[Any] | None = None, name: str = "scene"):
        self.objects: list[Any] = list(objects or [])
        self.name = name

    def add(self, *objects: Any) -> Viewer:
        """Add more objects (chainable)."""
        self.objects.extend(objects)
        return self

    def scene(self) -> dict:
        """Build the scene document for the current objects."""
        return build_scene(self.objects, name=self.name)

    def show(self, port: int = 8735, watch: bool = True, open_browser: bool = True) -> None:
        """Serve the scene in the browser (blocking).

        In capture mode (when re-run by the viewer's rebuild subprocess) this
        writes the scene JSON and exits instead, so the same script serves on
        first run and feeds rebuilds afterwards.

        Args:
            port: HTTP port for the viewer page.
            watch: rebuild automatically when the calling script is saved
                (requires the script to be a real file).
            open_browser: open the page in the default browser.
        """
        capture_path = os.environ.get(CAPTURE_ENV)
        if capture_path:
            self._capture(capture_path)
            sys.exit(0)

        from geometry.viewer.server import ViewerServer  # noqa: PLC0415  (avoid cycle)

        script = self._calling_script()
        started = time.time()
        try:
            document = self.scene()
            document["error"] = None
        except Exception:  # noqa: BLE001
            document = {"root": None, "error": traceback.format_exc()}
        document["script"] = str(script) if script else "(interactive)"
        document["elapsed"] = round(time.time() - started, 3)

        server = ViewerServer(script, port=port, watch=bool(watch and script),
                              initial_document=document)
        server.serve_forever(open_browser=open_browser)

    #: Alias for :meth:`show`.
    launch = show

    def _capture(self, out_path: str) -> None:
        """Write the scene document for the rebuild runner (capture mode)."""
        started = time.time()
        try:
            document = self.scene()
            document["error"] = None
        except Exception:  # noqa: BLE001
            document = {"root": None, "error": traceback.format_exc()}
        document["script"] = sys.argv[0]
        document["elapsed"] = round(time.time() - started, 3)
        Path(out_path).write_text(json.dumps(document))

    @staticmethod
    def _calling_script() -> Path | None:
        """The file of the running script (what rebuild re-runs), if it exists."""
        main = sys.modules.get("__main__")
        file = getattr(main, "__file__", None)
        if file is None:
            return None
        path = Path(file).resolve()
        return path if path.exists() else None
