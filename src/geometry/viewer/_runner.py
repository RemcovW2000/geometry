"""Rebuild runner: execute a user geometry script and emit the scene JSON.

Runs in its own subprocess (spawned by the viewer server) so a crash, an
infinite loop, or leaked gmsh state can never take down the viewer -- and each
rebuild starts from a completely fresh gmsh session.

Contract with the user script: it must define ::

    def build() -> list:        # objects the viewer knows how to display
        ...

Usage (normally invoked by the server, not by hand)::

    python -m geometry.viewer._runner path/to/script.py output.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def run_script(script_path: str) -> dict:
    """Import the script, call ``build()``, and return the scene document."""
    from geometry.viewer.base import build_scene  # noqa: PLC0415  (import after sys.path setup)

    path = Path(script_path).resolve()
    sys.path.insert(0, str(path.parent))  # let the script import its neighbours

    spec = importlib.util.spec_from_file_location("_viewer_user_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build"):
        raise RuntimeError(f"{path.name} does not define a build() function")
    objects = module.build()
    if objects is None:
        raise RuntimeError(f"{path.name}: build() returned None (return a list of objects)")
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    return build_scene(objects, name=path.stem)


def main(argv: list[str]) -> int:
    script, out_path = argv[0], argv[1]
    started = time.time()
    try:
        document = run_script(script)
        document["error"] = None
    except Exception:  # noqa: BLE001  (the traceback is the product here)
        document = {"root": None, "error": traceback.format_exc()}
    document["script"] = script
    document["elapsed"] = round(time.time() - started, 3)
    Path(out_path).write_text(json.dumps(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
