"""Rebuild runner: execute a user geometry script and emit the scene JSON.

Runs in its own subprocess (spawned by the viewer server) so a crash, an
infinite loop, or leaked gmsh state can never take down the viewer -- and each
rebuild starts from a completely fresh gmsh session.

Two script styles are supported:

1. The script calls ``Viewer([...]).show()``: the runner sets the capture
   environment variable before executing it, so ``show()`` writes the scene
   JSON (instead of serving) and exits.
2. The script defines ``build() -> list``: the runner calls it and builds the
   scene from its return value.

Usage (normally invoked by the server, not by hand)::

    python -m geometry.viewer._runner path/to/script.py output.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def run_script(script_path: str) -> dict:
    """Import the script and return the scene document from its ``build()``.

    If the script calls ``Viewer.show()`` at import time while capture mode is
    active, the resulting ``SystemExit`` propagates to the caller (the scene
    file has already been written by then).
    """
    from geometry.viewer.base import build_scene  # noqa: PLC0415  (import after sys.path setup)

    path = Path(script_path).resolve()
    sys.path.insert(0, str(path.parent))  # let the script import its neighbours

    spec = importlib.util.spec_from_file_location("_viewer_user_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build"):
        raise RuntimeError(
            f"{path.name} neither defines build() nor calls Viewer.show() -- "
            "one of the two is needed to produce a scene"
        )
    objects = module.build()
    if objects is None:
        raise RuntimeError(f"{path.name}: build() returned None (return a list of objects)")
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    return build_scene(objects, name=path.stem)


def main(argv: list[str]) -> int:
    script, out_path = argv[0], argv[1]
    from geometry.viewer.viewer import CAPTURE_ENV  # noqa: PLC0415

    os.environ[CAPTURE_ENV] = out_path  # makes any Viewer.show() capture instead of serve
    started = time.time()
    try:
        document = run_script(script)
        document["error"] = None
    except SystemExit:
        # A Viewer.show() captured the scene itself; pick up what it wrote.
        try:
            document = json.loads(Path(out_path).read_text())
        except (OSError, json.JSONDecodeError):
            document = {"root": None, "error": "script exited before producing a scene"}
    except Exception:  # noqa: BLE001  (the traceback is the product here)
        document = {"root": None, "error": traceback.format_exc()}
    document["script"] = script
    document["elapsed"] = round(time.time() - started, 3)
    Path(out_path).write_text(json.dumps(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
