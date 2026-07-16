"""CLI: ``python -m geometry.viewer path/to/script.py [--port N] [--watch]``.

The script must define ``build() -> list`` returning viewable objects
(gmsh shapes, GordonSurface, curves, meshes, Viewable subclasses, ...).
"""
from __future__ import annotations

import argparse

from geometry.viewer.server import ViewerServer


def main() -> None:
    """Parse arguments and serve the viewer."""
    parser = argparse.ArgumentParser(
        prog="python -m geometry.viewer",
        description="Interactive browser viewer for geometry scripts.",
    )
    parser.add_argument("script", help="Python file defining build() -> list of objects")
    parser.add_argument("--port", type=int, default=8735)
    parser.add_argument("--watch", action="store_true",
                        help="rebuild automatically when the script file is saved")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser tab")
    args = parser.parse_args()

    server = ViewerServer(args.script, port=args.port, watch=args.watch)
    server.serve_forever(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
