"""Prototype: build a small surface mesh and open the interactive viewer.

Run with the viewer extra installed::

    pip install -e '.[viewer]'
    python examples/viewer_prototype.py

A window opens showing a meshed saddle surface. Click any face, edge or node and
the banner (and console) reports where it sits in the hierarchy, e.g.::

    Edge n3–n4  [surface ▸ mesh ▸ Element 5 ▸ Edge n3–n4]

This is the pick-to-hierarchy loop described in ``docs/viewer_design.md``.
"""

from __future__ import annotations

import math

from geometry.mesh.surface_mesh import mesh_surface
from geometry.primitives import Point
from geometry.viewer import Scene, view


class SaddleSurface:
    """A tiny parametric surface: z = sin(pi u) * sin(pi v) over the unit square.

    Any object with ``point_at_parameter(u, v) -> Point`` can be meshed, so this
    stands in for a real :class:`~geometry.surfaces.GordonSurface`.
    """

    def point_at_parameter(self, u: float, v: float) -> Point:  # noqa: D102
        x = u
        y = v
        z = 0.1 * math.sin(math.pi * u) * math.sin(math.pi * v)
        return Point(x, y, z)


def build_scene() -> Scene:
    """Mesh the saddle surface into a 6x6 quad grid and wrap it in a Scene."""
    surface = SaddleSurface()
    stations = [i / 6 for i in range(7)]  # 7 stations -> 6 quads per direction
    mesh = mesh_surface(surface, stations, stations)
    return Scene(mesh, name="mesh", surface=surface, surface_name="saddle")


def main() -> None:
    """Build the scene and launch the interactive viewer."""
    scene = build_scene()
    print(
        f"Loaded {scene.name}: "
        f"{len(scene.mesh.elements)} elements, {len(scene.mesh.nodes)} nodes."
    )
    print("Click a face, edge or node to see its place in the hierarchy.")
    view(scene)


if __name__ == "__main__":
    main()
