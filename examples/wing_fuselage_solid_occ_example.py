"""Capped wing + fuselage SOLIDS, boolean-fused, then surface-meshed (OCC/gmsh).

  1. build a finite-tip wing as a WingSurface, loft its airfoil sections into a
     capped solid,
  2. build a fuselage solid by lofting elliptical cross-sections,
  3. boolean-fuse them (interior faces vanish -> no 'dead' surface inside),
  4. surface-mesh the fused solid (unstructured; with a note on structured wing),
  5. plot.

    pip install gmsh
    python examples/wing_fuselage_solid_occ_example.py
"""
import numpy as np

from geometry import Point
from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.occ import GmshSession
from geometry.occ.meshing import generate_surface_mesh
from geometry.occ.solids import fuse_solids, solid_from_section_loops, wing_section_loops


def build_wing() -> WingSurface:
    shape = WingShape(
        semispan=1.2,
        chord_distribution=linear_distr(0.35, 0.14),  # finite tip (no degenerate apex)
        twist_distribution=lambda e: 3.0 * (1 - e),
        airfoil_distribution=lambda e: AirfoilCST(
            upper=CSTPolynomial([0.3, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial([-0.3, -0.2, -0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )
    return WingSurface(shape, n_sections=8, n_chord=60)


def fuselage_section_loops(n_stations: int = 9, n_circ: int = 60) -> list[list[Point]]:
    """Elliptical cross-sections along x; small (non-zero) radii at the ends."""
    x0, x1 = -1.0, 1.5
    loops: list[list[Point]] = []
    thetas = np.linspace(0.0, 2 * np.pi, n_circ, endpoint=False)
    for x in np.linspace(x0, x1, n_stations):
        s = (x - x0) / (x1 - x0)
        r = 0.10 + 0.28 * np.sin(np.pi * s)  # rounded nose/tail, fat middle
        ry, rz = r, 0.85 * r
        loops.append([Point(float(x), float(ry * np.cos(t)), float(rz * np.sin(t))) for t in thetas])
    return loops


def main() -> None:
    wing = build_wing()

    with GmshSession() as s:
        wing_solid = solid_from_section_loops(s.gmsh, wing_section_loops(wing, n_loop=80))
        fuse_solid = solid_from_section_loops(s.gmsh, fuselage_section_loops())
        s.occ.synchronize()

        fused = fuse_solids(s.gmsh, [wing_solid, fuse_solid])
        print("fused solid tag:", fused)
        faces = [tag for (dim, tag) in s.model.getEntities(2)]
        print(f"outer faces: {len(faces)}")

        # Robust default: unstructured surface mesh of the whole solid skin.
        # (To make the OUTBOARD wing faces structured quads, call
        #  geometry.occ.meshing.make_structured_quads(s.gmsh, face_tag, n) on the
        #  4-sided wing faces before generating — the root face is cut by the
        #  fuse and stays unstructured.)
        mesh = generate_surface_mesh(s.gmsh, size_max=0.08)
        print(f"surface mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

        polygons = [np.array([n.as_array() for n in e.nodes]) for e in mesh.elements]

    _plot(polygons)


def _plot(polygons) -> None:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("MacOSX")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(polygons, facecolor=(0.7, 0.8, 0.95, 0.8), edgecolor="k", linewidths=0.2)
    )
    pts = np.vstack(polygons)
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = 0.5 * (pts.max(0) - pts.min(0)).max()
    for setlim, lo in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        setlim(lo - r, lo + r)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Fused wing+fuselage solid — surface mesh (no interior faces)")
    plt.show()


if __name__ == "__main__":
    main()
