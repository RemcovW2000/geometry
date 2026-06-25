"""Two Gordon surfaces (wing + fuselage) intersected and meshed with OCC/gmsh.

Stage 1 (interpolation fidelity, open sheets):
  - build a wing sheet and a fuselage half-barrel as Gordon surfaces,
  - convert each to an OCC B-spline face (interpolating control net),
  - `fragment` them so they split along their intersection curve (conformal),
  - unstructured-mesh the result and plot it, highlighting the seam,
  - export a STEP file.

Closed wing loops / full fuselage tubes are the follow-up ("more complex route").

    pip install gmsh
    python examples/wing_fuselage_occ_example.py
"""
import numpy as np

from geometry import GordonSurface, Point
from geometry.curves import InterpolatedLine
from geometry.occ import GmshSession
from geometry.occ.bspline import add_bspline_surface, surface_to_bspline_data
from geometry.occ.meshing import generate_surface_mesh, refine_near_curves
from geometry.occ.ops import export_step, fragment, intersection_curves


def _grid_gordon(point_fn, u_params, v_params) -> GordonSurface:
    """Build a compatible Gordon surface from a grid point function P(u, v)."""
    grid = [[point_fn(u, v) for u in u_params] for v in v_params]
    profiles = [InterpolatedLine(grid[i], params=list(u_params)) for i in range(len(v_params))]
    guides = [
        InterpolatedLine([grid[i][j] for i in range(len(v_params))], params=list(v_params))
        for j in range(len(u_params))
    ]
    return GordonSurface(profiles, guides, list(v_params), list(u_params))


def build_wing_sheet() -> GordonSurface:
    """A cambered wing sheet: chord in x, span in y, near z=0, crossing y=R."""
    chord = 1.0

    def P(u, v):  # u chordwise, v spanwise
        x = u * chord
        y = -0.1 + v * 1.3
        z = 0.08 * np.sin(np.pi * u)  # arch camber
        return Point(x, y, z)

    return _grid_gordon(P, u_params=[0, 0.25, 0.5, 0.75, 1.0], v_params=[0, 0.33, 0.66, 1.0])


def build_fuselage_sheet() -> GordonSurface:
    """An open half-barrel: axis along x, radius R, the wing pierces its +y side."""
    R = 0.35
    th0, th1 = np.radians(-120), np.radians(120)
    x0, x1 = -0.5, 1.5

    def P(u, v):  # u around (theta), v along x
        theta = th0 + u * (th1 - th0)
        x = x0 + v * (x1 - x0)
        return Point(x, R * np.cos(theta), R * np.sin(theta))

    return _grid_gordon(P, u_params=list(np.linspace(0, 1, 7)), v_params=list(np.linspace(0, 1, 5)))


def main() -> None:
    wing = build_wing_sheet()
    fuse = build_fuselage_sheet()

    us = list(np.linspace(0, 1, 18))
    vs = list(np.linspace(0, 1, 14))

    with GmshSession() as s:
        w = add_bspline_surface(s.gmsh, surface_to_bspline_data(wing, us, vs))
        f = add_bspline_surface(s.gmsh, surface_to_bspline_data(fuse, us, vs))
        s.occ.synchronize()

        out, _ = fragment(s.gmsh, [(2, w)], [(2, f)])
        faces = [tag for (dim, tag) in out if dim == 2]
        print("fragment produced faces:", faces)

        seam = []
        if len(faces) >= 2:
            seam = intersection_curves(s.gmsh, faces[0], faces[1])
            seam_tags = [t for (d, t) in s.model.getEntities(1)]
            refine_near_curves(s.gmsh, seam_tags, size=0.03, distance=0.2)

        mesh = generate_surface_mesh(s.gmsh, size_max=0.1)
        print(f"mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

        export_step(s.gmsh, "wing_fuselage.step")
        print("wrote wing_fuselage.step")

        # extract for plotting before the session closes
        polygons = [np.array([n.as_array() for n in e.nodes]) for e in mesh.elements]
        seam_xyz = [np.array([p.as_array() for p in c.points]) for c in seam]

    _plot(polygons, seam_xyz)


def _plot(polygons, seam_xyz) -> None:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("MacOSX")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(polygons, facecolor=(0.6, 0.75, 0.95, 0.6), edgecolor="k", linewidths=0.2)
    )
    for s in seam_xyz:
        ax.plot(s[:, 0], s[:, 1], s[:, 2], "r-", lw=3)

    pts = np.vstack(polygons)
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = 0.5 * (pts.max(0) - pts.min(0)).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Wing ∩ fuselage — conformal unstructured mesh (seam in red)")
    plt.show()


if __name__ == "__main__":
    main()
