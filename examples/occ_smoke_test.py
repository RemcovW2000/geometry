"""OCC backend smoke test — RUN THIS FIRST on a machine with gmsh installed.

Verifies that:
  1. gmsh is importable and a session works,
  2. a geometry surface becomes an OCC B-spline face with the correct control-
     point ordering (gmsh's surface evaluation matches ours), and
  3. the model meshes.

    pip install gmsh
    python examples/occ_smoke_test.py
"""
import numpy as np

from geometry import GordonSurface, Point
from geometry.curves import InterpolatedLine
from geometry.occ import GmshSession, available
from geometry.occ.bspline import add_bspline_surface, surface_to_bspline_data
from geometry.occ.meshing import generate_surface_mesh


def wavy_surface() -> GordonSurface:
    v_params = [0.0, 0.5, 1.0]
    u_params = [0.0, 0.4, 0.7, 1.0]
    xs = [2 * u for u in u_params]
    ys = [2 * v for v in v_params]
    h = lambda x, y: 0.4 * np.sin(np.pi * x / 2) * np.cos(np.pi * y / 2)
    grid = [[Point(xs[j], ys[i], h(xs[j], ys[i])) for j in range(len(u_params))]
            for i in range(len(v_params))]
    profiles = [InterpolatedLine(grid[i], params=u_params) for i in range(len(v_params))]
    guides = [InterpolatedLine([grid[i][j] for i in range(len(v_params))], params=v_params)
              for j in range(len(u_params))]
    return GordonSurface(profiles, guides, v_params, u_params)


def main() -> None:
    print("gmsh available:", available())
    surf = wavy_surface()
    us = list(np.linspace(0, 1, 12))
    vs = list(np.linspace(0, 1, 12))

    with GmshSession(terminal=False) as s:
        data = surface_to_bspline_data(surf, us, vs)
        tag = add_bspline_surface(s.gmsh, data)
        s.occ.synchronize()

        lo, hi = s.model.getParametrizationBounds(2, tag)
        print("surface param bounds:", lo, hi, "(expected ~[0,0]..[1,1])")

        errs = []
        for u in np.linspace(0, 1, 7):
            for v in np.linspace(0, 1, 7):
                xyz = np.array(s.model.getValue(2, tag, [float(u), float(v)]))
                errs.append(np.linalg.norm(xyz - surf.point_at_parameter(u, v).as_array()))
        print(f"max gmsh-vs-geometry surface error: {max(errs):.2e} (should be < 1e-6)")
        if max(errs) > 1e-3:
            print("  !! large error -> control-point ordering or param range is off")

        mesh = generate_surface_mesh(s.gmsh, size_max=0.15)
        print(f"mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    print("OK")


if __name__ == "__main__":
    main()
