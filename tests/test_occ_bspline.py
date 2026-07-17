import numpy as np
import pytest

from geometry import GordonSurface, Point
from geometry.curves import InterpolatedLine
from geometry.occ.bspline import surface_to_bspline_data


def _wavy_surface():
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


def test_bspline_interpolation_reproduces_surface_at_nodes():
    surf = _wavy_surface()
    us = list(np.linspace(0, 1, 14))
    vs = list(np.linspace(0, 1, 12))
    data = surface_to_bspline_data(surf, us, vs)

    assert data.control_points.shape == (14, 12, 3)
    # clamped knot vectors: sum(mult) == n_ctrl + degree + 1
    assert sum(data.mult_u) == data.control_points.shape[0] + data.degree_u + 1
    assert sum(data.mult_v) == data.control_points.shape[1] + data.degree_v + 1

    err = max(
        np.linalg.norm(data.evaluate(u, v) - surf.point_at_parameter(u, v).as_array())
        for u in us for v in vs
    )
    assert err < 1e-9


def test_bspline_interpolation_corners_exact():
    surf = _wavy_surface()
    data = surface_to_bspline_data(surf, list(np.linspace(0, 1, 10)), list(np.linspace(0, 1, 10)))
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            np.testing.assert_allclose(
                data.evaluate(a, b), surf.point_at_parameter(a, b).as_array(), atol=1e-9
            )


# ---- gmsh-backed tests (skipped automatically if gmsh isn't installed) ----

gmsh = pytest.importorskip("gmsh")


def test_gmsh_surface_matches_geometry_surface():
    from geometry.occ import GmshSession
    from geometry.occ.bspline import add_bspline_surface, surface_to_bspline_data

    surf = _wavy_surface()
    us = list(np.linspace(0, 1, 12))
    vs = list(np.linspace(0, 1, 12))
    with GmshSession() as s:
        data = surface_to_bspline_data(surf, us, vs)
        tag = add_bspline_surface(s.gmsh, data)
        s.occ.synchronize()
        errs = []
        for u in np.linspace(0, 1, 6):
            for v in np.linspace(0, 1, 6):
                xyz = np.array(s.model.getValue(2, tag, [float(u), float(v)]))
                errs.append(np.linalg.norm(xyz - surf.point_at_parameter(u, v).as_array()))
    assert max(errs) < 1e-6
