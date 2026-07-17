import numpy as np

from geometry import GordonSurface, Point
from geometry.curves import InterpolatedLine
from geometry.mesh import Element, Mesh, Node, mesh_surface


def _simple_surface():
    v_params = [0.0, 0.5, 1.0]
    u_params = [0.0, 0.5, 1.0]

    def z(x, y):
        return 0.3 * np.sin(np.pi * x) * np.cos(np.pi * y)

    xs = [u for u in u_params]
    ys = [v for v in v_params]
    grid = [[Point(xs[j], ys[i], z(xs[j], ys[i])) for j in range(3)] for i in range(3)]
    profiles = [InterpolatedLine(grid[i], params=u_params) for i in range(3)]
    guides = [InterpolatedLine([grid[i][j] for i in range(3)], params=v_params) for j in range(3)]
    return GordonSurface(profiles, guides, v_params, u_params)


def test_mesh_surface_counts_open():
    surf = _simple_surface()
    us = list(np.linspace(0, 1, 6))
    vs = list(np.linspace(0, 1, 4))
    mesh = mesh_surface(surf, us, vs)
    assert isinstance(mesh, Mesh)
    assert len(mesh.nodes) == 6 * 4
    assert len(mesh.elements) == (6 - 1) * (4 - 1)
    assert all(isinstance(e, Element) and len(e.nodes) == 4 for e in mesh.elements)
    assert all(isinstance(n, Node) for n in mesh.nodes)


def test_mesh_surface_nodes_lie_on_surface():
    surf = _simple_surface()
    us = list(np.linspace(0, 1, 5))
    vs = list(np.linspace(0, 1, 5))
    mesh = mesh_surface(surf, us, vs)
    # node grid is row-major in u then v
    k = 0
    for u in us:
        for v in vs:
            expected = surf.point_at_parameter(u, v).as_array()
            np.testing.assert_allclose(mesh.nodes[k].as_array(), expected, atol=1e-12)
            k += 1


def test_mesh_surface_wrap_u_adds_seam_ring():
    surf = _simple_surface()
    us = list(np.linspace(0, 1, 6, endpoint=False))
    vs = list(np.linspace(0, 1, 4))
    open_mesh = mesh_surface(surf, us, vs, wrap_u=False)
    wrapped = mesh_surface(surf, us, vs, wrap_u=True)
    # wrapping adds one extra ring of elements (n_v - 1) and no extra nodes
    assert len(wrapped.nodes) == len(open_mesh.nodes)
    assert len(wrapped.elements) == len(open_mesh.elements) + (len(vs) - 1)
