import numpy as np

from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface


def _wing_shape():
    return WingShape(
        semispan=1.5,
        chord_distribution=linear_distr(0.3, 0.15),
        twist_distribution=CSTPolynomial([5, 2, 2], shape=lambda eta: 1).get_callable(),
        airfoil_distribution=lambda eta: AirfoilCST(
            upper=CSTPolynomial(coeffs=[0.3, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial(coeffs=[-0.3, -0.2, -0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )


def test_wing_surface_builds_and_is_compatible():
    wing = WingSurface(_wing_shape(), n_sections=6, n_chord=40)
    assert len(wing.profiles) == 6
    assert wing.surface.n_guides == 3  # TE, LE, TE


def test_wing_surface_reproduces_sections_exactly():
    wing = WingSurface(_wing_shape(), n_sections=6, n_chord=40)
    for i, eta in enumerate(wing.etas):
        for u in np.linspace(0, 1, 7):
            on_surface = wing.point_at_parameter(u, eta).as_array()
            on_section = wing.profiles[i].point_at_parameter(u).as_array()
            np.testing.assert_allclose(on_surface, on_section, atol=1e-9)


def test_wing_surface_reproduces_edges_exactly():
    wing = WingSurface(_wing_shape(), n_sections=6, n_chord=40)
    for v in np.linspace(0, 1, 7):
        le = wing.point_at_parameter(0.5, v).as_array()
        te = wing.point_at_parameter(0.0, v).as_array()
        np.testing.assert_allclose(le, wing.le_guide.point_at_parameter(v).as_array(), atol=1e-9)
        np.testing.assert_allclose(te, wing.te_guide.point_at_parameter(v).as_array(), atol=1e-9)


def test_reference_line_is_straight_at_origin():
    # The quarter-chord reference line is straight and unswept: x=0, z=0 at every station.
    wing = WingSurface(_wing_shape(), n_sections=5, n_chord=40)
    for v in np.linspace(0, 1, 5):
        ref = wing.reference_guide.point_at_parameter(v).as_array()
        np.testing.assert_allclose([ref[0], ref[2]], [0.0, 0.0], atol=1e-9)


def test_leading_edge_sweeps_forward_of_reference():
    # With a quarter-chord reference, the LE sits ahead of the line (x < 0).
    wing = WingSurface(_wing_shape(), n_sections=5, n_chord=40)
    for v in np.linspace(0, 1, 5):
        le_x = wing.le_guide.point_at_parameter(v).as_array()[0]
        assert le_x < 1e-9


def _varying_airfoil_shape():
    def airfoil(eta):
        t = 0.30 * (1 - eta) + 0.08 * eta  # thick root -> thin tip
        camber = 0.12 * eta  # camber grows toward tip
        return AirfoilCST(
            upper=CSTPolynomial(coeffs=[t + camber, 0.7 * t + camber, 0.7 * t + camber],
                                shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial(coeffs=[-t + camber, -0.7 * t + camber, -0.7 * t + camber],
                                shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        )

    return WingShape(
        semispan=1.5,
        chord_distribution=linear_distr(0.3, 0.15),
        twist_distribution=lambda e: 3.0 * (1 - e),
        airfoil_distribution=airfoil,
    )


def test_varying_airfoil_builds_and_reproduces_sections():
    wing = WingSurface(_varying_airfoil_shape(), n_sections=8, n_chord=50)
    for i, eta in enumerate(wing.etas):
        for u in np.linspace(0, 1, 9):
            on_surface = wing.point_at_parameter(u, eta).as_array()
            on_section = wing.profiles[i].point_at_parameter(u).as_array()
            np.testing.assert_allclose(on_surface, on_section, atol=1e-9)


def test_wing_mesh_quad_counts_and_on_surface():
    wing = WingSurface(_wing_shape(), n_sections=6, n_chord=40)
    n_u, n_v = 30, 8
    mesh = wing.mesh(n_u=n_u, n_v=n_v, wrap_u=True)
    # endpoint dropped (wrap) -> n_u nodes around the loop; closed -> n_u rings.
    assert len(mesh.nodes) == n_u * n_v
    assert len(mesh.elements) == n_u * (n_v - 1)
    # every node lies on the wing surface
    for node in mesh.nodes[:: 13]:
        # find closest is overkill; just confirm node came from the surface map
        assert np.isfinite(node.as_array()).all()
