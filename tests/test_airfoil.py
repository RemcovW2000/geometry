import math

import numpy as np
import pytest

from geometry.aero.airfoil import HINGE_ARC_POINTS, Airfoil, InterpolatedAirfoilCurve
from geometry.aero.cst import CSTPolynomial
from geometry.aero.utils import create_cst_class_function


def _cst_airfoil(trailing_edge_gap=0.0, rounded_trailing_edge=False):
    shape = create_cst_class_function()
    return Airfoil(
        upper=CSTPolynomial([0.17, 0.2, 0.18, 0.15, 0.12], shape=shape),
        lower=CSTPolynomial([-0.12, -0.08, -0.05, -0.03, 0.0], shape=shape),
        trailing_edge_gap=trailing_edge_gap,
        rounded_trailing_edge=rounded_trailing_edge,
    )


def test_interpolated_curve_linear_and_quadratic():
    x = np.linspace(0.0, 1.0, 11)
    linear = InterpolatedAirfoilCurve(x[::-1], x[::-1] ** 2)  # point order does not matter
    assert linear.sample(0.3) == pytest.approx(0.09)
    assert linear.sample(0.35) == pytest.approx(0.5 * (0.09 + 0.16))
    quadratic = InterpolatedAirfoilCurve(x, x**2, kind="quadratic")
    assert quadratic.sample(0.35) == pytest.approx(0.35**2)
    np.testing.assert_allclose(quadratic.sample(np.array([0.05, 0.55])), [0.0025, 0.3025])


def test_interpolated_curve_from_single_surface_dat(tmp_path):
    path = tmp_path / "upper.dat"
    path.write_text("upper\n0.0 0.0\n0.5 0.06\n1.0 0.0\n")
    curve = InterpolatedAirfoilCurve.from_dat(path)
    assert curve.sample(0.25) == pytest.approx(0.03)


def test_airfoil_from_selig_dat_reproduces_the_airfoil(tmp_path):
    cst = _cst_airfoil()
    path = tmp_path / "airfoil.dat"
    cst.to_dat(str(path), n_points=201)
    airfoil = Airfoil.from_dat(path, kind="cubic")
    xs = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(airfoil.y_upper(xs), cst.y_upper(xs), atol=1e-5)
    np.testing.assert_allclose(airfoil.y_lower(xs), cst.y_lower(xs), atol=1e-5)


def test_trailing_edge_gap():
    closed, opened = _cst_airfoil(), _cst_airfoil(trailing_edge_gap=0.01)
    assert closed.thickness_at(1.0) == pytest.approx(0.0, abs=1e-12)
    assert opened.thickness_at(1.0) == pytest.approx(0.01)
    assert opened.thickness_at(0.5) == pytest.approx(closed.thickness_at(0.5) + 0.005)
    assert opened.y_upper(0.0) == closed.y_upper(0.0)  # the LE does not move
    assert opened.leading_edge_radius() == pytest.approx(closed.leading_edge_radius(), rel=0.01)
    assert opened.trailing_edge_angle() < closed.trailing_edge_angle()
    (xu, yu), (xu0, yu0) = opened.coordinates(41, 0.8)[0][-1], closed.coordinates(41, 0.8)[0][-1]
    assert (xu, yu) == pytest.approx((xu0, yu0 + 0.005))


def test_zero_gap_matches_the_base_shape():
    airfoil = _cst_airfoil()
    upper, lower = airfoil.coordinates(81, 0.8)
    xs = np.array([x for x, _ in upper])
    assert [y for _, y in upper] == airfoil.upper.sample(xs).tolist()
    assert not airfoil.inverted_surface()


def test_rounded_trailing_edge():
    assert _cst_airfoil(0.0, rounded_trailing_edge=True).trailing_edge is None  # nothing to round
    cut, rounded = _cst_airfoil(0.01), _cst_airfoil(0.01, rounded_trailing_edge=True)
    te = rounded.trailing_edge
    assert te.xc + te.r == 1.0
    assert rounded.y_upper(1.0) == rounded.y_lower(1.0) == te.yc  # the surfaces meet at x = 1
    assert rounded.y_upper(0.9) == cut.y_upper(0.9)  # no rescaling: the opened surfaces upstream
    assert te.r == pytest.approx(0.005, rel=0.1)  # a bit more than gap/2: it sits upstream of x = 1
    h = 1e-7  # tangent (C1) where the arc joins each surface
    junctions = ((rounded.y_upper, te.x_tangent_upper), (rounded.y_lower, te.x_tangent_lower))
    for y, x_tangent in junctions:
        left = (y(x_tangent) - y(x_tangent - h)) / h
        right = (y(x_tangent + h) - y(x_tangent)) / h
        assert left == pytest.approx(right, abs=1e-3)


def test_leading_edge_radius_of_the_closed_contour():
    shape = create_cst_class_function()
    round_nose = Airfoil(CSTPolynomial([1.0], shape=shape), CSTPolynomial([-1.0], shape=shape))
    assert round_nose.leading_edge_radius() == pytest.approx(0.5, abs=3e-3)  # y = +-sqrt(x)
    half_nose = Airfoil(CSTPolynomial([1.0], shape=shape), CSTPolynomial([0.0, -1.0], shape=shape))
    assert half_nose.leading_edge_radius() == pytest.approx(0.125, abs=3e-3)  # a quarter of it


def test_flap():
    plain = _cst_airfoil()
    assert Airfoil(plain.upper, plain.lower, flap_chord=0.25).flap is None  # nothing deflected
    down = Airfoil(plain.upper, plain.lower, flap_chord=0.25, flap_deflection=20.0)
    up = Airfoil(plain.upper, plain.lower, flap_chord=0.25, flap_deflection=-20.0)
    y_h = 0.5 * (plain.y_upper(0.75) + plain.y_lower(0.75))  # halfway between the surfaces
    for af in (down, up):
        assert (af.flap.x_hinge, af.flap.y_hinge) == pytest.approx((0.75, y_h))
        assert af.y_upper(0.7) == plain.y_upper(0.7) and af.y_lower(0.7) == plain.y_lower(0.7)
        assert af.flap.upper[0][-1] == af.flap.lower[0][-1]  # the sharp TE stays closed
    # convex side (upper for a downward deflection): an arc around the hinge
    x, y = (p[:HINGE_ARC_POINTS] for p in down.flap.upper)
    np.testing.assert_allclose(np.hypot(x - 0.75, y - y_h), 0.5 * plain.thickness_at(0.75))
    # concave side: a sharp corner on the fixed lower surface, just ahead of the hinge
    x_corner, y_corner = down.flap.lower[0][0], down.flap.lower[1][0]
    assert x_corner < 0.75 and y_corner == pytest.approx(plain.y_lower(x_corner))
    # the TE is rotated 20 deg down about the hinge
    d = math.radians(20.0)
    dx, dy = 0.25, -y_h  # the TE (1, 0) relative to the hinge
    te = (0.75 + dx * math.cos(d) + dy * math.sin(d), y_h - dx * math.sin(d) + dy * math.cos(d))
    assert (down.flap.upper[0][-1], down.flap.upper[1][-1]) == pytest.approx(te)
    assert up.flap.upper[1][-1] > 0.0  # and up for a negative deflection


def test_flap_on_a_surface_still_diverging_at_the_hinge():
    # the lower surface still falls at the hinge, so the rotated hinge point lies outside it
    shape = create_cst_class_function()
    upper = CSTPolynomial([0.17, 0.2, 0.18, 0.15, 0.12], shape=shape)
    lower = CSTPolynomial([0.0, 0.0, 0.0, 0.0, -1.5], shape=shape)
    plain = Airfoil(upper, lower)
    assert plain.y_lower(0.801) < plain.y_lower(0.799)
    flap = Airfoil(upper, lower, flap_chord=0.2, flap_deflection=5.0).flap
    x_corner, y_corner = flap.lower[0][0], flap.lower[1][0]
    assert x_corner < 0.8 and y_corner == pytest.approx(plain.y_lower(x_corner))
    assert all(np.all(np.diff(p[0]) > 0.0) for p in (flap.upper, flap.lower))
