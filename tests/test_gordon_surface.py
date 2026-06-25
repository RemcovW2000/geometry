import numpy as np
import pytest

from geometry import GordonPatch, GordonSurface, Point
from geometry.curves import InterpolatedLine
from geometry.errors import ConstructionError


def _compatible_network():
    """A deliberately compatible, non-planar 3x3 network.

    Profiles run in x at y = 0, 1, 2 with an identical z-bump z(x)=[0, 0.5, 0],
    so all profiles share the chord-length parameterization u=[0, 0.5, 1].
    Guides run straight in y at x = 0, 1, 2, sharing v=[0, 0.5, 1]. Every
    profile therefore crosses every guide at the same parameter -> compatible.
    """
    z = [0.0, 0.5, 0.0]
    ys = [0.0, 1.0, 2.0]
    profiles = [
        InterpolatedLine([Point(0.0, y, z[0]), Point(1.0, y, z[1]), Point(2.0, y, z[2])])
        for y in ys
    ]
    guides = [
        InterpolatedLine([Point(x, 0.0, zx), Point(x, 1.0, zx), Point(x, 2.0, zx)])
        for x, zx in zip([0.0, 1.0, 2.0], z)
    ]
    v_params = [0.0, 0.5, 1.0]
    u_params = [0.0, 0.5, 1.0]
    return profiles, guides, u_params, v_params


def test_reduces_to_gordon_patch():
    """With 2 profiles and 2 guides, GordonSurface == GordonPatch."""
    u_line_0 = InterpolatedLine([Point(0, 0, 0), Point(1, 0.1, 0), Point(2, 0, 0)])
    u_line_1 = InterpolatedLine([Point(0, 0, 1), Point(1, 0.15, 1), Point(2, 0, 1)])
    v_line_0 = InterpolatedLine([Point(0, 0, 0), Point(0, 0, 1)])
    v_line_1 = InterpolatedLine([Point(2, 0, 0), Point(2, 0, 1)])

    patch = GordonPatch(u_lines=[u_line_0, u_line_1], v_lines=[v_line_0, v_line_1])
    surface = GordonSurface(
        profiles=[u_line_0, u_line_1],
        guides=[v_line_0, v_line_1],
        profile_v_params=[0.0, 1.0],
        guide_u_params=[0.0, 1.0],
    )

    np.testing.assert_allclose(
        patch.sample_grid(11, 9), surface.sample_grid(11, 9), atol=1e-9
    )


def test_reproduces_profiles_exactly():
    """Surface evaluated at v = v_i must trace profile i exactly."""
    profiles, guides, u_params, v_params = _compatible_network()
    surf = GordonSurface(profiles, guides, v_params, u_params)

    for i, profile in enumerate(profiles):
        for u in np.linspace(0, 1, 7):
            on_surface = surf.point_at_parameter(u, v_params[i]).as_array()
            on_curve = profile.point_at_parameter(u).as_array()
            np.testing.assert_allclose(on_surface, on_curve, atol=1e-6)


def test_reproduces_guides_exactly():
    """Surface evaluated at u = u_j must trace guide j exactly."""
    profiles, guides, u_params, v_params = _compatible_network()
    surf = GordonSurface(profiles, guides, v_params, u_params)

    for j, guide in enumerate(guides):
        for v in np.linspace(0, 1, 7):
            on_surface = surf.point_at_parameter(u_params[j], v).as_array()
            on_curve = guide.point_at_parameter(v).as_array()
            np.testing.assert_allclose(on_surface, on_curve, atol=1e-6)


def test_interior_point_on_bump():
    """The surface center should sit on the shared z-bump (z=0.5 at x=1)."""
    profiles, guides, u_params, v_params = _compatible_network()
    surf = GordonSurface(profiles, guides, v_params, u_params)
    center = surf.point_at_parameter(0.5, 0.5)
    np.testing.assert_allclose(center.as_array(), [1.0, 1.0, 0.5], atol=1e-6)


def test_incompatible_network_raises():
    """Wrong guide u-params (intersections don't line up) must be rejected."""
    profiles, guides, _u, v_params = _compatible_network()
    with pytest.raises(ConstructionError, match="not compatible"):
        GordonSurface(profiles, guides, v_params, guide_u_params=[0.0, 0.7, 1.0])


def test_node_params_must_span_unit_interval():
    profiles, guides, u_params, _v = _compatible_network()
    with pytest.raises(ConstructionError, match="start at 0 and end at 1"):
        GordonSurface(profiles, guides, profile_v_params=[0.0, 0.5, 0.9], guide_u_params=u_params)


def test_node_params_must_be_increasing():
    profiles, guides, u_params, _v = _compatible_network()
    with pytest.raises(ConstructionError, match="strictly increasing"):
        GordonSurface(profiles, guides, profile_v_params=[0.0, 0.0, 1.0], guide_u_params=u_params)


def test_requires_two_curves_each():
    line = InterpolatedLine([Point(0, 0, 0), Point(1, 0, 0)])
    with pytest.raises(ConstructionError):
        GordonSurface([line], [line, line], [0.0], [0.0, 1.0])
