from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry import Point, Vector
from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.occ.shapes import Solid, fuse, import_step
from geometry.viewer import Viewer

# ----------------------------------------------------------------------------------
# Import fuselage
# ----------------------------------------------------------------------------------
FUSELAGE_STEP_PATH = Path(__file__).parent.parent / "data" / "simplified_fuselage_solid.step"

fuselage = import_step(str(FUSELAGE_STEP_PATH), name="fuselage")[0]

root_chord = 300
semispan = 1500
pylon_radius = root_chord * 0.9
wing_root_qq_position = Point(0, 0, 186)

# ----------------------------------------------------------------------------------
# define wing shape
# ----------------------------------------------------------------------------------
shape = WingShape(
    semispan=semispan,
    chord_distribution=linear_distr(root_chord, 0.4 * root_chord),
    twist_distribution=lambda eta: 2.0 * (1 - eta),
    airfoil_distribution=lambda eta: AirfoilCST(
        upper=CSTPolynomial([0.25, 0.18, 0.18], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        lower=CSTPolynomial([-0.15, -0.10, -0.10], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
    ),
)

surface = WingSurface(shape, n_sections=8, n_chord=50)

# loft the wing between sections:
us = np.linspace(0.0, 1.0, 8, endpoint=False)
sections = [
    [profile.point_at_parameter(float(u)) for u in us] for profile in
    surface.profiles
]

# create solid, and mirror:
right_wing_solid = Solid.loft(sections, name="wing")
right_wing_solid.translate(Vector(wing_root_qq_position.x, wing_root_qq_position.y, wing_root_qq_position.z))
right_wing_solid.named("wing_right")
left_wing_solid = right_wing_solid.mirrored(0.0, 1.0, 0.0, -wing_root_qq_position.y).named("wing_left")

# ----------------------------------------------------------------------------------
# define pylon
# ----------------------------------------------------------------------------------
# --- pylon: a cylinder sticking out of the fuselage into the wing ----------
pylon = Solid.cylinder(
    base=Point(wing_root_qq_position.x, wing_root_qq_position.y, 0),  # rooted inside the body
    axis=Vector(0.0, 0.0, 1.0),
    radius=pylon_radius,
    height=300,
    name="pylon",
)

Viewer([fuselage, left_wing_solid, right_wing_solid, pylon], name="wing_on_fuselage").show()
