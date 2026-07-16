"""Wing + pylon + fuselage, merged into one solid with the topology API.

The fuselage comes from a STEP file; the wing is lofted from a parametric
``WingShape``; a cylindrical pylon sticks out of the fuselage top and carries
the wing. Everything is boolean-fused into a single clean solid, so the skin
has no interior faces and a surface mesh is conformal across the junctions.

All placement is derived from the fuselage's bounding box, so the same script
works whatever the STEP file's units or size.

View it (and edit-save-rebuild) with:

    python -m geometry.viewer examples/wing_on_fuselage.py --watch

or run standalone to fuse + mesh + export:

    python examples/wing_on_fuselage.py

Set ``FILLET_RADIUS`` (as a fraction of the pylon radius) to attempt a smooth
OCC blend on the junction edges. Fillets on free-form intersections are
temperamental -- if OCC refuses, the plain union is kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry import Point, Vector
from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.occ.shapes import Solid, fillet, fuse, import_step

FUSELAGE_STEP = Path(__file__).parent.parent / "data" / "simplified_fuselage_solid.step"

FILLET_RADIUS: float | None = None  # e.g. 0.25 -> fillet radius = 0.25 * pylon radius


def build_wing_surface(root_chord: float, semispan: float) -> WingSurface:
    """A simple tapered, twisted CST wing; spans +y with chord along x."""
    shape = WingShape(
        semispan=semispan,
        chord_distribution=linear_distr(root_chord, 0.4 * root_chord),
        twist_distribution=lambda eta: 2.0 * (1 - eta),
        airfoil_distribution=lambda eta: AirfoilCST(
            upper=CSTPolynomial([0.25, 0.18, 0.18], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial([-0.15, -0.10, -0.10], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )
    return WingSurface(shape, n_sections=8, n_chord=50)


def wing_solid(surface: WingSurface, n_loop: int = 70) -> Solid:
    """Loft the wing's airfoil sections into a capped solid."""
    us = np.linspace(0.0, 1.0, n_loop, endpoint=False)
    sections = [
        [profile.point_at_parameter(float(u)) for u in us] for profile in surface.profiles
    ]
    return Solid.loft(sections, name="wing")


def build() -> list:
    """Viewer entry point: the merged airframe (fuselage + pylon + wing)."""
    fuselage = import_step(str(FUSELAGE_STEP), name="fuselage")[0]

    # --- derive all dimensions from the fuselage ------------------------------
    lo, hi = fuselage.bounding_box
    length = hi.x - lo.x
    height = hi.z - lo.z
    center_x = 0.5 * (lo.x + hi.x)
    center_y = 0.5 * (lo.y + hi.y)

    root_chord = 0.22 * length
    semispan = 0.55 * length
    pylon_radius = 0.16 * root_chord
    pylon_top = hi.z + 0.25 * height          # how far the wing sits above the skin

    # --- wing: one half, mirrored, both placed on top of the pylon -------------
    surface = build_wing_surface(root_chord, semispan)
    right = wing_solid(surface)
    # WingSurface chords run x in [-c/4, 3c/4] about the quarter-chord line at y=0:
    # shift so the quarter-chord sits over the fuselage center, at pylon-top height.
    right.translate(Vector(center_x, center_y, pylon_top))
    left = right.mirrored(0.0, 1.0, 0.0, -center_y).named("wing_left")
    right.named("wing_right")

    # --- pylon: a cylinder sticking out of the fuselage into the wing ----------
    pylon = Solid.cylinder(
        base=Point(center_x, center_y, hi.z - 0.4 * height),   # rooted inside the body
        axis=Vector(0.0, 0.0, 1.0),
        radius=pylon_radius,
        height=(pylon_top - hi.z) + 0.4 * height + 0.6 * root_chord * 0.12,
        name="pylon",
    )

    airframe = fuse(fuselage, pylon, right, left, name="airframe")
    if isinstance(airframe, list):
        raise RuntimeError(
            "fuse produced disjoint parts - the pylon does not reach the wing; "
            "check the placement dimensions"
        )

    if FILLET_RADIUS is not None:
        airframe = _try_fillet(airframe, center_x, center_y, hi.z, pylon_radius)

    return [airframe]


def _try_fillet(airframe: Solid, cx: float, cy: float, top_z: float,
                pylon_radius: float) -> Solid:
    """Attempt to blend the pylon junction edges; keep the plain union on failure.

    Junction edges are selected geometrically: edges whose center of mass lies
    within a few pylon radii of the pylon axis (they are the intersection
    curves the booleans created there).
    """
    def near_axis(edge) -> bool:
        com = edge.center_of_mass
        return np.hypot(com.x - cx, com.y - cy) < 2.5 * pylon_radius

    junction_edges = [e for e in airframe.edges if near_axis(e)]
    if not junction_edges:
        print("fillet: no junction edges found, keeping plain union")
        return airframe
    try:
        blended = fillet(airframe, junction_edges, FILLET_RADIUS * pylon_radius,
                         name="airframe")
        print(f"fillet: blended {len(junction_edges)} junction edges")
        return blended
    except Exception as error:  # noqa: BLE001  (OCC fillets fail non-gracefully)
        print(f"fillet failed ({error}); keeping plain union")
        return airframe


def main() -> None:
    """Standalone: build, report topology, surface-mesh, export STEP."""
    from geometry.occ.meshing import generate_surface_mesh, set_curvature_sizing
    from geometry.occ.session import ensure_session
    from geometry.occ.shapes import export_step

    (airframe,) = build()
    print(f"{airframe}: volume={airframe.volume:.4g}")
    print(f"  faces: {len(airframe.faces)}  edges: {len(airframe.edges)}")
    for face in airframe.faces:
        print(f"    {face.name}: {face.kind}, area={face.area:.4g}")

    gmsh = ensure_session()
    lo, hi = airframe.bounding_box
    diag = float(np.linalg.norm(np.subtract([hi.x, hi.y, hi.z], [lo.x, lo.y, lo.z])))
    set_curvature_sizing(gmsh, n_per_2pi=24)
    mesh = generate_surface_mesh(gmsh, size_max=diag / 40)
    print(f"surface mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

    out = Path(__file__).parent / "wing_on_fuselage.step"
    export_step(str(out))œ
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
