"""Imported fuselage (STEP) + two-section wing, fused, with mixed meshing.

Pipeline:
  1. import a fuselage solid from STEP,
  2. build a 300 mm root-chord wing split spanwise into an INBOARD and an
     OUTBOARD solid at `SPLIT_ETA`,
  3. position the wing (editable) and boolean-fuse fuselage + both wing parts,
  4. STRUCTURED quad mesh on the outboard wing faces with explicit chord/span
     node counts,
  5. UNSTRUCTURED, curvature-adaptive mesh everywhere else,
  6. plot.

Units are millimetres (matching the STEP). The wing is placed at the origin with
span along +X (out the fuselage side); adjust WING_* below to reposition.

    pip install gmsh
    python examples/wing_fuselage_step_mesh_example.py
"""
import math

import numpy as np

from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.occ import GmshSession
from geometry.occ.meshing import generate_surface_mesh, make_structured_quads_uv, set_curvature_sizing
from geometry.occ.ops import (
    describe_faces,
    export_step,
    face_centroid,
    face_tags,
    import_step,
    n_boundary_curves,
)
from geometry.occ.solids import loft_split_solid

STEP_PATH = "/Users/remcovanwoerkom/PycharmProjects/geometry/data/simplified_fuselage_solid.step"

ROOT_CHORD = 300.0       # mm
TIP_CHORD = 150.0        # mm
SEMISPAN = 900.0         # mm  (> fuselage half-width so the wing protrudes)
SPLIT_ETA = 0.4          # spanwise split between inboard and outboard

# Wing placement: rotate span Y->+X so the wing sticks out the fuselage side,
# then translate. Edit these to reposition the wing on the fuselage.
WING_ROTATION = dict(x=0.0, y=0.0, z=0.0, ax=0.0, ay=0.0, az=1.0, angle=-math.pi / 2)
WING_TRANSLATION = (0.0, 0.0, 0.0)

# Faces with centroid X beyond this are "outboard" (clear of the fuselage).
OUTBOARD_X_MIN = 200.0

# Mesh controls.
N_CHORD = 41             # structured nodes around the chord (per side)
N_SPAN = 25              # structured nodes along the outboard span
CURVATURE_N = 96.0       # unstructured: elements per 2*pi of curvature
SIZE_MIN, SIZE_MAX = 5, 30


def build_wing() -> WingSurface:
    shape = WingShape(
        semispan=SEMISPAN,
        chord_distribution=linear_distr(ROOT_CHORD, TIP_CHORD),
        twist_distribution=lambda e: 3.0 * (1 - e),
        airfoil_distribution=lambda e: AirfoilCST(
            upper=CSTPolynomial([0.3, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial([-0.3, -0.2, -0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )
    return WingSurface(shape, n_sections=10, n_chord=60)


def main() -> None:
    wing = build_wing()

    n_in = 5   # spanwise section count inboard
    n_out = 6  # spanwise section count outboard
    etas_in = np.linspace(0.0, SPLIT_ETA, n_in)
    etas_out = np.linspace(SPLIT_ETA, 1.0, n_out)

    with GmshSession() as s:
        occ = s.occ

        # 1. fuselage from STEP
        imported = import_step(s.gmsh, STEP_PATH)
        fus = [tag for (dim, tag) in imported if dim == 3]
        print("imported fuselage solids:", fus)

        # 2. wing solids: BOTH inboard and outboard use the split-face loft so
        # their sections match at the split (otherwise the mismatch shreds the
        # outboard root edge and the faces stop being 4-sided).
        inboard = loft_split_solid(s.gmsh, [wing.section_edges(e, 45) for e in etas_in])
        outboard = loft_split_solid(s.gmsh, [wing.section_edges(e, 45) for e in etas_out])
        occ.synchronize()

        # 3. place the wing, then fuse everything
        wing_dimtags = [(3, inboard), (3, outboard)]
        occ.rotate(wing_dimtags, WING_ROTATION["x"], WING_ROTATION["y"], WING_ROTATION["z"],
                   WING_ROTATION["ax"], WING_ROTATION["ay"], WING_ROTATION["az"],
                   WING_ROTATION["angle"])
        occ.translate(wing_dimtags, *WING_TRANSLATION)
        occ.synchronize()

        # DIAGNOSTIC: outboard solid faces BEFORE fuse — are they 4-sided, and
        # did the rotation put them at large X (the span direction)?
        ob_faces = [t for (d, t) in s.model.getBoundary([(3, outboard)], oriented=False)]
        print("--- outboard faces before fuse ---")
        for t in ob_faces:
            print(f"    face {t}: n_curves={n_boundary_curves(s.gmsh, t)} "
                  f"centroid={np.round(face_centroid(s.gmsh, t), 1)}")

        out, _ = occ.fuse([(3, fus[0])], [(3, inboard), (3, outboard)])
        occ.synchronize()
        print("fused volumes:", out)

        # Export the fused geometry to STEP for inspection in a CAD viewer.
        export_step(s.gmsh, "wing_fuselage_fused.step")
        print("wrote wing_fuselage_fused.step")

        # DIAGNOSTIC: all faces after fuse.
        describe_faces(s.gmsh, span_axis=0, outboard_min=OUTBOARD_X_MIN)

        # 4. structured quads on outboard wing faces (4-sided, centroid beyond fuselage)
        n_struct = 0
        for tag in face_tags(s.gmsh):
            cx = face_centroid(s.gmsh, tag)[0]
            if cx > OUTBOARD_X_MIN and n_boundary_curves(s.gmsh, tag) == 4:  # noqa: PLR2004
                try:
                    if make_structured_quads_uv(s.gmsh, tag, N_CHORD, N_SPAN, span_axis=0):
                        n_struct += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  face {tag}: structured failed ({exc}); leaving unstructured")
        print(f"structured faces: {n_struct}")

        # 5. curvature-adaptive sizing for the unstructured remainder
        set_curvature_sizing(s.gmsh, n_per_2pi=CURVATURE_N, size_min=SIZE_MIN, size_max=SIZE_MAX)

        mesh = generate_surface_mesh(s.gmsh)
        print(f"surface mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

        polygons = [np.array([n.as_array() for n in e.nodes]) for e in mesh.elements]

    _plot(polygons)


def _plot(polygons) -> None:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("MacOSX")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(polygons, facecolor=(0.7, 0.8, 0.95, 0.85), edgecolor="k", linewidths=0.15)
    )
    pts = np.vstack(polygons)
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = 0.5 * (pts.max(0) - pts.min(0)).max()
    for setlim, lo in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        setlim(lo - r, lo + r)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Fuselage(STEP) + 2-section wing — structured outboard, unstructured elsewhere")
    plt.show()


if __name__ == "__main__":
    main()
