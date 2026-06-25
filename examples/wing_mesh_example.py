"""Mesh a WingSurface and plot the resulting quad mesh.

Builds the wing Gordon surface, tessellates it into a quad mesh with a chosen
u (chordwise) and v (spanwise) station spacing, and draws the mesh.

Run:
    python examples/wing_mesh_example.py
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import elliptic_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface


def main() -> None:
    wing_shape = WingShape(
        semispan=1.5,
        chord_distribution=elliptic_distr(0.3),
        twist_distribution=CSTPolynomial([5, 2, 2], shape=lambda eta: 1).get_callable(),
        airfoil_distribution=lambda eta: AirfoilCST(
            upper=CSTPolynomial(coeffs=[0.3, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial(coeffs=[-0.3, -0.2, -0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )

    wing = WingSurface(wing_shape, n_sections=12, n_chord=80)

    # Chordwise clustered near LE/TE (cosine), spanwise clustered near the tip.
    n_u, n_v = 60, 100
    u_params = list(0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_u, endpoint=False))))
    u_params = list(np.linspace(0, 1, n_u, endpoint=False))
    v_params = list(np.linspace(0, 1, n_v) ** 1.3)

    mesh = wing.mesh(u_params=u_params, v_params=v_params, wrap_u=True)
    print(f"mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} quad elements")

    polygons = [np.array([n.as_array() for n in elem.nodes]) for elem in mesh.elements]

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(polygons, facecolor=(0.6, 0.75, 0.95, 0.85), edgecolor="k",
                            linewidths=0.3)
    ax.add_collection3d(coll)

    pts = np.vstack(polygons)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    r = 0.5 * (maxs - mins).max()
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z")
    ax.set_title(f"Wing surface mesh ({len(mesh.elements)} quads)")
    ax.view_init(elev=22, azim=-65)
    plt.show()


if __name__ == "__main__":
    main()
