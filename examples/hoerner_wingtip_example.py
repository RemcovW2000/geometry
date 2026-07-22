"""Hoerner wingtip: trim the wing surface, grow the cap from its edge curve,
sew both into ONE surface.

The cap's profile is the trimmed wing's edge iso-curve -- the very same Curve
object -- so the seam is exact by construction and SewnSurface stitches the two
into a single (u, v) surface, sampled here in one go.

Run:
    python examples/hoerner_wingtip_example.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from geometry import SewnSurface, Vector
from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.aero.wingtip import HoernerTip

matplotlib.use("MacOSX")

TIP_CAP_CHORDS = 0.3  # cap length as a fraction of the tip chord


if __name__ == "__main__":
    wing_shape = WingShape(
        semispan=1.5,
        chord_distribution=linear_distr(0.3, 0.15),
        twist_distribution=linear_distr(0.0, -3.0),
        airfoil_distribution=lambda eta: AirfoilCST(
            upper=CSTPolynomial(coeffs=[0.25, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial(coeffs=[-0.15, -0.05, -0.05], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )
    wing = WingSurface(wing_shape, n_sections=10, n_chord=60)

    # Trim the wing surface (v is spanwise), grow the cap from the edge curve.
    tip_chord = float(wing_shape.chord_distribution(1.0))
    cap_length = TIP_CAP_CHORDS * tip_chord
    eta_tip = (wing_shape.semispan - cap_length) / wing_shape.semispan

    trimmed = wing.surface.trimmed(v=(0.0, eta_tip))
    profile = trimmed.iso_v(1.0)  # == wing.surface.iso_v(eta_tip), same geometry
    tip = HoernerTip(profile, direction=Vector(0, 1, 0), length=cap_length)
    full = SewnSurface([trimmed, tip], along="v", breaks=[0.0, eta_tip, 1.0])
    print(f"sewn OK: seam at v = {eta_tip:.4f}, cap length {cap_length*1e3:.0f} mm")

    # One surface, one sample: v > eta_tip is automatically the cap.
    us = np.linspace(0.0, 1.0, 90)
    vs = np.concatenate([np.linspace(0.0, eta_tip, 25), np.linspace(eta_tip, 1.0, 20)[1:]])
    s = np.array([[full.point_at_parameter(u, v).as_array() for u in us] for v in vs])

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(s[:, :, 0], s[:, :, 1], s[:, :, 2], cmap="viridis",
                    alpha=0.9, edgecolor="none")

    # Seam profile (red) and the sharp lower edge (black), both as Curves.
    for curve, style, lw, label in ((profile, "r-", 1.5, "seam profile"),
                                    (tip.sharp_edge, "k-", 2.5, "sharp lower edge")):
        pts = np.array([curve.point_at_parameter(t).as_array() for t in np.linspace(0, 1, 120)])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], style, lw=lw, label=label)

    # Equal aspect around the tip region, seen from ahead and below.
    pts = s[vs > 0.8].reshape(-1, 3)
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = 1.2 * (pts.max(0) - pts.min(0)).max()
    for setlim, lo in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        setlim(lo - r / 2, lo + r / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=-25, azim=-35)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z")
    ax.set_title("Wing + Hoerner cap sewn into one surface")
    ax.legend()
    plt.show()