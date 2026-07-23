"""Wingtips: trim the wing, grow a cap from its cut section, sew into ONE surface.

``wing.fit_tip(TipClass, ...)`` inserts the cut station into the wing's
skinning stations (exact seam), builds the cap from the wing's own airfoil
distribution and stores it as ``wing.tip``; ``wing.tipped_surface`` is the
sewn result. Shown: the Hoerner cap (closes on the raised camber line) and a
swept/drooped ParametricTip.

Run:
    python examples/hoerner_wingtip_example.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
from geometry.aero.wingtip import HoernerTip, ParametricTip

matplotlib.use("MacOSX")

TIP_CAP_CHORDS = 0.3  # cap length as a fraction of the tip chord


def build_wing() -> WingSurface:
    wing_shape = WingShape(
        semispan=1.5,
        chord_distribution=linear_distr(0.3, 0.15),
        twist_distribution=linear_distr(0.0, -3.0),
        airfoil_distribution=lambda eta: AirfoilCST(
            upper=CSTPolynomial(coeffs=[0.25, 0.2, 0.2], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
            lower=CSTPolynomial(coeffs=[-0.15, -0.05, -0.05], shape=lambda u: u**0.5 * (1 - u) ** 1.0),
        ),
    )
    return WingSurface(wing_shape, n_sections=10, n_chord=60)


def plot_tipped(ax, wing: WingSurface, title: str) -> None:
    tip = wing.tip
    full = wing.tipped_surface
    us = np.linspace(0.0, 1.0, 180)
    vs = np.concatenate([np.linspace(0.0, tip.eta_cut, 50),
                         np.linspace(tip.eta_cut, 1.0, 40)[1:]])
    s = np.array([[full.point_at_parameter(u, v).as_array() for u in us] for v in vs])
    ax.plot_surface(s[:, :, 0], s[:, :, 1], s[:, :, 2], cmap="viridis",
                    alpha=0.9, edgecolor="none")
    edge = np.array([tip.sharp_edge.point_at_parameter(t).as_array()
                     for t in np.linspace(0, 1, 120)])
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], "k-", lw=2.5)

    pts = s[vs > 0.85].reshape(-1, 3)
    c = 0.5 * (pts.min(0) + pts.max(0))
    r = 1.2 * (pts.max(0) - pts.min(0)).max()
    for setlim, lo in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        setlim(lo - r / 2, lo + r / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=-25, azim=-35)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_title(title)


if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 6))

    wing = build_wing()
    cap_length = TIP_CAP_CHORDS * float(wing.wing_shape.chord_distribution(1.0))
    wing.fit_tip(HoernerTip, length=cap_length)
    plot_tipped(fig.add_subplot(121, projection="3d"), wing,
                "HoernerTip: closes on the raised camber line")

    wing2 = build_wing()
    wing2.fit_tip(ParametricTip, length=cap_length, tip_chord=wing2.wing_shape.chord_distribution(1.0) * 0.9,
                  delta_x=0.03, delta_z=0.015, camber_scale=0.5)
    plot_tipped(fig.add_subplot(122, projection="3d"), wing2,
                "ParametricTip: swept back + raised, half camber")

    plt.show()
