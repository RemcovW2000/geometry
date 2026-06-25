"""WingSurface example: a Gordon surface skinned over a WingShape.

Profiles are the airfoil sections; guides run along the leading and trailing
edges. The wing surface passes through every section and edge exactly.

Run:
    python examples/wing_surface_example.py
"""
import matplotlib.pyplot as plt
import numpy as np

from geometry.aero.cst import AirfoilCST, CSTPolynomial
from geometry.aero.utils import linear_distr, elliptic_distr
from geometry.aero.wing_shape import WingShape
from geometry.aero.wing_surface import WingSurface
import matplotlib
matplotlib.use("MacOSX")

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

    wing = WingSurface(wing_shape, n_sections=10, n_chord=80)

    s = wing.sample_grid(120, 40)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(s[:, :, 0], s[:, :, 1], s[:, :, 2], cmap="viridis", alpha=0.8,
                    edgecolor="none")

    ts = np.linspace(0, 1, 80)
    # A few section profiles (red).
    for profile in wing.profiles[:: max(1, len(wing.profiles) // 5)]:
        pts = np.array([profile.point_at_parameter(t).as_array() for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", lw=1.2)
    # Leading and trailing edge guides (blue / black).
    le = np.array([wing.le_guide.point_at_parameter(t).as_array() for t in ts])
    te = np.array([wing.te_guide.point_at_parameter(t).as_array() for t in ts])
    ax.plot(le[:, 0], le[:, 1], le[:, 2], "b-", lw=2, label="leading edge")
    ax.plot(te[:, 0], te[:, 1], te[:, 2], "k-", lw=2, label="trailing edge")

    ax.set_title("Wing surface (Gordon) — sections in red, LE/TE guides")
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z")
    ax.legend()
    try:
        ax.set_box_aspect((1, 3, 1))
    except Exception:
        pass
    plt.show()


if __name__ == "__main__":
    main()
