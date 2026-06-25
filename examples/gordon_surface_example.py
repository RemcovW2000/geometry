"""Minimal GordonSurface example: a few profile and guide curves.

Builds a compatible 3x4 curve network by sampling points off an analytic
height field, then interpolates them with a Gordon surface. Because the curves
are constructed through a shared grid of points (with explicit, shared
parameters), the network is exactly compatible and the surface passes through
every input curve.

Run:
    python examples/gordon_surface_example.py
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")
import numpy as np

from geometry import GordonSurface, Point
from geometry.curves import InterpolatedLine


def main() -> None:
    # 3 profiles (along u) at these v positions, 4 guides (along v) at these u positions.
    v_params = [0.0, 0.5, 1.0]
    u_params = [0.0, 0.4, 0.7, 1.0]

    xs = [2.0 * u for u in u_params]
    ys = [2.0 * v for v in v_params]

    def height(x: float, y: float) -> float:
        return 0.4 * np.sin(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)

    # Shared grid of points P[i][j] -> guarantees a compatible network.
    grid = [[Point(xs[j], ys[i], height(xs[j], ys[i])) for j in range(len(u_params))]
            for i in range(len(v_params))]

    profiles = [InterpolatedLine(grid[i], params=u_params) for i in range(len(v_params))]
    guides = [
        InterpolatedLine([grid[i][j] for i in range(len(v_params))], params=v_params)
        for j in range(len(u_params))
    ]

    surface = GordonSurface(
        profiles=profiles,
        guides=guides,
        profile_v_params=v_params,
        guide_u_params=u_params,
    )

    s = surface.sample_grid(40, 40)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(s[:, :, 0], s[:, :, 1], s[:, :, 2], cmap="viridis", alpha=0.75,
                    edgecolor="none")

    ts = np.linspace(0, 1, 60)
    for profile in profiles:
        pts = np.array([profile.point_at_parameter(t).as_array() for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", lw=2, label="_")
    for guide in guides:
        pts = np.array([guide.point_at_parameter(t).as_array() for t in ts])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b-", lw=2, label="_")

    ax.set_title("Gordon surface (3 profiles in red, 4 guides in blue)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    main()
