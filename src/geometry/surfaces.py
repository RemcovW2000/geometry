import numpy as np

from geometry.errors import ConstructionError
from geometry import Point
from geometry.curves import InterpolatedLine


class GordonPatch:
    """A Gordon surface patch defined by exactly 2 u-lines and 2 v-lines.

    The patch interpolates between the boundary curves using the Gordon formula:
        S(u, v) = S_u(u, v) + S_v(u, v) - S_uv(u, v)

    where:
        - S_u: linear blend between u_lines[0] and u_lines[1] along v
        - S_v: linear blend between v_lines[0] and v_lines[1] along u
        - S_uv: bilinear interpolation of the 4 corner points

    The boundary curves are exactly preserved:
        - S(u, 0) = u_lines[0](u)
        - S(u, 1) = u_lines[1](u)
        - S(0, v) = v_lines[0](v)
        - S(1, v) = v_lines[1](v)

    Args:
        u_lines: Two InterpolatedLine instances running in the u-direction (v=0 and v=1).
        v_lines: Two InterpolatedLine instances running in the v-direction (u=0 and u=1).
    """

    def __init__(self, u_lines: list[InterpolatedLine], v_lines: list[InterpolatedLine]):
        self.u_lines = u_lines  # u_lines[0] at v=0, u_lines[1] at v=1
        self.v_lines = v_lines  # v_lines[0] at u=0, v_lines[1] at u=1

        self.validate_lines()
        self._corners = self._build_corners()

    def validate_lines(self, tol: float = 1e-6) -> None:
        """Validate that the provided lines form a valid Gordon patch.

        Checks that the corner points of the u_lines and v_lines match.

        Args:
            tol: Tolerance for point equality.
        """
        if len(self.u_lines) != 2:  # noqa: PLR2004
            raise ConstructionError("GordonPatch requires exactly 2 u-direction lines.")
        if len(self.v_lines) != 2:  # noqa: PLR2004
            raise ConstructionError("GordonPatch requires exactly 2 v-direction lines.")

        if not self.u_lines[0].start.is_at_point(self.v_lines[0].start, tol):
            raise ConstructionError(
                "Corner point (u=0,v=0) does not match between u_lines and v_lines."
            )
        if not self.u_lines[0].end.is_at_point(self.v_lines[1].start, tol):
            raise ConstructionError(
                "Corner point (u=1,v=0) does not match between u_lines and v_lines."
            )

        if not self.u_lines[1].start.is_at_point(self.v_lines[0].end, tol):
            raise ConstructionError(
                "Corner point (u=0,v=1) does not match between u_lines and v_lines."
            )
        if not self.u_lines[1].end.is_at_point(self.v_lines[1].end, tol):
            raise ConstructionError(
                "Corner point (u=1,v=1) does not match between u_lines and v_lines."
            )

        if any(
            [
                self.u_lines[0].start.is_at_point(self.u_lines[1].start, tol),
                self.u_lines[0].start.is_at_point(self.u_lines[1].end, tol),
                self.u_lines[0].end.is_at_point(self.u_lines[1].start, tol),
                self.u_lines[0].end.is_at_point(self.u_lines[1].end, tol),
            ]
        ):
            raise ConstructionError(
                "The two u_lines must not have overlapping start- or endpoints."
            )

    def _build_corners(self) -> np.ndarray:
        """Build the 4 corner points array with shape (2, 2, 3).

        corners[i, j] = point at (u=i, v=j) where i,j ∈ {0, 1}
        """
        corners = np.zeros((2, 2, 3), dtype=float)
        # Use u_lines to get corners (could also use v_lines - should be same points)
        corners[0, 0] = self.u_lines[0].point_at_parameter(0.0).as_array()  # u=0, v=0
        corners[1, 0] = self.u_lines[0].point_at_parameter(1.0).as_array()  # u=1, v=0
        corners[0, 1] = self.u_lines[1].point_at_parameter(0.0).as_array()  # u=0, v=1
        corners[1, 1] = self.u_lines[1].point_at_parameter(1.0).as_array()  # u=1, v=1
        return corners

    def _eval_loft_u(self, u: float, v: float) -> np.ndarray:
        """Linear blend between u_lines[0] and u_lines[1] at parameter u."""
        p0 = self.u_lines[0].point_at_parameter(u).as_array()
        p1 = self.u_lines[1].point_at_parameter(u).as_array()
        return (1.0 - v) * p0 + v * p1

    def _eval_loft_v(self, u: float, v: float) -> np.ndarray:
        """Linear blend between v_lines[0] and v_lines[1] at parameter v."""
        p0 = self.v_lines[0].point_at_parameter(v).as_array()
        p1 = self.v_lines[1].point_at_parameter(v).as_array()
        return (1.0 - u) * p0 + u * p1

    def _eval_bilinear(self, u: float, v: float) -> np.ndarray:
        """Bilinear interpolation of the 4 corner points."""
        c00 = self._corners[0, 0]
        c10 = self._corners[1, 0]
        c01 = self._corners[0, 1]
        c11 = self._corners[1, 1]
        return (1.0 - u) * (1.0 - v) * c00 + u * (1.0 - v) * c10 + (1.0 - u) * v * c01 + u * v * c11

    def point_at_parameter(self, u: float, v: float) -> Point:
        """Evaluate the Gordon patch at parameters (u, v).

        Args:
            u: Parameter in u-direction, in [0, 1].
            v: Parameter in v-direction, in [0, 1].

        Returns:
            Point on the surface at (u, v).
        """
        s_u = self._eval_loft_u(u, v)
        s_v = self._eval_loft_v(u, v)
        s_uv = self._eval_bilinear(u, v)

        # Gordon formula: S(u,v) = S_u + S_v - S_uv
        result = s_u + s_v - s_uv
        return Point(float(result[0]), float(result[1]), float(result[2]))

    def sample_grid(self, n_u: int = 20, n_v: int = 20) -> np.ndarray:
        """Sample the patch on a regular grid.

        Returns:
            Array of shape (n_u, n_v, 3) with sampled points.
        """
        us = np.linspace(0.0, 1.0, n_u)
        vs = np.linspace(0.0, 1.0, n_v)
        grid = np.zeros((n_u, n_v, 3), dtype=float)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                pt = self.point_at_parameter(u, v)
                grid[i, j] = [pt.x, pt.y, pt.z]
        return grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create u-direction lines (2 lines at v=0 and v=1)
    u_line_0 = InterpolatedLine([Point(0, 0, 0), Point(1, 0.1, 0), Point(2, 0, 0)])  # v=0
    u_line_1 = InterpolatedLine([Point(0, 0, 1), Point(1, 0.15, 1), Point(2, 0, 1)])  # v=1

    # Create v-direction lines (2 lines at u=0 and u=1)
    v_line_0 = InterpolatedLine([Point(0, 0, 0), Point(0, 0, 1)])  # u=0
    v_line_1 = InterpolatedLine([Point(2, 0, 0), Point(2, 0, 1)])  # u=1

    # Create the Gordon patch
    patch = GordonPatch(u_lines=[u_line_0, u_line_1], v_lines=[v_line_0, v_line_1])

    # Query a point on the surface
    pt = patch.point_at_parameter(0.5, 0.5)
    print(f"Center point: {pt}")

    # Verify boundary curves are preserved
    print("\nBoundary verification:")
    for u in [0.0, 0.5, 1.0]:
        surf_pt = patch.point_at_parameter(u, 0.0)
        line_pt = u_line_0.point_at_parameter(u)
        print(f"  u={u}, v=0: surface={surf_pt.as_array()}, u_line_0={line_pt.as_array()}")

    # Sample for visualization
    grid = patch.sample_grid(n_u=30, n_v=15)

    # Plot the surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    X = grid[:, :, 0]
    Y = grid[:, :, 1]
    Z = grid[:, :, 2]

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.8)

    # Plot the defining curves
    for u_line in [u_line_0, u_line_1]:
        pts = np.array([u_line.point_at_parameter(t).as_array() for t in np.linspace(0, 1, 50)])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", linewidth=2)

    for v_line in [v_line_0, v_line_1]:
        pts = np.array([v_line.point_at_parameter(t).as_array() for t in np.linspace(0, 1, 50)])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b-", linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Gordon Patch (2 u-lines, 2 v-lines)")

    plt.show()
