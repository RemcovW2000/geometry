import numpy as np
from scipy.interpolate import make_interp_spline

from geometry.errors import ConstructionError
from geometry import Point
from geometry.curves import Curve, InterpolatedLine, IsoCurve


class Surface:
    """Base for parametric surfaces mapping (u, v) in [0, 1]^2 to points."""

    def point_at_parameter(self, u: float, v: float) -> Point:
        raise NotImplementedError

    def sample_grid(self, n_u: int = 20, n_v: int = 20) -> np.ndarray:
        """Sample the surface on a regular (n_u, n_v, 3) grid."""
        grid = np.zeros((n_u, n_v, 3), dtype=float)
        for a, u in enumerate(np.linspace(0.0, 1.0, n_u)):
            for b, v in enumerate(np.linspace(0.0, 1.0, n_v)):
                pt = self.point_at_parameter(u, v)
                grid[a, b] = [pt.x, pt.y, pt.z]
        return grid

    def iso_u(self, u: float) -> IsoCurve:
        """The curve of constant u (varying v), referencing this surface."""
        return IsoCurve(self, u=u)

    def iso_v(self, v: float) -> IsoCurve:
        """The curve of constant v (varying u), referencing this surface."""
        return IsoCurve(self, v=v)

    def trimmed(self, u: tuple[float, float] = (0.0, 1.0),
                v: tuple[float, float] = (0.0, 1.0)) -> "TrimmedSurface":
        """Rectangular parameter-space trim, reparameterized to [0, 1]^2."""
        return TrimmedSurface(self, u=u, v=v)


class GordonPatch(Surface):
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


class GordonSurface(Surface):
    """A Gordon surface interpolating a compatible network of curves.

    The surface is the Boolean sum of three surfaces (Gordon 1969):

        S(u, v) = S_u(u, v) + S_v(u, v) - T(u, v)

    where:
        - S_u: skinning surface interpolating all profile curves f_i(u)
        - S_v: skinning surface interpolating all guide curves g_j(v)
        - T:   tensor-product surface interpolating the grid of intersection
               points P_ij = f_i(u_j) = g_j(v_i)

    This is the strict (vector) variant: it requires the network to already be
    *compatible*, i.e. every profile crosses guide ``j`` at the **same**
    parameter ``u_j`` (shared across all profiles), and every guide crosses
    profile ``i`` at the same parameter ``v_i`` (shared across all guides). No
    reparameterization is performed; the input curves are evaluated by their own
    ``point_at_parameter`` and are never re-fit, so the profiles and guides are
    reproduced exactly.

    With exactly 2 profiles and 2 guides (and the linear blending that results),
    this reduces to :class:`GordonPatch`.

    Args:
        profiles: Profile curves f_i(u), ordered by increasing v position.
        guides: Guide curves g_j(v), ordered by increasing u position.
        profile_v_params: The v-parameter v_i of each profile (length M). Must be
            strictly increasing with v_0 = 0 and v_{M-1} = 1.
        guide_u_params: The u-parameter u_j of each guide (length N). Must be
            strictly increasing with u_0 = 0 and u_{N-1} = 1.
        tol: Tolerance for the compatibility check on intersection points.
    """

    def __init__(
        self,
        profiles: list[Curve],
        guides: list[Curve],
        profile_v_params: list[float],
        guide_u_params: list[float],
        tol: float = 1e-6,
    ):
        self.profiles = list(profiles)
        self.guides = list(guides)
        self.v_nodes = np.asarray(profile_v_params, dtype=float)
        self.u_nodes = np.asarray(guide_u_params, dtype=float)
        self.tol = tol

        self._validate()

        # Cross-interpolation degrees: linear for 2 curves, else up to cubic.
        self._deg_v = min(3, len(self.profiles) - 1)
        self._deg_u = min(3, len(self.guides) - 1)

        # Grid of intersection points P[i, j] = f_i(u_j), shape (M, N, 3).
        self._grid = self._build_grid()

    @property
    def n_profiles(self) -> int:
        """Number of profile curves (M)."""
        return len(self.profiles)

    @property
    def n_guides(self) -> int:
        """Number of guide curves (N)."""
        return len(self.guides)

    def _validate(self) -> None:
        """Validate counts, node parameters, and network compatibility."""
        m, n = len(self.profiles), len(self.guides)
        if m < 2:  # noqa: PLR2004
            raise ConstructionError("GordonSurface requires at least 2 profile curves.")
        if n < 2:  # noqa: PLR2004
            raise ConstructionError("GordonSurface requires at least 2 guide curves.")
        if len(self.v_nodes) != m:
            raise ConstructionError(
                f"profile_v_params length ({len(self.v_nodes)}) must equal "
                f"the number of profiles ({m})."
            )
        if len(self.u_nodes) != n:
            raise ConstructionError(
                f"guide_u_params length ({len(self.u_nodes)}) must equal "
                f"the number of guides ({n})."
            )

        self._check_nodes(self.v_nodes, "profile_v_params")
        self._check_nodes(self.u_nodes, "guide_u_params")

        self._check_compatibility()

    @staticmethod
    def _check_nodes(nodes: np.ndarray, name: str) -> None:
        """Check that node parameters are strictly increasing and span [0, 1]."""
        if not np.all(np.diff(nodes) > 0):
            raise ConstructionError(f"{name} must be strictly increasing.")
        if abs(nodes[0]) > 1e-9 or abs(nodes[-1] - 1.0) > 1e-9:  # noqa: PLR2004
            raise ConstructionError(
                f"{name} must start at 0 and end at 1 (got {nodes[0]} .. {nodes[-1]})."
            )

    def _check_compatibility(self) -> None:
        """Verify f_i(u_j) coincides with g_j(v_i) for every (i, j) within tol."""
        for i, profile in enumerate(self.profiles):
            for j, guide in enumerate(self.guides):
                p_profile = profile.point_at_parameter(float(self.u_nodes[j])).as_array()
                p_guide = guide.point_at_parameter(float(self.v_nodes[i])).as_array()
                dist = float(np.linalg.norm(p_profile - p_guide))
                if dist > self.tol:
                    raise ConstructionError(
                        f"Network not compatible at profile {i} / guide {j}: "
                        f"f_{i}(u_{j}={self.u_nodes[j]:.4f}) and "
                        f"g_{j}(v_{i}={self.v_nodes[i]:.4f}) differ by {dist:.2e} > tol={self.tol:.2e}."
                    )

    def _build_grid(self) -> np.ndarray:
        """Build the (M, N, 3) array of intersection points P_ij = f_i(u_j)."""
        m, n = len(self.profiles), len(self.guides)
        grid = np.zeros((m, n, 3), dtype=float)
        for i, profile in enumerate(self.profiles):
            for j in range(n):
                grid[i, j] = profile.point_at_parameter(float(self.u_nodes[j])).as_array()
        return grid

    @staticmethod
    def _interp(nodes: np.ndarray, values: np.ndarray, query: float, degree: int) -> np.ndarray:
        """Interpolate vector-valued ``values`` defined at ``nodes``, evaluated at ``query``.

        Uses a B-spline interpolant of the given degree. The same operator is
        used for the skinning surfaces and the tensor-product term, which is what
        makes the Boolean sum reproduce the input curves exactly.
        """
        spline = make_interp_spline(nodes, np.asarray(values, dtype=float), k=degree, axis=0)
        return np.asarray(spline(query), dtype=float)

    def _eval_loft_u(self, u: float, v: float) -> np.ndarray:
        """S_u: evaluate all profiles at u, then interpolate across v."""
        pts = np.array([p.point_at_parameter(u).as_array() for p in self.profiles])
        return self._interp(self.v_nodes, pts, v, self._deg_v)

    def _eval_loft_v(self, u: float, v: float) -> np.ndarray:
        """S_v: evaluate all guides at v, then interpolate across u."""
        pts = np.array([g.point_at_parameter(v).as_array() for g in self.guides])
        return self._interp(self.u_nodes, pts, u, self._deg_u)

    def _eval_tensor(self, u: float, v: float) -> np.ndarray:
        """T: interpolate the intersection grid in v (per guide), then in u."""
        # For each guide column j, interpolate the M grid points across v.
        t_cols = np.array(
            [self._interp(self.v_nodes, self._grid[:, j, :], v, self._deg_v) for j in range(self.n_guides)]
        )
        # Then interpolate those N points across u.
        return self._interp(self.u_nodes, t_cols, u, self._deg_u)

    def point_at_parameter(self, u: float, v: float) -> Point:
        """Evaluate the Gordon surface at parameters (u, v), each in [0, 1]."""
        result = self._eval_loft_u(u, v) + self._eval_loft_v(u, v) - self._eval_tensor(u, v)
        return Point(float(result[0]), float(result[1]), float(result[2]))

    def sample_grid(self, n_u: int = 20, n_v: int = 20) -> np.ndarray:
        """Sample the surface on a regular (n_u, n_v, 3) grid."""
        us = np.linspace(0.0, 1.0, n_u)
        vs = np.linspace(0.0, 1.0, n_v)
        grid = np.zeros((n_u, n_v, 3), dtype=float)
        for a, u in enumerate(us):
            for b, v in enumerate(vs):
                pt = self.point_at_parameter(u, v)
                grid[a, b] = [pt.x, pt.y, pt.z]
        return grid


class TrimmedSurface(Surface):
    """A rectangular parameter-space trim of a surface, reparameterized to [0, 1]^2.

    References the base surface directly (no geometry is copied), so e.g.
    ``TrimmedSurface(s, v=(0, 0.9)).iso_v(1.0)`` is exactly ``s.iso_v(0.9)``.
    """

    def __init__(self, base: Surface, u: tuple[float, float] = (0.0, 1.0),
                 v: tuple[float, float] = (0.0, 1.0)):
        for name, (lo, hi) in (("u", u), ("v", v)):
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"{name} range must satisfy 0 <= lo < hi <= 1, got {(lo, hi)}")
        self.base = base
        self.u_range = (float(u[0]), float(u[1]))
        self.v_range = (float(v[0]), float(v[1]))

    def point_at_parameter(self, u: float, v: float) -> Point:
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        return self.base.point_at_parameter(u0 + (u1 - u0) * float(u),
                                            v0 + (v1 - v0) * float(v))


class SewnSurface(Surface):
    """Surfaces stitched along one parameter direction into a single surface.

    The pieces are traversed in order as the ``along`` parameter runs 0 -> 1;
    adjacent pieces must share their boundary curve (piece i at along=1 equals
    piece i+1 at along=0), which is checked on construction. Ideally the pieces
    literally reference the same curve (e.g. a tip cap built on
    ``wing.iso_v(eta)``), making the seam exact by construction.

    Args:
        surfaces: the pieces, in order.
        along: "v" (default) or "u" -- the direction of concatenation.
        breaks: global parameter values of the seams, length ``len(surfaces)+1``,
            from 0 to 1. Default: equal spans.
        tol: seam-matching tolerance.
    """

    def __init__(self, surfaces: list[Surface], along: str = "v",
                 breaks: list[float] | None = None, tol: float = 1e-6):
        if along not in ("u", "v"):
            raise ValueError("along must be 'u' or 'v'")
        if len(surfaces) < 2:  # noqa: PLR2004
            raise ValueError("SewnSurface needs at least two surfaces")
        self.surfaces = list(surfaces)
        self.along = along
        n = len(self.surfaces)
        self.breaks = [i / n for i in range(n + 1)] if breaks is None else [float(b) for b in breaks]
        if len(self.breaks) != n + 1 or self.breaks[0] != 0.0 or self.breaks[-1] != 1.0 \
                or any(b1 <= b0 for b0, b1 in zip(self.breaks, self.breaks[1:])):
            raise ValueError("breaks must be strictly increasing from 0 to 1, one per seam")
        self._check_seams(tol)

    def _piece_point(self, i: int, s: float, t: float) -> Point:
        """Evaluate piece i with s along the sewing direction, t across it."""
        if self.along == "v":
            return self.surfaces[i].point_at_parameter(t, s)
        return self.surfaces[i].point_at_parameter(s, t)

    def _check_seams(self, tol: float) -> None:
        for i in range(len(self.surfaces) - 1):
            for t in np.linspace(0.0, 1.0, 7):
                a = self._piece_point(i, 1.0, float(t)).as_array()
                b = self._piece_point(i + 1, 0.0, float(t)).as_array()
                if np.linalg.norm(a - b) > tol:
                    raise ConstructionError(
                        f"seam {i}: surfaces disagree by {np.linalg.norm(a - b):.3e} "
                        f"at t={t:.2f} (tol {tol:g})"
                    )

    def point_at_parameter(self, u: float, v: float) -> Point:
        s = float(v if self.along == "v" else u)
        t = float(u if self.along == "v" else v)
        i = max(0, min(len(self.surfaces) - 1,
                       int(np.searchsorted(self.breaks, s, side="right")) - 1))
        b0, b1 = self.breaks[i], self.breaks[i + 1]
        return self._piece_point(i, (s - b0) / (b1 - b0), t)


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
