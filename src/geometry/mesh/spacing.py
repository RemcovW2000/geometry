"""Node distribution helpers for meshing.

Pure 1-D spacing generators used to place interior nodes along an edge/curve.
They return interior parameter values (excluding the two endpoints).
"""
import math
from typing import Optional


def nodes_linspace(nr_nodes: int, start: float = 0.0, end: float = 1.0) -> list[float]:
    """Generate a list of linearly spaced values of length nr_nodes - 2."""
    if nr_nodes < 2:  # noqa: PLR2004
        raise ValueError(f"nr_nodes must be >= 2: {nr_nodes}")

    step = 1 / (nr_nodes - 1)
    raw = [(i + 1) * step for i in range(nr_nodes - 2)]
    return [start + v * (end - start) for v in raw]


def nodes_cosine(nr_nodes: int, start: float = 0.0, end: float = 1.0) -> list[float]:
    """Generate a list of cosine spaced values of length nr_nodes - 2."""
    if nr_nodes < 2:  # noqa: PLR2004
        raise ValueError(f"nr_nodes must be >= 2: {nr_nodes}")

    raw = [0.5 * (1 - math.cos(math.pi * i / (nr_nodes - 1))) for i in range(1, nr_nodes - 1)]
    return [start + v * (end - start) for v in raw]


def nodes_exponential(
    nr_nodes: int, base: float = math.e, start: float = 0.0, end: float = 1.0
) -> list[float]:
    """Generate a list of exponentially spaced values of length nr_nodes - 2.

    Nodes are clustered near start. Higher base increases clustering strength.
    """
    if nr_nodes < 2:  # noqa: PLR2004
        raise ValueError(f"nr_nodes must be >= 2: {nr_nodes}")
    if base <= 0 or base == 1:
        raise ValueError(f"base must be > 0 and != 1: {base}")

    raw = [(base ** (i / (nr_nodes - 1)) - 1) / (base - 1) for i in range(1, nr_nodes - 1)]
    return [start + v * (end - start) for v in raw]


def nodes_exponential_dual(
    nr_nodes: int, base: float = math.e, start: float = 0.0, end: float = 1.0
) -> list[float]:
    """Generate exponentially spaced nodes clustered near both start and end.

    Higher base increases clustering strength at both ends.
    """
    if nr_nodes < 2:  # noqa: PLR2004
        raise ValueError(f"nr_nodes must be >= 2: {nr_nodes}")
    if base <= 0 or base == 1:
        raise ValueError(f"base must be > 0 and != 1: {base}")

    def _exp_map(t: float) -> float:
        return (base**t - 1) / (base - 1)

    raw = []
    for i in range(1, nr_nodes - 1):
        t = i / (nr_nodes - 1)
        v = _exp_map(2 * t) / 2 if t <= 0.5 else 1.0 - _exp_map(2 * (1 - t)) / 2  # noqa: PLR2004
        raw.append(v)

    return [start + v * (end - start) for v in raw]


def nodes_first_last_spacing(
    first: float, last: float, start: float = 0.0, end: float = 1.0
) -> list[float]:
    """
    Generate nodes with exponential spacing from first to last, summing to (end - start).

    Given spacings are approximate to given input. The spacing values are absolute:
    the second point is always at start + first, regardless of start/end.
    """
    if first <= 0 or last <= 0:
        raise ValueError("first and last must be positive")

    span = end - start

    # Quick approximation: use average spacing
    avg_spacing = (first + last) / 2
    nr_nodes = max(3, int(span / avg_spacing) + 2)

    # Refine with just a few iterations
    for _ in range(10):
        ratio = (last / first) ** (1 / (nr_nodes - 2)) if nr_nodes > 2 else 1.0  # noqa: PLR2004

        total = (
            first * (1 - ratio ** (nr_nodes - 1)) / (1 - ratio)  # noqa: PLR2004
            if abs(ratio - 1.0) > 1e-10  # noqa: PLR2004
            else first * (nr_nodes - 1)  # noqa: PLR2004
        )

        if total < 0.99 * span:  # noqa: PLR2004
            nr_nodes += 1
        elif total > 1.01 * span:  # noqa: PLR2004
            nr_nodes -= 1
        else:
            break

    # Compute nodes
    ratio = (last / first) ** (1 / (nr_nodes - 2)) if nr_nodes > 2 else 1.0  # noqa: PLR2004
    total = (
        first * (1 - ratio ** (nr_nodes - 1)) / (1 - ratio)
        if abs(ratio - 1.0) > 1e-10  # noqa: PLR2004
        else first * (nr_nodes - 1)
    )

    nodes = []
    cumulative = 0.0
    for i in range(nr_nodes - 1):
        cumulative += first * ratio**i
        if i < nr_nodes - 2:
            nodes.append(start + cumulative / total * span)

    return nodes


def nodes_exponential_from_first(
    first: float, ratio: float, reversed: bool = False, start: float = 0.0, end: float = 1.0
) -> list[float]:
    """Generate nodes with exponential spacing.

    Each spacing is ratio times the previous: first, first*ratio, first*ratio^2, ...
    Normalized to span (end - start), then offset by start.

    The first spacing is absolute: with start=0.1 and first=0.01, the second point is 0.11.

    Args:
        first: The first spacing value (absolute)
        ratio: Multiplier for each successive spacing
        reversed: If true, clustering is near end instead of start
        start: Starting position (default 0.0)
        end: Ending position (default 1.0)

    Returns:
        Interior node positions (excluding start and end)
    """
    if first <= 0:  # noqa: PLR2004
        raise ValueError("first must be positive")
    if ratio < 1.0:  # noqa: PLR2004
        raise ValueError(f"ratio must be >= 1.0: {ratio}")

    span = end - start

    spacings = []
    cumulative = 0.0
    i = 0

    while cumulative <= span:  # noqa: PLR2004
        spacing = first * (ratio**i)
        spacings.append(spacing)
        cumulative += spacing
        i += 1

    # Normalize to sum to span
    total = sum(spacings)
    normalized_spacings = [s / total * span for s in spacings]

    # Build interior node positions
    nodes = []
    cumulative = 0.0
    for i in range(len(normalized_spacings) - 1):
        cumulative += normalized_spacings[i]
        nodes.append(start + cumulative)

    if reversed:
        nodes = [start + end - val for val in nodes]

    return nodes


def nodes_spacing_start_end_refinement(  # Noqa PLR0912
    spacing: float,
    start_refinement: Optional[float],
    end_refinement: Optional[float],
    start: float = 0.0,
    end: float = 1.0,
    max_ratio: float = 1.2,
) -> list[float]:
    """
    Generate nodes with uniform spacing, but with optional refinement near start and end.

    The spacing is the target interval between nodes in the uniform region. It cannot be
    met exactly if the span is not evenly divisible, but will be as close as possible.
    Refinement zones use a geometric progression from the fine spacing up to the uniform
    spacing (ratio <= max_ratio per step).

    When refinement zones are long enough to touch, there is no uniform middle body.
    Instead, the two progressions are merged at the index where their spacing values
    are closest.

    Args:
        spacing: Target uniform spacing in the middle region.
        start_refinement: First (finest) spacing near start. Must be <= spacing. None to skip.
        end_refinement: First (finest) spacing near end. Must be <= spacing. None to skip.
        start: Start of the range (default 0.0).
        end: End of the range (default 1.0).
        max_ratio: Maximum growth ratio between consecutive spacings in a refinement zone.

    Returns:
        Interior node positions (excluding start and end).
    """
    if start_refinement is not None and start_refinement > spacing:
        raise ValueError(f"start_refinement ({start_refinement}) must be <= spacing ({spacing})")
    if end_refinement is not None and end_refinement > spacing:
        raise ValueError(f"end_refinement ({end_refinement}) must be <= spacing ({spacing})")

    span = end - start

    def _refinement_spacings(fine: float, coarse: float) -> list[float]:
        """Geometric progression from fine up to coarse, ratio <= max_ratio."""
        if fine >= coarse:
            return [fine]
        n = 2
        while (coarse / fine) ** (1.0 / (n - 1)) > max_ratio and n < 50:  # noqa: PLR2004
            n += 1
        ratio = (coarse / fine) ** (1.0 / (n - 1))
        return [fine * ratio**i for i in range(n)]

    start_spacings = (
        _refinement_spacings(start_refinement, spacing) if start_refinement is not None else []
    )
    end_spacings = (
        _refinement_spacings(end_refinement, spacing) if end_refinement is not None else []
    )

    middle_width = span - sum(start_spacings) - sum(end_spacings)

    if middle_width >= 0.0:  # noqa PLR2004
        # Zones don't touch — fill middle with uniform spacing
        n_middle = round(middle_width / spacing)
        if middle_width > 1e-10 and n_middle == 0:  # noqa: PLR2004
            n_middle = 1
        middle_spacings = [middle_width / n_middle] * n_middle if n_middle > 0 else []
        all_spacings = start_spacings + middle_spacings + list(reversed(end_spacings))

        nodes = []
        cumulative = 0.0
        for s in all_spacings[:-1]:
            cumulative += s
            nodes.append(start + cumulative)
        return nodes

    # Overlap case — find the merge point where spacing values are closest
    # Enumerate how many spacings to take from start (i) and find the maximum
    # that fit on the end side (j), then score by |start_spacings[i-1] - end_spacings[j-1]|
    best_i, best_j = 0, 0
    best_diff = float("inf")

    cum_start = 0.0
    for i in range(1, len(start_spacings) + 1):
        cum_start += start_spacings[i - 1]
        if cum_start >= span:
            break

        cum_end = 0.0
        j = 0
        for k, s in enumerate(end_spacings):
            if cum_end + s > span - cum_start:
                break
            cum_end += s
            j = k + 1

        if j == 0 and end_spacings:
            continue

        last_end = end_spacings[j - 1] if j > 0 else 0.0
        diff = abs(start_spacings[i - 1] - last_end)
        if diff < best_diff:
            best_diff = diff
            best_i, best_j = i, j

    if best_i == 0:
        # Only end refinement fits (or nothing), just take what fits from each side
        best_i = 0

    # Build interior nodes from the start side (left to right)
    start_nodes: list[float] = []
    cum = 0.0
    for k in range(best_i):
        cum += start_spacings[k]
        start_nodes.append(start + cum)

    # Build interior nodes from the end side (right to left, then sort ascending)
    end_nodes: list[float] = []
    cum = 0.0
    for k in range(best_j):
        cum += end_spacings[k]
        end_nodes.append(end - cum)
    end_nodes.reverse()

    return start_nodes + end_nodes
