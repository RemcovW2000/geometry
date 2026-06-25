import enum
from pathlib import Path

from geometry import Point


class AirfoilNames(enum.Enum):
    CLARK_Y = Path(__file__).parent / "airfoils" / "clark_y.txt"


def parse_airfoil_file(
    source: Path,
    z: float = 0.0,
    scale: float = 1.0,
    header_lines: int = 1,
) -> list[Point]:
    """
    Parse an airfoil file (or an iterable of lines) and return a list of Point objects.
    - source: file path or open file object or iterable of lines.
    - z: z coordinate to assign to every point.
    - scale: multiply both x and y by this factor.
    - header_lines: number of initial non-numeric header lines to skip (default 1).
    """
    # obtain lines
    with open(str(source)) as f:
        raw_lines = f.readlines()

    points: list[Point] = []
    for line in raw_lines[header_lines:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:  # noqa: PLR2004
            continue
        try:
            x = float(parts[0]) * scale
            y = float(parts[1]) * scale
        except ValueError:
            continue
        points.append(Point(x=x, y=y, z=z))

    return points
