from itertools import pairwise

import numpy as np

from geometry import InterpolatedLine
from geometry.curves import TrimmedCurve
from geometry.primitives import Plane


def split_line_by_planes(
    line: InterpolatedLine, planes: list[Plane], n_reinterpolation_points: int = 10
) -> list[InterpolatedLine]:
    """
    Split a line by a list of planes, returning the segments of the line that lie between the planes.

    conditions:
    - Line should intersect plane
    - Line should intersect each plane only once
    - Line should not intersect plane at exact start or end
    """
    intersection_us = [0.0]
    for plane in planes:
        pt = line.plane_intersection_point(plane)[0]
        u = line.parameter_at_point(pt)
        intersection_us.append(u)
    intersection_us.append(1.0)

    trimmed_curves = []
    for u_s, u_e in pairwise(intersection_us):
        trimmed_curves.append(TrimmedCurve(base_curve=line, u_start=u_s, u_end=u_e))

    re_interpolated_curves = []
    for curve in trimmed_curves:
        pts = [curve.point_at_parameter(u) for u in np.linspace(0, 1.0, n_reinterpolation_points)]
        re_interpolated_curves.append(InterpolatedLine(pts))
    return re_interpolated_curves
