import dataclasses
from typing import Callable

from geometry.aero.cst import AirfoilCST


@dataclasses.dataclass
class WingShape:
    """
    Wing parametrisation class, contains information about wing geometry.

    Any callable used for distribution takes a single argument between 0 and 1,
    representing the normalized spanwise position (0 at root, 1 at tip).

    Args:
        semispan: Wing semispan (m)
        chord_distribution: Returns chord length (m) as function of eta [0, 1]
        twist_distribution: Returns twist angle (deg) as function of eta [0, 1]
        airfoil_distribution: Returns AirfoilCST as function of eta [0, 1]
    """

    chord_distribution: Callable[[float], float]
    twist_distribution: Callable[[float], float]
    airfoil_distribution: Callable[[float], AirfoilCST]
    semispan: float
