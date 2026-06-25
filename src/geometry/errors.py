class GeometryError(Exception):
    """Base exception for geometry-related errors."""

    pass


class NotPlanarError(GeometryError):
    """Exception raised when a geometry is not planar."""

    pass


class ConstructionError(GeometryError):
    """Exception raised during construction of geometric entities."""

    pass


class ToleranceError(GeometryError):
    """Exception raised during construction of geometric entities."""

    pass
