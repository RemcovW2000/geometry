"""OpenCASCADE-backed operations for `geometry`, via gmsh's OCC kernel.

This subpackage is **optional**. The pure-Python `geometry` package never
imports it, so installing `geometry` alone pulls in no CAD dependency. The
features here (turning surfaces into real B-spline CAD faces, surface
intersection/boolean, STEP import/export, and unstructured meshing) require
``gmsh``::

    pip install gmsh        # or: pip install geometry[occ]

gmsh embeds the OpenCASCADE (OCCT) kernel, so the heavy CAD math is genuine
OCCT — this module is a thin, geometry-friendly wrapper around it.

Typical use::

    from geometry.occ import GmshSession
    from geometry.occ.bspline import surface_to_bspline_data, add_bspline_surface

    with GmshSession() as s:
        data = surface_to_bspline_data(my_surface, us, vs)
        tag = add_bspline_surface(s.gmsh, data)
        s.occ.synchronize()
        ...
"""
from __future__ import annotations

import importlib.util


class BackendNotAvailable(ImportError):
    """Raised when an OCC-backed feature is used but gmsh is not installed."""


def available() -> bool:
    """Return True if the gmsh-backed OCC backend can be used."""
    return importlib.util.find_spec("gmsh") is not None


def _require_gmsh():
    """Import and return the gmsh module, or raise a helpful error."""
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BackendNotAvailable(
            "The geometry.occ backend requires gmsh. Install it with "
            "`pip install gmsh` (or `pip install geometry[occ]`)."
        ) from exc
    return gmsh


class GmshSession:
    """Context manager that owns gmsh's global state for one piece of work.

    gmsh is a process-global singleton (``initialize``/``finalize`` and a single
    active model), so all OCC work must be scoped. On exit the session always
    finalizes, leaving no global state behind.

    Attributes available inside the ``with`` block:
        gmsh:  the gmsh module
        model: ``gmsh.model``
        occ:   ``gmsh.model.occ``
    """

    def __init__(self, name: str = "geometry", terminal: bool = False):
        self.name = name
        self.terminal = terminal
        self.gmsh = None
        self.model = None
        self.occ = None

    def __enter__(self) -> "GmshSession":
        self.gmsh = _require_gmsh()
        self.gmsh.initialize()
        self.gmsh.option.setNumber("General.Terminal", 1 if self.terminal else 0)
        self.gmsh.model.add(self.name)
        self.model = self.gmsh.model
        self.occ = self.gmsh.model.occ
        return self

    def __exit__(self, *exc) -> None:
        if self.gmsh is not None:
            self.gmsh.finalize()
