"""Implicit, process-global gmsh session for the shape API.

The object-oriented shape layer (:mod:`geometry.occ.shapes`) needs gmsh to be
initialized before any call, but forcing every script to open a context manager
makes simple things noisy::

    box = Solid.box(1, 1, 1)          # should just work

so the shape API initializes gmsh lazily on first use and finalizes it at
process exit. The viewer's rebuild-in-a-subprocess model fits this perfectly:
each rebuild is a fresh process, so there is never stale model state.

For explicitly scoped work (tests, batch scripts building several independent
models) the :class:`~geometry.occ.GmshSession` context manager still exists;
:func:`ensure_session` cooperates with it by never re-initializing an already
initialized gmsh.
"""
from __future__ import annotations

import atexit

from geometry.occ import _require_gmsh

_owns_session = False


def ensure_session():
    """Return the gmsh module with an initialized session, starting one if needed.

    If gmsh was already initialized (e.g. inside a ``GmshSession``), it is used
    as-is. Otherwise a process-global session is started (message console off)
    and finalized automatically at interpreter exit.
    """
    global _owns_session  # noqa: PLW0603  (deliberate module-level lifecycle flag)
    gmsh = _require_gmsh()
    if not gmsh.isInitialized():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        _owns_session = True
        atexit.register(_finalize)
    return gmsh


def _finalize() -> None:
    """Finalize the implicit session at exit (if we started it and it is alive)."""
    global _owns_session  # noqa: PLW0603  (deliberate module-level lifecycle flag)
    gmsh = _require_gmsh()
    if _owns_session and gmsh.isInitialized():
        gmsh.finalize()
    _owns_session = False


def synchronize() -> None:
    """Synchronize the OCC CAD representation with the gmsh model."""
    ensure_session().model.occ.synchronize()


def clear_model() -> None:
    """Remove every entity from the current model (fresh start, same session)."""
    gmsh = ensure_session()
    gmsh.clear()
