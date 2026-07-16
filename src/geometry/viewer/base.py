"""The viewable object model: a tree of nodes with render payloads.

Everything the browser viewer shows is a tree of :class:`ViewNode` objects,
each carrying a label, optional render geometry (triangles / polylines /
points), metadata, and children. Two ways to get into that tree:

1. Subclass :class:`Viewable` and declare child attributes with :class:`Child`::

       class WingAssembly(Viewable):
           wing = Child()
           pylon = Child()

           def __init__(self, wing, pylon):
               self.wing = wing        # validated: must itself be viewable
               self.pylon = pylon

   The base class collects declared children (in definition order) into the
   tree, and enforces at assignment time that every child is something the
   viewer knows how to display.

2. Register an adapter for an existing type with :func:`register_adapter`
   (built-in adapters cover ``Point``, curves, Gordon surfaces, wing surfaces,
   meshes and the gmsh shape API), then just hand instances to the viewer.

:func:`build_scene` turns a list of objects into the JSON document the browser
consumes.

Render payloads are plain dicts (JSON-ready):

- ``{"kind": "mesh",   "positions": [x,y,z,...], "indices": [a,b,c,...]}``
- ``{"kind": "lines",  "positions": [x,y,z,...], "segments": [a,b,...]}``
- ``{"kind": "points", "positions": [x,y,z,...]}``
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Render payload helpers
# --------------------------------------------------------------------------- #


def mesh_payload(vertices: np.ndarray, triangles: np.ndarray, color: str | None = None) -> dict:
    """Triangle-mesh payload from an (N, 3) vertex array and (M, 3) index array."""
    payload = {
        "kind": "mesh",
        "positions": np.asarray(vertices, dtype=float).reshape(-1).tolist(),
        "indices": np.asarray(triangles, dtype=int).reshape(-1).tolist(),
    }
    if color:
        payload["color"] = color
    return payload


def grid_payload(grid: np.ndarray, color: str | None = None) -> dict:
    """Triangle-mesh payload from an (n_u, n_v, 3) sampled surface grid."""
    grid = np.asarray(grid, dtype=float)
    n_u, n_v, _ = grid.shape
    vertices = grid.reshape(-1, 3)
    triangles = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i * n_v + j
            b = a + 1
            c = a + n_v
            d = c + 1
            triangles.append((a, b, c))
            triangles.append((b, d, c))
    return mesh_payload(vertices, np.array(triangles, dtype=int), color=color)


def polyline_payload(points: Sequence, color: str | None = None) -> dict:
    """Line payload through consecutive points (each with .x/.y/.z or a 3-array)."""
    coords = np.array([_xyz(p) for p in points], dtype=float)
    segments = []
    for i in range(len(coords) - 1):
        segments += [i, i + 1]
    payload = {
        "kind": "lines",
        "positions": coords.reshape(-1).tolist(),
        "segments": segments,
    }
    if color:
        payload["color"] = color
    return payload


def segments_payload(vertices: np.ndarray, segments: np.ndarray, color: str | None = None) -> dict:
    """Line payload from explicit vertices and (M, 2) segment indices."""
    payload = {
        "kind": "lines",
        "positions": np.asarray(vertices, dtype=float).reshape(-1).tolist(),
        "segments": np.asarray(segments, dtype=int).reshape(-1).tolist(),
    }
    if color:
        payload["color"] = color
    return payload


def points_payload(points: Sequence, color: str | None = None) -> dict:
    """Point-cloud payload."""
    coords = np.array([_xyz(p) for p in points], dtype=float)
    payload = {"kind": "points", "positions": coords.reshape(-1).tolist()}
    if color:
        payload["color"] = color
    return payload


def _xyz(p) -> tuple[float, float, float]:
    """Coordinates of a Point-like or 3-sequence."""
    if hasattr(p, "x"):
        return (float(p.x), float(p.y), float(p.z))
    a = np.asarray(p, dtype=float).reshape(3)
    return (float(a[0]), float(a[1]), float(a[2]))


# --------------------------------------------------------------------------- #
# The node tree
# --------------------------------------------------------------------------- #


@dataclass
class ViewNode:
    """One node of the viewer tree: label + render geometry + children."""

    label: str
    kind: str = "object"
    geoms: list[dict] = field(default_factory=list)
    children: list[ViewNode] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, node_id: str = "0") -> dict:
        """Serialize the subtree with hierarchical ids ('0', '0/2', ...)."""
        return {
            "id": node_id,
            "label": self.label,
            "kind": self.kind,
            "meta": self.meta,
            "geoms": self.geoms,
            "children": [
                child.to_dict(f"{node_id}/{i}") for i, child in enumerate(self.children)
            ],
        }


# --------------------------------------------------------------------------- #
# Adapters: turning arbitrary objects into ViewNodes
# --------------------------------------------------------------------------- #

_ADAPTERS: list[tuple[type | Callable[[Any], bool], Callable[[Any], ViewNode]]] = []
_SCENE_HOOKS: list[Callable[[], None]] = []


def register_adapter(matcher: type | Callable[[Any], bool],
                     adapter: Callable[[Any], ViewNode]) -> None:
    """Register an adapter turning matching objects into a :class:`ViewNode`.

    ``matcher`` is a type (matched with isinstance) or a predicate. Adapters
    registered later win, so applications can override the built-ins.
    """
    _ADAPTERS.append((matcher, adapter))


def register_scene_hook(hook: Callable[[], None]) -> None:
    """Register a hook run once at the start of every :func:`build_scene` call.

    Used by backends that cache per-scene work (e.g. the gmsh adapter's
    display tessellation).
    """
    _SCENE_HOOKS.append(hook)


def _find_adapter(obj: Any) -> Callable[[Any], ViewNode] | None:
    for matcher, adapter in reversed(_ADAPTERS):
        if isinstance(matcher, type):
            if isinstance(obj, matcher):
                return adapter
        elif matcher(obj):
            return adapter
    return None


def can_view(obj: Any) -> bool:
    """Whether the viewer knows how to display ``obj`` (or a list of such)."""
    if isinstance(obj, Viewable):
        return True
    if isinstance(obj, (list, tuple)):
        return all(can_view(item) for item in obj)
    return _find_adapter(obj) is not None


def as_viewable(obj: Any, label: str | None = None) -> ViewNode:
    """Turn any supported object into a :class:`ViewNode` subtree."""
    if isinstance(obj, Viewable):
        node = obj._view_node()
    elif isinstance(obj, (list, tuple)):
        node = ViewNode(label=label or "group", kind="group",
                        children=[as_viewable(item) for item in obj])
    else:
        adapter = _find_adapter(obj)
        if adapter is None:
            raise TypeError(
                f"Don't know how to display {type(obj).__name__}. Subclass Viewable "
                "or register an adapter with geometry.viewer.register_adapter()."
            )
        node = adapter(obj)
    if label:
        node.label = label
    return node


# --------------------------------------------------------------------------- #
# Viewable: the superclass with declared children
# --------------------------------------------------------------------------- #


class Child:
    """Descriptor declaring an attribute as a viewer child of a :class:`Viewable`.

    Assignment is validated: the value must itself be displayable (a
    ``Viewable``, an adapted type, or a list of those). Declared children
    appear in the viewer tree in class-definition order.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.slot = f"_child_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.slot]
        except KeyError:
            raise AttributeError(f"child {self.name!r} was never assigned") from None

    def __set__(self, instance, value) -> None:
        if not can_view(value):
            raise TypeError(
                f"{type(instance).__name__}.{self.name} must be viewable, got "
                f"{type(value).__name__}. Subclass Viewable or register an adapter."
            )
        instance.__dict__[self.slot] = value


class Viewable:
    """Base class for composite objects shown in the viewer.

    Subclasses declare their children as class-level :class:`Child` attributes
    and simply assign them in ``__init__``. Optionally override:

    - ``viz_label`` (property or attribute): the tree label (default: class name).
    - ``viz_geometry()``: render payloads drawn for this node itself.
    - ``viz_meta()``: metadata shown in the inspector panel.
    """

    @property
    def viz_label(self) -> str:
        """The tree label (default: the class name)."""
        return type(self).__name__

    def viz_geometry(self) -> list[dict]:
        """Render payloads for this node itself (default none)."""
        return []

    def viz_meta(self) -> dict[str, Any]:
        """Metadata for the inspector panel (default empty)."""
        return {}

    @classmethod
    def _declared_children(cls) -> list[str]:
        """Names of Child attributes, in MRO/class-definition order."""
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Child) and name not in names:
                    names.append(name)
        return names

    def children(self) -> list[tuple[str, Any]]:
        """(name, value) for every assigned declared child."""
        out = []
        for name in self._declared_children():
            slot = f"_child_{name}"
            if slot in self.__dict__:
                out.append((name, self.__dict__[slot]))
        return out

    def _view_node(self) -> ViewNode:
        return ViewNode(
            label=self.viz_label,
            kind=type(self).__name__,
            geoms=self.viz_geometry(),
            meta=self.viz_meta(),
            children=[as_viewable(value, label=name) for name, value in self.children()],
        )


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #


def build_scene(objects: Iterable[Any], name: str = "scene") -> dict:
    """Build the JSON scene document for a list of objects.

    This is what the viewer's rebuild runner calls with the return value of the
    user script's ``build()``.
    """
    for hook in _SCENE_HOOKS:
        hook()
    root = ViewNode(label=name, kind="scene",
                    children=[as_viewable(obj) for obj in objects])
    return {"root": root.to_dict()}


# Register the built-in adapters (import side effects keep this in one place).
from geometry.viewer import adapters as _adapters  # noqa: E402,F401

try:  # gmsh shape adapters only when the OCC backend is importable
    from geometry.viewer import gmsh_adapters as _gmsh_adapters  # noqa: E402,F401
except ImportError:  # pragma: no cover - gmsh not installed
    pass
