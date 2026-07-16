"""Stable identifiers for pickable primitives in the viewer.

A :class:`SceneId` uniquely names one pickable thing (a surface, a mesh, an
element, an edge, or a node). The same id is carried on the rendered VTK
primitive and used as the key into the :class:`~geometry.viewer.hierarchy.Hierarchy`,
so a click in the 3D view resolves back to exactly one Python object and its
place in the tree. Ids are frozen and hashable so they can be dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Recognised primitive kinds, ordered from container to leaf.
KINDS = ("surface", "mesh", "group", "element", "edge", "node")


@dataclass(frozen=True)
class SceneId:
    """A stable, hashable id for one pickable primitive.

    Args:
        kind: One of :data:`KINDS`.
        key: A tuple that uniquely identifies the primitive within its kind,
            e.g. ``(element_index,)`` for an element or
            ``(element_index, edge_index)`` for an edge.
    """

    kind: str
    key: tuple

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown kind {self.kind!r}; expected one of {KINDS}.")

    def __str__(self) -> str:
        inner = ", ".join(str(k) for k in self.key)
        return f"{self.kind}({inner})"
