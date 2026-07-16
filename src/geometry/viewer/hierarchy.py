"""The object hierarchy that a picked primitive resolves into.

This module is pure Python with no rendering dependency, which keeps the
"where does this edge sit in what I'm doing" logic small and fully testable.
A :class:`Hierarchy` is a tree of :class:`HierarchyNode` objects; every node
carries a :class:`~geometry.viewer.ids.SceneId`, a human label, a back-reference
to the underlying geometry object, and free-form metadata (e.g. which elements
border an edge).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from geometry.viewer.ids import SceneId


@dataclass
class HierarchyNode:
    """One node in the viewer hierarchy tree."""

    id: SceneId
    label: str
    obj: Any = None
    parent: HierarchyNode | None = None
    children: list[HierarchyNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_child(
        self,
        id: SceneId,
        label: str,
        obj: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> HierarchyNode:
        """Create, attach and return a child node."""
        child = HierarchyNode(
            id=id,
            label=label,
            obj=obj,
            parent=self,
            metadata=metadata or {},
        )
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return (
            f"HierarchyNode({self.id}, {self.label!r}, n_children={len(self.children)})"
        )


class Hierarchy:
    """A tree of :class:`HierarchyNode` with fast id lookup.

    The tree answers the core question of the viewer: given the id of a picked
    primitive, return its node and the labelled path from the root down to it.
    """

    def __init__(self, root: HierarchyNode):
        self.root = root
        self._by_id: dict[SceneId, HierarchyNode] = {}
        for node in self._walk(root):
            self._register(node)

    def _register(self, node: HierarchyNode) -> None:
        if node.id in self._by_id:
            raise ValueError(f"Duplicate SceneId in hierarchy: {node.id}")
        self._by_id[node.id] = node

    def register_subtree(self, node: HierarchyNode) -> None:
        """Index ``node`` and all of its descendants (call after adding them)."""
        for descendant in self._walk(node):
            if descendant.id not in self._by_id:
                self._register(descendant)

    @staticmethod
    def _walk(node: HierarchyNode) -> Iterator[HierarchyNode]:
        yield node
        for child in node.children:
            yield from Hierarchy._walk(child)

    def node_for_id(self, id: SceneId) -> HierarchyNode:
        """Return the node with the given id, or raise ``KeyError``."""
        return self._by_id[id]

    def get(self, id: SceneId) -> HierarchyNode | None:
        """Return the node with the given id, or ``None`` if absent."""
        return self._by_id.get(id)

    def path(self, id: SceneId) -> list[str]:
        """Return the labels from the root down to the node for ``id``."""
        node: HierarchyNode | None = self.node_for_id(id)
        labels: list[str] = []
        while node is not None:
            labels.append(node.label)
            node = node.parent
        return list(reversed(labels))

    def path_str(self, id: SceneId, sep: str = " ▸ ") -> str:
        """Return the root-to-node path as a single joined string."""
        return sep.join(self.path(id))

    def __iter__(self) -> Iterator[HierarchyNode]:
        return self._walk(self.root)

    def __len__(self) -> int:
        return len(self._by_id)
