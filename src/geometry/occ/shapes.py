"""Object-oriented topology over gmsh's OCC kernel: Solid / Face / Edge / Vertex.

gmsh's own API identifies everything by ``(dim, tag)`` integer pairs, which
makes model code read like bookkeeping. This module wraps those pairs in shape
objects with real topology navigation, in the style of ParaPy / CadQuery::

    from geometry.occ.shapes import Solid, fuse

    box = Solid.box(1.0, 1.0, 1.0)
    cyl = Solid.cylinder(Point(0.5, 0.5, 1.0), Vector(0, 0, 1), radius=0.2, height=0.5)
    part = fuse(box, cyl).named("bracket")

    part.faces                # [Face, Face, ...]
    part.faces[0].edges       # [Edge, ...]
    part.faces[0].edges[0].vertices[0].point   # geometry Point
    part.volume, part.center_of_mass
    top = part.face_nearest(Point(0.5, 0.5, 1.5))

Every constructor and boolean returns *new* shape objects and synchronizes the
model, so the result is immediately queryable. Topology accessors (``.faces``,
``.edges``, ``.vertices``) query gmsh lazily -- shapes are lightweight handles,
not copies.

Identity and invalidation: a shape is a handle to an entity in the current gmsh
model. Boolean operations *consume* their inputs (like OCC): treat every handle
you pass into ``fuse``/``cut``/``common``/``fillet`` as dead afterwards and use
only the returned shapes. Note that OCC frequently *reuses* a consumed input's
tag for the result, so a stale handle may silently alias the new entity --
``shape.exists`` can therefore only detect tags that vanished, not aliasing.
Two handles are equal iff they refer to the same ``(dim, tag)``.

The gmsh session is implicit (see :mod:`geometry.occ.session`): the first shape
operation initializes gmsh for the whole process. No context manager needed.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from geometry.occ.session import ensure_session
from geometry.primitives import Point, Position, Vector


@dataclass
class Operation:
    """The operation that produced a shape: its build history.

    Every constructor and boolean records one of these on the shape it returns
    (``shape.provenance``), so a fused airframe knows it came from a ``fuse`` of
    a fuselage, a pylon and two wings, and a lofted wing knows the section
    curves it was skinned through. The viewer renders this chain as a
    "history" subtree.

    Attributes:
        name: the operation ("box", "loft", "fuse", "import_step", ...).
        inputs: what went in -- consumed :class:`Shape` handles (structure and
            their own provenance remain inspectable; their geometry is gone)
            and/or raw geometry (e.g. loft section point-loops, which the
            viewer can still draw).
        params: JSON-friendly parameters (dimensions, radii, file paths, ...).
    """

    name: str
    inputs: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _xyz_tuple(p) -> tuple[float, float, float]:
    """A Point/Vector as a plain tuple (JSON-friendly for provenance params)."""
    return (float(p.x), float(p.y), float(p.z))


class Shape:
    """A handle to one entity in the gmsh OCC model, with topology navigation.

    Attributes:
        tag: the gmsh entity tag.
        name: a human label (used by the viewer tree); settable, chainable
            via :meth:`named`.
        provenance: the :class:`Operation` that produced this shape (build
            history), or None for bare wrapped entities.
    """

    dim: int = -1

    def __init__(self, tag: int, name: str | None = None,
                 provenance: Operation | None = None):
        ensure_session()
        self.tag = int(tag)
        self.name = name or f"{type(self).__name__.lower()}{self.tag}"
        self.provenance = provenance

    # -- identity ------------------------------------------------------------- #

    @property
    def dimtag(self) -> tuple[int, int]:
        """The gmsh ``(dim, tag)`` pair this shape wraps."""
        return (self.dim, self.tag)

    def named(self, name: str) -> Shape:
        """Set the shape's name and return the shape (chainable)."""
        self.name = name
        return self

    @property
    def exists(self) -> bool:
        """Whether the entity is still present in the model (booleans consume inputs)."""
        gmsh = ensure_session()
        return (self.dim, self.tag) in gmsh.model.getEntities(self.dim)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shape) and other.dimtag == self.dimtag

    def __hash__(self) -> int:
        return hash(self.dimtag)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tag={self.tag}, name={self.name!r})"

    # -- topology --------------------------------------------------------------- #

    def boundary(self, recursive: bool = False) -> list[Shape]:
        """The entities bounding this one, one dimension down.

        With ``recursive=True`` the boundary is taken all the way to vertices.
        """
        gmsh = ensure_session()
        dimtags = gmsh.model.getBoundary([self.dimtag], combined=False,
                                         oriented=False, recursive=recursive)
        seen: dict[tuple[int, int], Shape] = {}
        for dim, tag in dimtags:
            seen.setdefault((dim, abs(tag)), _wrap(dim, abs(tag)))
        return list(seen.values())

    def _boundary_of_dim(self, dim: int) -> list[Shape]:
        """Unique boundary entities of a specific dimension."""
        if dim == self.dim - 1:
            return self.boundary()
        shapes: dict[tuple[int, int], Shape] = {}
        for sub in self.boundary():
            for subsub in sub._boundary_of_dim(dim):
                shapes.setdefault(subsub.dimtag, subsub)
        return list(shapes.values())

    # -- geometry --------------------------------------------------------------- #

    @property
    def center_of_mass(self) -> Point:
        """Center of mass (of the length/area/volume measure of this entity)."""
        gmsh = ensure_session()
        x, y, z = gmsh.model.occ.getCenterOfMass(self.dim, self.tag)
        return Point(x, y, z)

    @property
    def bounding_box(self) -> tuple[Point, Point]:
        """Axis-aligned bounding box as ``(min_corner, max_corner)``."""
        gmsh = ensure_session()
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(self.dim, self.tag)
        return Point(x0, y0, z0), Point(x1, y1, z1)

    def distance_to(self, point: Point) -> float:
        """Shortest distance from ``point`` to this (trimmed) entity.

        gmsh's ``getClosestPoint`` projects onto the *underlying* geometry (an
        infinite plane, a full cylinder, ...), so the projection can land
        outside the trimmed face/edge. When that happens (checked with
        ``isInside``), the distance is taken to the entity's boundary instead,
        which keeps selection helpers like :meth:`Solid.face_nearest` honest.
        """
        gmsh = ensure_session()
        coord = [float(point.x), float(point.y), float(point.z)]
        if self.dim == 0:
            x, y, z = gmsh.model.getValue(0, self.tag, [])
            return float(np.linalg.norm(np.array([x, y, z]) - coord))
        if self.dim in (1, 2):
            closest, _ = gmsh.model.getClosestPoint(self.dim, self.tag, coord)
            distance = float(np.linalg.norm(np.asarray(closest) - coord))
            if gmsh.model.isInside(self.dim, self.tag, list(closest)):
                return distance
            boundary = self.boundary()
            if not boundary:  # closed periodic entity: the projection lies on it
                return distance
            return min(sub.distance_to(point) for sub in boundary)
        # Solids: closest point over the bounding faces.
        return min(face.distance_to(point) for face in self._boundary_of_dim(2))

    # -- transforms (in place; return self for chaining) ------------------------- #

    def translate(self, vector: Vector) -> Shape:
        """Translate the shape by ``vector`` (in place)."""
        gmsh = ensure_session()
        gmsh.model.occ.translate([self.dimtag], float(vector.x), float(vector.y), float(vector.z))
        gmsh.model.occ.synchronize()
        return self

    def rotate(self, point: Point, axis: Vector, angle: float) -> Shape:
        """Rotate the shape ``angle`` radians about the axis through ``point`` (in place)."""
        gmsh = ensure_session()
        gmsh.model.occ.rotate([self.dimtag], float(point.x), float(point.y), float(point.z),
                              float(axis.x), float(axis.y), float(axis.z), float(angle))
        gmsh.model.occ.synchronize()
        return self

    def mirror(self, a: float, b: float, c: float, d: float = 0.0) -> Shape:
        """Reflect the shape about the plane ``a x + b y + c z + d = 0`` (in place)."""
        gmsh = ensure_session()
        gmsh.model.occ.mirror([self.dimtag], float(a), float(b), float(c), float(d))
        gmsh.model.occ.synchronize()
        return self

    def scale(self, factor: float, center: Point | None = None) -> Shape:
        """Uniformly scale the shape about ``center`` (default origin; in place)."""
        gmsh = ensure_session()
        c = center or Point(0.0, 0.0, 0.0)
        gmsh.model.occ.dilate([self.dimtag], float(c.x), float(c.y), float(c.z),
                              float(factor), float(factor), float(factor))
        gmsh.model.occ.synchronize()
        return self

    def place(self, position: Position) -> Shape:
        """Rigidly place the shape with a geometry ``Position`` (in place).

        Applies ``global = R @ local + origin``, matching ``Position.point_in_global``.
        """
        gmsh = ensure_session()
        r = position.orientation.as_matrix()
        o = position.origin.as_array()
        affine = [
            r[0, 0], r[0, 1], r[0, 2], o[0],
            r[1, 0], r[1, 1], r[1, 2], o[1],
            r[2, 0], r[2, 1], r[2, 2], o[2],
        ]
        gmsh.model.occ.affineTransform([self.dimtag], [float(v) for v in affine])
        gmsh.model.occ.synchronize()
        return self

    def copy(self) -> Shape:
        """Return an independent copy of this shape."""
        gmsh = ensure_session()
        (dim, tag), = gmsh.model.occ.copy([self.dimtag])
        gmsh.model.occ.synchronize()
        result = _wrap(dim, tag, name=f"{self.name}_copy")
        result.provenance = Operation("copy", inputs=[self])
        return result

    def mirrored(self, a: float, b: float, c: float, d: float = 0.0) -> Shape:
        """Return a mirrored copy (the original is untouched)."""
        result = self.copy().mirror(a, b, c, d)
        result.name = f"{self.name}_mirror"
        result.provenance = Operation("mirror", inputs=[self],
                                      params={"plane": (a, b, c, d)})
        return result

    def remove(self, recursive: bool = True) -> None:
        """Delete the entity from the model."""
        gmsh = ensure_session()
        gmsh.model.occ.remove([self.dimtag], recursive=recursive)
        gmsh.model.occ.synchronize()

    # -- boolean sugar ------------------------------------------------------------ #

    def __add__(self, other: Shape) -> Shape:
        """``a + b``: boolean union (consumes both handles)."""
        return fuse(self, other)

    def __sub__(self, other: Shape) -> Shape:
        """``a - b``: boolean cut (consumes both handles)."""
        return cut(self, other)

    def __and__(self, other: Shape) -> Shape:
        """``a & b``: boolean intersection (consumes both handles)."""
        return common(self, other)


class Vertex(Shape):
    """A topological vertex (gmsh dim 0)."""

    dim = 0

    @property
    def point(self) -> Point:
        """The vertex location as a geometry ``Point``."""
        gmsh = ensure_session()
        x, y, z = gmsh.model.getValue(0, self.tag, [])
        return Point(x, y, z)


class Edge(Shape):
    """A topological edge / curve (gmsh dim 1)."""

    dim = 1

    @property
    def vertices(self) -> list[Vertex]:
        """The end vertices of the edge."""
        return [v for v in self.boundary() if isinstance(v, Vertex)]

    @property
    def length(self) -> float:
        """Arc length of the edge."""
        gmsh = ensure_session()
        return float(gmsh.model.occ.getMass(1, self.tag))

    @property
    def kind(self) -> str:
        """The underlying curve type (e.g. ``'Line'``, ``'Circle'``, ``'BSpline'``)."""
        gmsh = ensure_session()
        return gmsh.model.getType(1, self.tag)

    def sample(self, n: int = 50) -> list[Point]:
        """Sample ``n`` points along the edge (uniform in parameter)."""
        gmsh = ensure_session()
        (t0,), (t1,) = gmsh.model.getParametrizationBounds(1, self.tag)
        params = np.linspace(t0, t1, n)
        coords = gmsh.model.getValue(1, self.tag, list(params))
        pts = np.asarray(coords, dtype=float).reshape(-1, 3)
        return [Point(*p) for p in pts]


class Face(Shape):
    """A topological face / surface (gmsh dim 2)."""

    dim = 2

    @property
    def edges(self) -> list[Edge]:
        """The edges bounding the face."""
        return [e for e in self.boundary() if isinstance(e, Edge)]

    @property
    def vertices(self) -> list[Vertex]:
        """The unique vertices of the face."""
        return [v for v in self._boundary_of_dim(0) if isinstance(v, Vertex)]

    @property
    def area(self) -> float:
        """Surface area of the face."""
        gmsh = ensure_session()
        return float(gmsh.model.occ.getMass(2, self.tag))

    @property
    def kind(self) -> str:
        """The underlying surface type (e.g. ``'Plane'``, ``'Cylinder'``, ``'BSpline'``)."""
        gmsh = ensure_session()
        return gmsh.model.getType(2, self.tag)

    @property
    def normal_at_center(self) -> Vector:
        """The surface normal at the parametric center of the face."""
        gmsh = ensure_session()
        (u0, v0), (u1, v1) = gmsh.model.getParametrizationBounds(2, self.tag)
        nx, ny, nz = gmsh.model.getNormal(self.tag, [(u0 + u1) / 2, (v0 + v1) / 2])
        return Vector(nx, ny, nz)


class Solid(Shape):
    """A topological solid / volume (gmsh dim 3)."""

    dim = 3

    @property
    def faces(self) -> list[Face]:
        """The faces bounding the solid."""
        return [f for f in self.boundary() if isinstance(f, Face)]

    @property
    def edges(self) -> list[Edge]:
        """The unique edges of the solid."""
        return [e for e in self._boundary_of_dim(1) if isinstance(e, Edge)]

    @property
    def vertices(self) -> list[Vertex]:
        """The unique vertices of the solid."""
        return [v for v in self._boundary_of_dim(0) if isinstance(v, Vertex)]

    @property
    def volume(self) -> float:
        """Enclosed volume."""
        gmsh = ensure_session()
        return float(gmsh.model.occ.getMass(3, self.tag))

    def face_nearest(self, point: Point) -> Face:
        """The bounding face closest to ``point`` (handy for BCs and loads)."""
        return min(self.faces, key=lambda f: f.distance_to(point))

    def edge_nearest(self, point: Point) -> Edge:
        """The edge closest to ``point``."""
        return min(self.edges, key=lambda e: e.distance_to(point))

    def faces_where(self, predicate: Callable[[Face], bool]) -> list[Face]:
        """The bounding faces for which ``predicate(face)`` is true."""
        return [f for f in self.faces if predicate(f)]

    # -- primitive constructors ----------------------------------------------- #

    @classmethod
    def box(cls, dx: float, dy: float, dz: float, corner: Point | None = None,
            name: str | None = None) -> Solid:
        """An axis-aligned box with side lengths (dx, dy, dz) from ``corner``."""
        gmsh = ensure_session()
        c = corner or Point(0.0, 0.0, 0.0)
        tag = gmsh.model.occ.addBox(float(c.x), float(c.y), float(c.z),
                                    float(dx), float(dy), float(dz))
        gmsh.model.occ.synchronize()
        provenance = Operation("box", params={"dx": dx, "dy": dy, "dz": dz,
                                              "corner": _xyz_tuple(c)})
        return cls(tag, name=name or "box", provenance=provenance)

    @classmethod
    def cylinder(cls, base: Point, axis: Vector, radius: float, height: float,
                 name: str | None = None) -> Solid:
        """A cylinder from ``base``, extruded ``height`` along ``axis``."""
        gmsh = ensure_session()
        a = np.array([float(axis.x), float(axis.y), float(axis.z)])
        norm = np.linalg.norm(a)
        if norm == 0:
            raise ValueError("cylinder axis must be non-zero")
        d = a / norm * float(height)
        tag = gmsh.model.occ.addCylinder(float(base.x), float(base.y), float(base.z),
                                         d[0], d[1], d[2], float(radius))
        gmsh.model.occ.synchronize()
        provenance = Operation("cylinder", params={
            "base": _xyz_tuple(base), "axis": _xyz_tuple(axis),
            "radius": radius, "height": height,
        })
        return cls(tag, name=name or "cylinder", provenance=provenance)

    @classmethod
    def sphere(cls, center: Point, radius: float, name: str | None = None) -> Solid:
        """A sphere."""
        gmsh = ensure_session()
        tag = gmsh.model.occ.addSphere(float(center.x), float(center.y), float(center.z),
                                       float(radius))
        gmsh.model.occ.synchronize()
        provenance = Operation("sphere", params={"center": _xyz_tuple(center), "radius": radius})
        return cls(tag, name=name or "sphere", provenance=provenance)

    @classmethod
    def loft(cls, sections: Sequence[Sequence[Point]], ruled: bool = False,
             name: str | None = None) -> Solid:
        """Loft a capped solid through closed point-loop sections.

        Each section is an ordered loop of points (do NOT repeat the first
        point); a closed interpolating spline wire is built through each loop
        and the wires are skinned with capped ends (``addThruSections``).
        """
        gmsh = ensure_session()
        occ = gmsh.model.occ
        sections = [list(loop) for loop in sections]
        wires = []
        for loop in sections:
            pt_tags = [occ.addPoint(float(p.x), float(p.y), float(p.z)) for p in loop]
            curve = occ.addSpline(pt_tags + [pt_tags[0]])
            wires.append(occ.addWire([curve]))
        out = occ.addThruSections(wires, makeSolid=True, makeRuled=ruled)
        occ.synchronize()
        vols = [tag for (dim, tag) in out if dim == 3]  # noqa: PLR2004
        if not vols:
            raise RuntimeError(f"loft did not produce a solid: {out}")
        provenance = Operation("loft", inputs=sections,
                               params={"ruled": ruled, "n_sections": len(sections)})
        return cls(vols[0], name=name or "loft", provenance=provenance)


_SHAPE_TYPES: dict[int, type[Shape]] = {0: Vertex, 1: Edge, 2: Face, 3: Solid}


def _wrap(dim: int, tag: int, name: str | None = None) -> Shape:
    """Wrap a gmsh ``(dim, tag)`` in the matching shape class."""
    return _SHAPE_TYPES[dim](tag, name=name)


def _wrap_all(dimtags: Iterable[tuple[int, int]], name: str | None = None,
              provenance: Operation | None = None) -> list[Shape]:
    """Wrap a list of dimtags, numbering names when a base name is given."""
    shapes = [_wrap(dim, tag) for dim, tag in dimtags]
    for i, shape in enumerate(shapes):
        if name:
            shape.name = name if len(shapes) == 1 else f"{name}{i}"
        shape.provenance = provenance
    return shapes


# ------------------------------------------------------------------------------ #
# Booleans and CAD operations (module-level; all synchronize and return new shapes)
# ------------------------------------------------------------------------------ #

def _top(dimtags: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The highest-dimension entities of a boolean result."""
    top_dim = max(dim for dim, _ in dimtags)
    return [(dim, tag) for dim, tag in dimtags if dim == top_dim]


def _single_or_list(shapes: list[Shape]) -> Shape | list[Shape]:
    return shapes[0] if len(shapes) == 1 else shapes


def fuse(*shapes: Shape, name: str | None = None) -> Shape | list[Shape]:
    """Boolean union. Consumes the input handles; returns the fused shape.

    Interior faces are removed, so the result is one clean solid (or several,
    if the inputs were disjoint -- then a list is returned).
    """
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("fuse needs at least two shapes")
    gmsh = ensure_session()
    out, _ = gmsh.model.occ.fuse([shapes[0].dimtag], [s.dimtag for s in shapes[1:]])
    gmsh.model.occ.synchronize()
    provenance = Operation("fuse", inputs=list(shapes))
    return _single_or_list(_wrap_all(_top(out), name=name or "fused", provenance=provenance))


def cut(target: Shape, tool: Shape | Sequence[Shape], keep_tool: bool = False,
        name: str | None = None) -> Shape | list[Shape]:
    """Boolean difference ``target - tool``. Consumes handles (tool kept if asked)."""
    gmsh = ensure_session()
    tools = [tool] if isinstance(tool, Shape) else list(tool)
    out, _ = gmsh.model.occ.cut([target.dimtag], [t.dimtag for t in tools],
                                removeTool=not keep_tool)
    gmsh.model.occ.synchronize()
    provenance = Operation("cut", inputs=[target, *tools], params={"keep_tool": keep_tool})
    return _single_or_list(_wrap_all(_top(out), name=name or f"{target.name}_cut",
                                     provenance=provenance))


def common(a: Shape, b: Shape, name: str | None = None) -> Shape | list[Shape]:
    """Boolean intersection of two shapes. Consumes both handles."""
    gmsh = ensure_session()
    out, _ = gmsh.model.occ.intersect([a.dimtag], [b.dimtag])
    gmsh.model.occ.synchronize()
    provenance = Operation("common", inputs=[a, b])
    return _single_or_list(_wrap_all(_top(out), name=name or "common", provenance=provenance))


def fragment(*shapes: Shape, name: str | None = None) -> list[Shape]:
    """Fragment shapes into a conformal model: intersections become shared topology.

    Unlike :func:`fuse` the parts stay separate volumes, but they share the
    faces/edges where they touch -- exactly what a conformal FEM mesh across an
    interface needs. Consumes the input handles.
    """
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("fragment needs at least two shapes")
    gmsh = ensure_session()
    out, _ = gmsh.model.occ.fragment([shapes[0].dimtag], [s.dimtag for s in shapes[1:]])
    gmsh.model.occ.synchronize()
    provenance = Operation("fragment", inputs=list(shapes))
    return _wrap_all(_top(out), name=name or "fragment", provenance=provenance)


def fillet(solid: Solid, edges: Sequence[Edge], radius: float,
           name: str | None = None) -> Solid:
    """Fillet the given edges of a solid with a constant radius.

    Consumes the solid handle and returns the filleted solid. OCC fillets on
    free-form intersections can fail for large radii; start small.
    """
    gmsh = ensure_session()
    out = gmsh.model.occ.fillet([solid.tag], [e.tag for e in edges], [float(radius)])
    gmsh.model.occ.synchronize()
    vols = [tag for (dim, tag) in out if dim == 3]  # noqa: PLR2004
    if not vols:
        raise RuntimeError(f"fillet produced no solid: {out}")
    provenance = Operation("fillet", inputs=[solid],
                           params={"radius": radius, "n_edges": len(list(edges))})
    return Solid(vols[0], name=name or f"{solid.name}_fillet", provenance=provenance)


def import_step(path: str, name: str | None = None) -> list[Shape]:
    """Import a STEP/BREP/IGES file; returns the top-level shapes (usually solids)."""
    gmsh = ensure_session()
    dimtags = gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    provenance = Operation("import_step", params={"path": str(path)})
    return _wrap_all(dimtags, name=name, provenance=provenance)


def export_step(path: str) -> None:
    """Write the whole current model to a STEP/BREP file (by extension)."""
    gmsh = ensure_session()
    gmsh.write(str(path))


def bspline_face(surface, n_u: int = 25, n_v: int = 25, name: str | None = None) -> Face:
    """Turn a parametric surface (``point_at_parameter(u, v)``) into a real CAD face.

    The surface is sampled on an (n_u, n_v) grid and approximated by an OCC
    B-spline surface (see :mod:`geometry.occ.bspline`).
    """
    from geometry.occ.bspline import (  # noqa: PLC0415  (optional heavy import)
        add_bspline_surface,
        surface_to_bspline_data,
    )

    gmsh = ensure_session()
    us = np.linspace(0.0, 1.0, n_u)
    vs = np.linspace(0.0, 1.0, n_v)
    data = surface_to_bspline_data(surface, us, vs)
    tag = add_bspline_surface(gmsh, data)
    gmsh.model.occ.synchronize()
    provenance = Operation("bspline_face", inputs=[surface],
                           params={"n_u": n_u, "n_v": n_v})
    return Face(tag, name=name or "bspline_face", provenance=provenance)


def all_shapes(dim: int | None = None) -> list[Shape]:
    """Every entity currently in the model (optionally of one dimension)."""
    gmsh = ensure_session()
    return [_wrap(d, t) for d, t in gmsh.model.getEntities(-1 if dim is None else dim)]


def nearest(shapes: Sequence[Shape], point: Point) -> Shape:
    """The shape from ``shapes`` closest to ``point``."""
    return min(shapes, key=lambda s: s.distance_to(point))
