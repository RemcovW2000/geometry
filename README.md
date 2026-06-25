# geometry

Standalone Python library of geometry primitives and pure-geometry meshing.
It has **no dependency on `structures`, FEM, or `composites`** — other
projects can depend on it just for geometry without pulling in structural
analysis code.

## Layout

```
src/geometry/
  primitives.py   Point, Vector, Orientation, Position, Line, Plane
  curves.py       Curve, InterpolatedLine, TrimmedCurve
  surfaces.py     GordonPatch
  utils.py        split_line_by_planes and other geometric helpers
  errors.py       Geometry exception hierarchy
  aero/           Airfoil (CST) parametrisation and wing geometry
  mesh/           Pure-geometry meshing:
                    Node      a point in a mesh
                    Element   the connectedness of a set of nodes
                    Mesh      a collection of nodes and elements
                    spacing   1-D node distribution helpers (nodes_*)
                    planar    are_points_planar
```

The `mesh` subpackage is intentionally physics-free: an `Element` only
describes which nodes are connected. Analysis layers (e.g. an FEM model in the
`structures` project) compose these geometry objects and add behaviour such as
stiffness-matrix assembly on top.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Test

```bash
pytest
```
