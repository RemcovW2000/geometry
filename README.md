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

## CAD topology API (`geometry.occ.shapes`)

An object-oriented layer over gmsh's OCC kernel — no more `(dim, tag)`
bookkeeping. Operations return shape objects; topology is navigable:

```python
from geometry import Point, Vector
from geometry.occ.shapes import Solid, fuse, cut, fillet, import_step

fuselage = import_step("data/simplified_fuselage_solid.step")[0]
pylon = Solid.cylinder(Point(0, 0, 1), Vector(0, 0, 1), radius=0.2, height=1.0)
airframe = fuse(fuselage, pylon, name="airframe")     # or: fuselage + pylon

airframe.faces                       # [Face, ...]
airframe.faces[0].edges[0].vertices[0].point
airframe.volume, airframe.center_of_mass, airframe.bounding_box
top = airframe.face_nearest(Point(0, 0, 5))           # selection for BCs later
```

The gmsh session is implicit (starts on first use, one per process); booleans
consume their inputs like OCC does — `shape.exists` tells you if a handle is
still alive. `fragment(...)` keeps parts separate but conformal (shared
interface topology), which is what a conformal FEM mesh needs.

## Browser viewer (`geometry.viewer`)

Write geometry code in your IDE, see it live in the browser. The API is
explicit — what you pass to the `Viewer` is what is shown:

```python
from geometry.viewer import Viewer

airframe = ...   # shapes, surfaces, curves, meshes, Viewable objects
mesh = ...
Viewer([airframe, mesh]).show()     # python my_script.py
```

Save the file (or hit **Rebuild**) and the page updates: the script is re-run
in a fresh subprocess whose `show()` call feeds the new scene back instead of
serving (so crashes just show a traceback in the page). The CLI form for
`build() -> list` scripts also works:
`python -m geometry.viewer examples/viewer_demo.py --watch`.

The tree panel mirrors the object hierarchy (a solid's faces, a face's edges,
a Gordon surface's profile curves, a curve's points, ...); clicking geometry
selects the tree node and shows its metadata (areas, lengths, tags). Solids
also carry their **build history**: a fused airframe shows the fuselage, pylon
and wings that went into the `fuse`; a lofted wing shows its input section
curves (drawn, hidden by default) — the full provenance chain, recursively.

Built-in adapters cover `Point`, curves, `GordonPatch`/`GordonSurface`,
`WingSurface`, `Mesh`, and the gmsh shape API. Your own composite objects
subclass `Viewable` and declare children:

```python
from geometry.viewer import Viewable, Child

class WingAssembly(Viewable):
    wing = Child()      # assignment is validated: children must be viewable
    pylon = Child()
```

See `examples/viewer_demo.py` and `examples/wing_on_fuselage.py` (STEP fuselage
+ parametric wing + cylinder pylon fused into one solid).

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
