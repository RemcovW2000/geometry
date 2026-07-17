# Live Geometry Viewer — Design

An interactive tool for *programmatic CAD*: type Python that builds geometry,
see it appear in a 3D view immediately, and click any element / surface / edge /
node to see exactly where it sits in the object hierarchy — so you never have to
remember an edge index again.

Status: **design / for review** (no code written yet).

---

## 1. What we're building

Three surfaces of one tool:

1. **A live console** — you type Python (`s = GordonSurface(...)`, `m = SurfaceMesh(s).mesh()`),
   and the result renders in 3D as you run it. The console shares the same
   namespace as the viewer, so any object you name is available to inspect.
2. **A 3D viewport** — renders whatever geometry objects are currently "live".
   Auto-updates when the namespace changes.
3. **A hierarchy panel** — a tree of the live objects
   (`GordonSurface → SurfaceMesh → Mesh → Element → Node`). Picking works both
   ways: click a face in 3D and the tree highlights `Element 42`; click a node
   in the tree and it highlights in 3D.

The point is the *link* between the 3D primitive and the Python object. That
link is the actual product; the rendering is commodity.

---

## 2. The key insight: split the core from the frontend

The hardest and most valuable part is **not** the window. It's the layer that:

- turns a `geometry` object into renderable primitives, and
- keeps a stable map from *rendered primitive* → *Python object → its path in the hierarchy*.

That layer is pure Python with **no GUI dependency**. It is fully unit-testable.
Once it exists, *any* frontend (desktop, notebook, web) is a relatively thin
consumer of it. This is what makes the choice of frontend low-risk and
reversible — and it's the honest answer to "which is most scalable": **the
architecture is what scales, not the framework.**

```
                 ┌───────────────────────────────────────┐
                 │   geometry.viewer  (core, no GUI)       │
                 │                                         │
 geometry objs → │  adapters   → PolyData + id arrays      │
                 │  hierarchy  → tree model + id↔path map  │  ← tested
                 │  registry   → tracks live/named objects │
                 └───────────────────┬─────────────────────┘
                                     │  (consumes PyVista/VTK data)
              ┌──────────────────────┼───────────────────────┐
              ▼                      ▼                        ▼
      Qt frontend            Jupyter frontend          Web frontend
   (pyvistaqt +            (pyvista trame            (trame server:
    qtconsole)              jupyter backend)          Monaco + vtk.js)
```

Both PyVista/`pyvistaqt` (desktop) **and** trame (web) render the *same* VTK
`PolyData` objects the core produces. So the adapter/hierarchy layer is written
once and reused verbatim regardless of which frontend we ship. We commit to the
core now and keep the frontend swappable.

---

## 3. Frontend comparison — the honest version

You leaned toward a web app. That instinct is right *for scalability and reach*,
but it's worth being precise about the tradeoff, because "best UX" and "most
scalable" don't point at the same option.

### Desktop (pyvistaqt + Jupyter QtConsole)
- **UX:** best *tight loop*. Zero network latency, so picking feels instant and
  the console↔viewport link is snappy. A real embedded IPython REPL with
  autocomplete, sharing the viewer's namespace in-process.
- **Picking:** VTK hardware picking is the gold standard in Python — reliable
  cell/point IDs. Easiest place to get edge/face/node picking solid.
- **Cost:** heavy desktop deps (Qt). Local-only; not shareable via a link.
- **Scales for:** a serious solo/CAD workstation tool.

### Jupyter (pyvista trame jupyter backend)
- **UX:** most natural "type and see" because cells *are* the REPL. Almost no UI
  code to write.
- **Picking:** works via the trame backend, but the click→hierarchy panel is
  more awkward to build as inline widgets, and it's less of a standing "app".
- **Cost:** lowest. Good for prototyping the core and for exploratory work.
- **Scales for:** experimentation, docs, sharing notebooks — not a polished app.

### Web (trame — VTK/Kitware's own web framework)  ← recommended primary
- **UX:** a proper app in the browser: Monaco code editor + 3D viewport + tree,
  shareable by URL, no install for viewers. Runs locally too (`localhost`).
- **Why it fits here:** trame is built by Kitware (the VTK authors) precisely for
  Python-driven 3D web apps. Your geometry code stays in Python on the backend —
  **nothing is reimplemented in JS.** The backend produces the same VTK PolyData
  our core already emits.
- **Picking:** supported. Two rendering modes matter:
  - *server mode* — VTK renders server-side and streams frames; picking uses
    full VTK hardware picking (most robust, best for the id→hierarchy mapping).
  - *client/local mode* — geometry is shipped to the browser and rendered with
    vtk.js/WebGL; picking happens in-browser (snappy, but picking is less
    battle-tested in PyVista's client mode today — see Sources).
  A sane default: **server-mode picking** for correctness now, revisit local
  rendering for latency later. Both reuse the same core.
- **Cost:** most moving parts — an async server, shared reactive state, a
  WebSocket transport. This is the real price of "scalable/shareable."
- **Scales for:** multi-user, remote, embedding in a larger toolchain, sharing a
  live model with a colleague.

### Recommendation

**Build the frontend-agnostic core first (Phase 1–2), validate it in a Jupyter
notebook because that's the cheapest way to see 3D + picking working, then build
the trame web app as the primary shipped frontend (Phase 4).** Keep a desktop
(pyvistaqt) frontend as a documented option — it's a small adapter on the same
core if you ever want the lowest-latency local experience.

Rationale: the web app is the most *scalable* endpoint and matches your instinct;
Jupyter is the fastest way to de-risk the core before investing in web plumbing;
and because everything sits on one tested core, picking the "wrong" frontend
never means a rewrite.

---

## 4. Core module layout

Proposed new package `src/geometry/viewer/` (kept out of the base `geometry`
import so the core stays numpy/matplotlib-only; viewer deps are an optional
extra — see §8).

```
src/geometry/viewer/
  __init__.py
  ids.py         # SceneId: stable, hashable id for every pickable primitive
  hierarchy.py   # HierarchyNode tree + id↔node lookups (pure, tested)
  adapters.py    # geometry object -> pyvista.PolyData with id/kind cell arrays
  registry.py    # tracks named live objects; diffs on change for live update
  scene.py       # Scene: owns registry + hierarchy + assembled PolyData
  frontends/
    __init__.py
    notebook.py  # trame jupyter backend: view(scene) + click handler
    web.py       # trame server app: Monaco editor + viewport + tree
    desktop.py   # (optional) pyvistaqt + QtConsole
```

### `hierarchy.py` (the heart, no GUI, fully tested)
A `HierarchyNode` has: `id`, `kind` (`surface|mesh|element|edge|node`), a
human label, `parent`, `children`, and a back-reference to the Python object.
Provides `node_for_id(id)`, `path(id) -> ["wing_upper", "Mesh", "Element 42"]`,
and iteration for building tree UIs. This is what answers "where does this edge
sit in what I'm doing."

### `adapters.py`
One function per geometry type, each returning a `pyvista.PolyData` plus arrays:
- `cell_id` — the `SceneId` for each cell (element face / edge segment),
- `kind` — element vs edge vs node, so we can render/pick them separately,
- `point_id` — the `SceneId` for each point (node).

Rendering strategy that makes **edges** first-class (you specifically want to
click edges):
- **faces** → one PolyData of element polygons (quads/tris), `kind=element`.
- **edges** → a separate PolyData of line cells (unique node pairs from element
  connectivity), `kind=edge`. Edges are their own pickable cells, not just a
  wireframe overlay — that's what lets a click resolve to a specific edge.
- **nodes** → a point cloud PolyData, `kind=node`.

Picking returns `(kind, cell_or_point_index)` → look up the `cell_id`/`point_id`
array → `hierarchy.node_for_id(...)` → highlight + show the path. Because ids
are assigned in the adapter and carried in VTK arrays, the exact same mapping
works in Qt, Jupyter, and web.

### `registry.py` + `scene.py`
The registry holds the live namespace (names → geometry objects). On each console
execution the frontend calls `scene.sync(namespace)`; the registry diffs
(added/removed/changed), the scene rebuilds only affected PolyData and hierarchy
subtrees, and signals the frontend to re-render. That diff-based sync is what
makes the "updates live as I type/run" behavior cheap.

---

## 5. Live-update flow (web frontend)

```
type code ─► [Run]/on-change ─► exec in backend kernel (shared namespace)
     ▲                                    │
     │                                    ▼
  highlight ◄─ tree click ──┐      scene.sync(namespace)
     │                      │             │
 viewport ◄────────────────┴──── rebuild PolyData + hierarchy
     │                                    │
 click face/edge/node ─► pick ─► id ─► hierarchy path ─► highlight + tree select
```

"Live as you type" can be either explicit (a Run button / Shift-Enter) or
debounced on edit. Recommend starting with explicit-run for predictability, then
adding a debounced "auto-run" toggle.

---

## 6. Mapping onto the existing hierarchy

Today's chain is already a clean fit:

| geometry type   | hierarchy kind | pickable as              |
|-----------------|----------------|--------------------------|
| `GordonSurface` | surface        | the tessellated surface  |
| `SurfaceMesh`   | mesh (group)   | (container)              |
| `Mesh`          | mesh           | (container)              |
| `Element`       | element        | a face cell              |
| edge of Element | edge           | a line cell              |
| `Node` (`Point`)| node           | a point                  |

`Mesh` already exposes `nodes`, `elements`, `node_id_map`, `bounds()`, and
`Element` exposes `node_coordinates()` — the adapter can build PolyData directly
from these with no changes to the geometry classes. `Mesh.visualize()`
(matplotlib) stays as the lightweight, dependency-free fallback.

---

## 7. Testing strategy (fits the repo's "incremental + tested" style)

Everything valuable is testable without a GUI:
- **hierarchy**: build a small mesh, assert `path(id)`, parent/child links,
  `node_for_id` round-trips.
- **adapters**: assert a `Mesh` yields PolyData with the right cell/point counts,
  that `cell_id`/`kind` arrays have correct length and values, and that a known
  element's face carries the expected `SceneId`.
- **id round-trip**: `pick → id → hierarchy node → object` returns the same
  object we started from (the core guarantee of the whole tool).
- **registry diff**: adding/removing/renaming objects produces the expected
  add/remove/change sets.

GUI frontends get a thin smoke test (import + construct headless) and are
otherwise validated manually.

---

## 8. Dependencies

Keep base `geometry` untouched (numpy/matplotlib/scipy only). Add optional
extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
viewer     = ["pyvista>=0.44"]                       # core adapters + notebook
viewer-web = ["pyvista>=0.44", "trame", "trame-vtk",
              "trame-vuetify", "trame-code"]         # web app (Monaco editor)
viewer-qt  = ["pyvista>=0.44", "pyvistaqt", "qtconsole", "PyQt5"]  # desktop
```

`import geometry` never pulls these; `import geometry.viewer` requires `viewer`.

---

## 9. Phased roadmap (incremental, each phase shippable + tested)

1. **Core ids + hierarchy** — `ids.py`, `hierarchy.py`, tests. No rendering.
2. **Adapters** — `Mesh`/`Element`/`Node`/edges → PolyData with id arrays;
   `GordonSurface` tessellation. Tests on counts + id arrays.
3. **Notebook frontend** — `view(scene)` in Jupyter via pyvista's trame backend;
   wire click→hierarchy path printout. First time you *see* it work.
4. **Web app (primary)** — trame server: Monaco editor pane, viewport, tree
   panel; server-mode picking; explicit Run then debounced auto-run.
5. **Registry live-sync** — diff-based namespace sync driving incremental
   viewport/tree updates.
6. **Selection polish** — highlight styles, edge vs face vs node pick toggles,
   bidirectional tree↔viewport selection.
7. **(Optional) Desktop frontend** — pyvistaqt + QtConsole on the same core.

Phases 1–3 already give a usable "type in a notebook, see it, click to inspect"
loop; 4 delivers the standalone web app.

---

## 10. Open questions for you

- **Auto-run vs Run button** as the default "live" behavior (recommend Run first).
- **Surfaces**: pick at surface granularity, or also expose per-patch
  (`GordonPatch`) picking?
- **Persistence**: should a session (the code + camera) be saveable/shareable, or
  is live-only fine for v1?
- **Deploy target for the web app**: local `localhost` only, or eventually hosted
  so you can share a link?

---

### Sources
- [PyVista Trame integration (trame-pyvista)](https://github.com/pyvista/trame-pyvista)
- [Trame — PyVista Tutorial](https://tutorial.pyvista.org/tutorial/09_trame/index.html)
- [Pick point/cell/mesh with pyvista trame client mode (discussion #5132)](https://github.com/pyvista/pyvista/discussions/5132)
- [Plotter.enable_cell_picking — PyVista docs](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.enable_cell_picking)
