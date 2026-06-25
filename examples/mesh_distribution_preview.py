"""Preview/tune the structured wing mesh distribution (chord and span).

Plots node positions and local spacing for several distributions so you can dial
in clustering before meshing:
  - your own geometry.mesh.spacing functions (exact), e.g. dual-sided exponential
    for the chord (cluster LE & TE) and root-clustered for the span;
  - the gmsh transfinite "Progression" law (what make_structured_quads_uv emits),
    so you can pick a law/coef that matches your target.

Mapping to make_structured_quads_uv(... law, coef):
  - one-sided clustering  -> law="Progression", coef != 1
  - dual-sided clustering -> law="Bump", coef < 1   (≈ nodes_exponential_dual)

    python examples/mesh_distribution_preview.py
"""
import matplotlib.pyplot as plt
import numpy as np

from geometry.mesh.spacing import (
    nodes_cosine,
    nodes_exponential,
    nodes_exponential_dual,
    nodes_linspace,
)
from geometry.occ.meshing import transfinite_positions


def _with_ends(interior: list[float]) -> np.ndarray:
    """Your spacing helpers return interior nodes only; add the 0 and 1 ends."""
    return np.array([0.0, *interior, 1.0])


def main() -> None:
    n = 21
    distributions = {
        "uniform": _with_ends(nodes_linspace(n)),
        "cosine (LE&TE)": _with_ends(nodes_cosine(n)),
        "dual exp (LE&TE)": _with_ends(nodes_exponential_dual(n, base=6.0)),
        "exp (root)": _with_ends(nodes_exponential(n, base=6.0)),
        "gmsh Progression 1.12": transfinite_positions(n, "Progression", 1.12),
    }

    fig, (ax_pos, ax_spc) = plt.subplots(2, 1, figsize=(10, 6))
    for i, (label, pos) in enumerate(distributions.items()):
        pos = np.asarray(pos)
        ax_pos.plot(pos, np.full_like(pos, i), "o-", ms=4, label=label)
        mids = 0.5 * (pos[1:] + pos[:-1])
        ax_spc.plot(mids, np.diff(pos), "o-", ms=3, label=label)

    ax_pos.set_yticks(range(len(distributions)))
    ax_pos.set_yticklabels(list(distributions))
    ax_pos.set_xlabel("normalized position along edge")
    ax_pos.set_title("Node positions")
    ax_spc.set_xlabel("position")
    ax_spc.set_ylabel("local spacing")
    ax_spc.set_title("Spacing (smaller = finer)")
    ax_spc.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
