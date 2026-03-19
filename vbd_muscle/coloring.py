"""Graph coloring for VBD parallel vertex updates."""

import numpy as np
from collections import defaultdict


def build_vertex_adjacency(tets, n_vertices):
    """Build vertex adjacency set from tet connectivity."""
    adj = defaultdict(set)
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                adj[tet[i]].add(tet[j])
                adj[tet[j]].add(tet[i])
    return adj


def greedy_color(adj, n_vertices):
    """Greedy graph coloring (smallest-available strategy).

    Returns:
        colors: per-vertex color, shape (n_vertices,).
        n_colors: number of colors used.
        color_groups: list of arrays, color_groups[c] = vertex indices with color c.
    """
    colors = -np.ones(n_vertices, dtype=int)

    for v in range(n_vertices):
        neighbor_colors = set()
        for u in adj.get(v, []):
            if colors[u] >= 0:
                neighbor_colors.add(colors[u])
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[v] = c

    n_colors = int(colors.max()) + 1
    color_groups = [np.where(colors == c)[0] for c in range(n_colors)]
    return colors, n_colors, color_groups
