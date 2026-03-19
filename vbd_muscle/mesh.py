"""Tetrahedral mesh generation for muscle simulation."""

import numpy as np

# Kuhn 6-tet decomposition of a hex cell (all positive orientation).
# Hex vertices: 0=(000) 1=(100) 2=(110) 3=(010) 4=(001) 5=(101) 6=(111) 7=(011)
_KUHN_TETS = [
    [0, 1, 2, 6],
    [0, 1, 6, 5],
    [0, 3, 6, 2],
    [0, 3, 7, 6],
    [0, 4, 5, 6],
    [0, 4, 6, 7],
]


def generate_box_mesh(lx, ly, lz, nx, ny, nz):
    """Generate a structured tet mesh for a rectangular box.

    Uses 6-tet Kuhn triangulation per hex cell.

    Args:
        lx, ly, lz: Box dimensions.
        nx, ny, nz: Number of cells in each direction.
    Returns:
        nodes: shape (n_vertices, 3).
        tets:  shape (n_elements, 4), int.
    """
    nodes = []
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes.append([ix * lx / nx, iy * ly / ny, iz * lz / nz])
    nodes = np.array(nodes)

    def nid(ix, iy, iz):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    tets = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v = [
                    nid(ix, iy, iz),
                    nid(ix + 1, iy, iz),
                    nid(ix + 1, iy + 1, iz),
                    nid(ix, iy + 1, iz),
                    nid(ix, iy, iz + 1),
                    nid(ix + 1, iy, iz + 1),
                    nid(ix + 1, iy + 1, iz + 1),
                    nid(ix, iy + 1, iz + 1),
                ]
                for lt in _KUHN_TETS:
                    tets.append([v[lt[0]], v[lt[1]], v[lt[2]], v[lt[3]]])

    return nodes, np.array(tets, dtype=int)


def generate_cylinder_mesh(length, radius, n_axial=10, n_radial=3, n_circ=12):
    """Generate a tet mesh for a cylinder (axis along z).

    Args:
        length: Cylinder length (z direction).
        radius: Cylinder radius.
        n_axial: Number of segments along z.
        n_radial: Number of concentric rings (excluding center).
        n_circ: Number of circumferential divisions.
    Returns:
        nodes: shape (n_vertices, 3).
        tets:  shape (n_elements, 4), int.
    """
    nodes = []
    n_per_layer = 1 + n_radial * n_circ

    for iz in range(n_axial + 1):
        z = iz * length / n_axial
        nodes.append([0.0, 0.0, z])  # center
        for ir in range(1, n_radial + 1):
            r = radius * ir / n_radial
            for ic in range(n_circ):
                theta = 2.0 * np.pi * ic / n_circ
                nodes.append([r * np.cos(theta), r * np.sin(theta), z])
    nodes = np.array(nodes)

    def idx(iz, ir, ic):
        if ir == 0:
            return iz * n_per_layer
        return iz * n_per_layer + 1 + (ir - 1) * n_circ + (ic % n_circ)

    tets = []

    for iz in range(n_axial):
        # Center-to-ring1: triangular prisms -> 3 tets each
        for ic in range(n_circ):
            ic_next = (ic + 1) % n_circ
            c0, a0, b0 = idx(iz, 0, 0), idx(iz, 1, ic), idx(iz, 1, ic_next)
            c1, a1, b1 = idx(iz + 1, 0, 0), idx(iz + 1, 1, ic), idx(iz + 1, 1, ic_next)
            tets.append([c0, a0, b0, c1])
            tets.append([a0, b0, c1, a1])
            tets.append([b0, c1, a1, b1])

        # Ring-to-ring: hex cells -> 6 tets each
        for ir in range(1, n_radial):
            for ic in range(n_circ):
                ic_next = (ic + 1) % n_circ
                v = [
                    idx(iz, ir, ic),
                    idx(iz, ir + 1, ic),
                    idx(iz, ir + 1, ic_next),
                    idx(iz, ir, ic_next),
                    idx(iz + 1, ir, ic),
                    idx(iz + 1, ir + 1, ic),
                    idx(iz + 1, ir + 1, ic_next),
                    idx(iz + 1, ir, ic_next),
                ]
                for lt in _KUHN_TETS:
                    tets.append([v[lt[0]], v[lt[1]], v[lt[2]], v[lt[3]]])

    tets = np.array(tets, dtype=int)

    # Fix orientation: ensure all tets have positive volume
    valid = []
    for e in range(len(tets)):
        v0, v1, v2, v3 = tets[e]
        Dm = np.column_stack([
            nodes[v1] - nodes[v0],
            nodes[v2] - nodes[v0],
            nodes[v3] - nodes[v0],
        ])
        det = np.linalg.det(Dm)
        if abs(det) < 1e-15:
            continue  # skip degenerate tets
        if det < 0:
            tets[e, 1], tets[e, 2] = tets[e, 2], tets[e, 1]
        valid.append(e)
    tets = tets[valid]

    return nodes, tets


def assign_fiber_directions(nodes, tets, axis=None):
    """Assign uniform fiber direction d0 for each element.

    Args:
        axis: Fiber direction vector (default: z-axis [0,0,1]).
    Returns:
        fiber_dirs: shape (n_elements, 3), unit vectors.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    return np.tile(axis, (len(tets), 1))
