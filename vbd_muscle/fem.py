"""Finite Element Method utilities for linear tetrahedral elements."""

import numpy as np


def compute_rest_data(X, tets):
    """Precompute rest-state data for all elements.

    Args:
        X: Rest positions, shape (n_vertices, 3).
        tets: Tet connectivity, shape (n_elements, 4).
    Returns:
        Dm_inv: Inverse rest shape matrices, shape (n_elements, 3, 3).
        volumes: Element rest volumes, shape (n_elements,).
    """
    n_elem = len(tets)
    Dm_inv = np.zeros((n_elem, 3, 3))
    volumes = np.zeros(n_elem)

    for e in range(n_elem):
        v0, v1, v2, v3 = tets[e]
        Dm = np.column_stack([X[v1] - X[v0], X[v2] - X[v0], X[v3] - X[v0]])
        det_Dm = np.linalg.det(Dm)
        volumes[e] = abs(det_Dm) / 6.0
        Dm_inv[e] = np.linalg.inv(Dm)

    return Dm_inv, volumes


def compute_deformation_gradient(x_elem, Dm_inv):
    """Compute deformation gradient F for one element.

    Args:
        x_elem: Current vertex positions, shape (4, 3).
        Dm_inv: Inverse rest shape matrix, shape (3, 3).
    Returns:
        F: Deformation gradient, shape (3, 3).
    """
    Ds = np.column_stack([
        x_elem[1] - x_elem[0],
        x_elem[2] - x_elem[0],
        x_elem[3] - x_elem[0],
    ])
    return Ds @ Dm_inv


def get_b_vec(Dm_inv, local_idx):
    """Gradient basis vector for vertex at local_idx in a tet element.

    For F = Ds @ B where B = Dm_inv:
      dF/dx_{k+1} has factor B[k, :] (k = 0,1,2 for local_idx 1,2,3)
      dF/dx_0 has factor -(B[0,:] + B[1,:] + B[2,:])

    Args:
        Dm_inv: shape (3, 3), rows are B[0], B[1], B[2].
        local_idx: 0, 1, 2, or 3.
    Returns:
        b: shape (3,).
    """
    if local_idx == 0:
        return -(Dm_inv[0] + Dm_inv[1] + Dm_inv[2])
    else:
        return Dm_inv[local_idx - 1].copy()


def vertex_gradient_from_pk1(P, volume, b_vec):
    """Per-vertex elastic gradient from 1st Piola-Kirchhoff stress.

    grad_i = V * P @ b_vec

    Args:
        P: 1st PK stress, shape (3, 3).
        volume: Element reference volume.
        b_vec: Gradient basis vector, shape (3,).
    Returns:
        g: shape (3,).
    """
    return volume * (P @ b_vec)


def compute_lumped_masses(tets, volumes, density, n_vertices):
    """Compute lumped vertex masses (1/4 of each incident element mass).

    Args:
        tets: shape (n_elements, 4).
        volumes: shape (n_elements,).
        density: kg/m^3.
        n_vertices: total number of vertices.
    Returns:
        masses: shape (n_vertices,).
    """
    masses = np.zeros(n_vertices)
    for e in range(len(tets)):
        m_quarter = density * volumes[e] / 4.0
        for k in range(4):
            masses[tets[e, k]] += m_quarter
    return masses
