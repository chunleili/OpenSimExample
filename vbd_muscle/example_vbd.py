"""
This is based on NVIDIA Newton's VBD. I deleted some irrelevant part and only kept the soft body simulation part. This single script can be runned by it self, making it easy to copy to other places. The example constracut a soft beam with left side fixed, and it will bend down under gravity. The output is located at "output/ply" folder, which are ascii ply sequence of 300 frames.
"""


import warp as wp
import numpy as np
import os
import time


class ParticleFlags:
    """Flags for particle properties."""
    ACTIVE = 1 << 0


VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}

class Model:
    """Simulation model: mesh topology, material parameters, and scratch buffers.

    Attributes are populated by ModelBuilder.finalize().
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.particle_count = 0
        self.requires_grad = False

    def state(self) -> "State":
        return State(self)


class State:
    """Per-timestep mutable state: positions, velocities, external forces."""

    def __init__(self, model: Model):
        n = model.particle_count
        device = model.device
        self.particle_q = wp.clone(model.particle_q) if n > 0 else wp.zeros(1, dtype=wp.vec3, device=device)
        self.particle_qd = wp.clone(model.particle_qd) if n > 0 else wp.zeros(1, dtype=wp.vec3, device=device)
        self.particle_f = wp.zeros(max(n, 1), dtype=wp.vec3, device=device)


class Contacts:
    pass


class Control:
    pass

class vec9(wp.types.vector(length=9, dtype=wp.float32)):
    pass

class mat99(wp.types.matrix(shape=(9, 9), dtype=wp.float32)):
    pass

@wp.struct
class ParticleForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    v_adj_tets: wp.array(dtype=int)
    v_adj_tets_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ParticleForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            adjacency_gpu.v_adj_tets = self.v_adj_tets.to(device)
            adjacency_gpu.v_adj_tets_offsets = self.v_adj_tets_offsets.to(device)

            return adjacency_gpu
        



@wp.func
def get_vertex_num_adjacent_edges(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.func
def get_vertex_num_adjacent_tets(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_tets_offsets[vertex + 1] - adjacency.v_adj_tets_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_tet_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, tet: wp.int32):
    offset = adjacency.v_adj_tets_offsets[vertex]
    return adjacency.v_adj_tets[offset + tet * 2], adjacency.v_adj_tets[offset + tet * 2 + 1]

@wp.func
def compute_cofactor(F: wp.mat33) -> wp.mat33:
    """Compute the cofactor (adjugate) matrix directly without using inverse.

    This is numerically stable even when det(F) ≈ 0, unlike J * transpose(inverse(F)).
    """
    F11, F21, F31 = F[0, 0], F[1, 0], F[2, 0]
    F12, F22, F32 = F[0, 1], F[1, 1], F[2, 1]
    F13, F23, F33 = F[0, 2], F[1, 2], F[2, 2]

    return wp.mat33(
        F22 * F33 - F23 * F32,
        F23 * F31 - F21 * F33,
        F21 * F32 - F22 * F31,
        F13 * F32 - F12 * F33,
        F11 * F33 - F13 * F31,
        F12 * F31 - F11 * F32,
        F12 * F23 - F13 * F22,
        F13 * F21 - F11 * F23,
        F11 * F22 - F12 * F21,
    )

@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # StVK energy density: psi = mu * ||G||_F^2 + 0.5 * lambda * (trace(G))^2

    # Deformation gradient F = [f0, f1] (3x2 matrix as two 3D column vectors)
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    # Cache tri_pose elements
    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Compute F columns directly: F = [x01, x02] * tri_pose = [f0, f1]
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Green strain tensor: G = 0.5(F^T F - I) = [[G00, G01], [G01, G11]] (symmetric 2x2)
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0_dot_f0 - 1.0)
    G11 = 0.5 * (f1_dot_f1 - 1.0)
    G01 = 0.5 * f0_dot_f1

    # Frobenius norm squared of Green strain: ||G||_F^2 = G00^2 + G11^2 + 2 * G01^2
    G_frobenius_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_frobenius_sq < 1.0e-20:
        return wp.vec3(0.0), wp.mat33(0.0)

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress tensor (StVK model)
    # PK1 = 2*mu*F*G + lambda*trace(G)*F = [PK1_col0, PK1_col1] (3x2)
    lambda_trace_G = lmbd * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lambda_trace_G) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lambda_trace_G)

    # Vertex selection using masks to avoid branching
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    # Deformation gradient derivatives w.r.t. current vertex position
    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force via chain rule: force = -(dpsi/dF) : (dF/dx)
    dpsi_dx = PK1_col0 * df0_dx + PK1_col1 * df1_dx
    force = -dpsi_dx

    # Hessian computation using Cauchy-Green invariants
    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx
    df0_df1_cross = df0_dx * df1_dx

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd
    I33 = wp.identity(n=3, dtype=float)

    f0_outer_f0 = wp.outer(f0, f0)
    f1_outer_f1 = wp.outer(f1, f1)
    f0_outer_f1 = wp.outer(f0, f1)
    f1_outer_f0 = wp.outer(f1, f0)

    H_IIc00_scaled = mu * (f0_dot_f0 * I33 + 2.0 * f0_outer_f0 + f1_outer_f1)
    H_IIc11_scaled = mu * (f1_dot_f1 * I33 + 2.0 * f1_outer_f1 + f0_outer_f0)
    H_IIc01_scaled = mu * (f0_dot_f1 * I33 + f1_outer_f0)

    # d2(psi)/dF^2 components
    d2E_dF2_00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I33 + H_IIc00_scaled
    d2E_dF2_01 = lmbd * f0_outer_f1 + H_IIc01_scaled
    d2E_dF2_11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I33 + H_IIc11_scaled

    # Chain rule: H = (dF/dx)^T * (d2(psi)/dF^2) * (dF/dx)
    hessian = df0_dx_sq * d2E_dF2_00 + df1_dx_sq * d2E_dF2_11 + df0_df1_cross * (d2E_dF2_01 + wp.transpose(d2E_dF2_01))

    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Previous deformation gradient for velocity
        x0_prev = pos_anchor[v0]
        x01_prev = pos_anchor[v1] - x0_prev
        x02_prev = pos_anchor[v2] - x0_prev

        vel_x01 = (x01 - x01_prev) * inv_dt
        vel_x02 = (x02 - x02_prev) * inv_dt

        df0_dt = vel_x01 * DmInv00 + vel_x02 * DmInv10
        df1_dt = vel_x01 * DmInv01 + vel_x02 * DmInv11

        # First constraint: Cmu = ||G||_F (Frobenius norm of Green strain)
        Cmu = wp.sqrt(G_frobenius_sq)

        G00_normalized = G00 / Cmu
        G01_normalized = G01 / Cmu
        G11_normalized = G11 / Cmu

        # Time derivative of Green strain: dG/dt = 0.5 * (F^T * dF/dt + (dF/dt)^T * F)
        dG_dt_00 = wp.dot(f0, df0_dt)  # dG00/dt
        dG_dt_11 = wp.dot(f1, df1_dt)  # dG11/dt
        dG_dt_01 = 0.5 * (wp.dot(f0, df1_dt) + wp.dot(f1, df0_dt))  # dG01/dt

        # Time derivative of first constraint: dCmu/dt = (1/||G||_F) * (G : dG/dt)
        dCmu_dt = G00_normalized * dG_dt_00 + G11_normalized * dG_dt_11 + 2.0 * G01_normalized * dG_dt_01

        # Gradient of first constraint w.r.t. deformation gradient: dCmu/dF = (G/||G||_F) * F
        dCmu_dF_col0 = G00_normalized * f0 + G01_normalized * f1  # dCmu/df0
        dCmu_dF_col1 = G01_normalized * f0 + G11_normalized * f1  # dCmu/df1

        # Gradient of constraint w.r.t. vertex position: dCmu/dx = (dCmu/dF) : (dF/dx)
        dCmu_dx = df0_dx * dCmu_dF_col0 + df1_dx * dCmu_dF_col1

        # Damping force from first constraint: -mu * damping * (dCmu/dt) * (dCmu/dx)
        kd_mu = mu * damping
        force += -kd_mu * dCmu_dt * dCmu_dx

        # Damping Hessian: mu * damping * (1/dt) * (dCmu/dx) x (dCmu/dx)
        hessian += kd_mu * inv_dt * wp.outer(dCmu_dx, dCmu_dx)

        # Second constraint: Clmbd = trace(G) = G00 + G11 (trace of Green strain)
        # Time derivative of second constraint: dClmbd/dt = trace(dG/dt)
        dClmbd_dt = dG_dt_00 + dG_dt_11

        # Gradient of second constraint w.r.t. deformation gradient: dClmbd/dF = F
        dClmbd_dF_col0 = f0  # dClmbd/df0
        dClmbd_dF_col1 = f1  # dClmbd/df1

        # Gradient of Clmbd w.r.t. vertex position: dClmbd/dx = (dClmbd/dF) : (dF/dx)
        dClmbd_dx = df0_dx * dClmbd_dF_col0 + df1_dx * dClmbd_dF_col1

        # Damping force from second constraint: -lambda * damping * (dClmbd/dt) * (dClmbd/dx)
        kd_lmbd = lmbd * damping
        force += -kd_lmbd * dClmbd_dt * dClmbd_dx

        # Damping Hessian from second constraint: lambda * damping * (1/dt) * (dClmbd/dx) x (dClmbd/dx)
        hessian += kd_lmbd * inv_dt * wp.outer(dClmbd_dx, dClmbd_dx)

    # Apply area scaling
    force *= area
    hessian *= area

    return force, hessian


@wp.func
def compute_cofactor_derivative(F: wp.mat33, scale: float) -> mat99:
    """scale * ∂cof(F)/∂F"""

    F11, F21, F31 = F[0, 0], F[1, 0], F[2, 0]
    F12, F22, F32 = F[0, 1], F[1, 1], F[2, 1]
    F13, F23, F33 = F[0, 2], F[1, 2], F[2, 2]

    return mat99(
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F33,
        -scale * F23,
        0.0,
        -scale * F32,
        scale * F22,
        0.0,
        0.0,
        0.0,
        -scale * F33,
        0.0,
        scale * F13,
        scale * F32,
        0.0,
        -scale * F12,
        0.0,
        0.0,
        0.0,
        scale * F23,
        -scale * F13,
        0.0,
        -scale * F22,
        scale * F12,
        0.0,
        0.0,
        -scale * F33,
        scale * F23,
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F31,
        -scale * F21,
        scale * F33,
        0.0,
        -scale * F13,
        0.0,
        0.0,
        0.0,
        -scale * F31,
        0.0,
        scale * F11,
        -scale * F23,
        scale * F13,
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F21,
        -scale * F11,
        0.0,
        0.0,
        scale * F32,
        -scale * F22,
        0.0,
        -scale * F31,
        scale * F21,
        0.0,
        0.0,
        0.0,
        -scale * F32,
        0.0,
        scale * F12,
        scale * F31,
        0.0,
        -scale * F11,
        0.0,
        0.0,
        0.0,
        scale * F22,
        -scale * F12,
        0.0,
        -scale * F21,
        scale * F11,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@wp.func
def assemble_tet_vertex_force_and_hessian(
    dE_dF: vec9,
    H: mat99,
    m1: float,
    m2: float,
    m3: float,
):
    f = wp.vec3(
        -(dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3),
        -(dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3),
        -(dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3),
    )
    h = wp.mat33()

    h[0, 0] += (
        m1 * (H[0, 0] * m1 + H[3, 0] * m2 + H[6, 0] * m3)
        + m2 * (H[0, 3] * m1 + H[3, 3] * m2 + H[6, 3] * m3)
        + m3 * (H[0, 6] * m1 + H[3, 6] * m2 + H[6, 6] * m3)
    )

    h[1, 0] += (
        m1 * (H[1, 0] * m1 + H[4, 0] * m2 + H[7, 0] * m3)
        + m2 * (H[1, 3] * m1 + H[4, 3] * m2 + H[7, 3] * m3)
        + m3 * (H[1, 6] * m1 + H[4, 6] * m2 + H[7, 6] * m3)
    )

    h[2, 0] += (
        m1 * (H[2, 0] * m1 + H[5, 0] * m2 + H[8, 0] * m3)
        + m2 * (H[2, 3] * m1 + H[5, 3] * m2 + H[8, 3] * m3)
        + m3 * (H[2, 6] * m1 + H[5, 6] * m2 + H[8, 6] * m3)
    )

    h[0, 1] += (
        m1 * (H[0, 1] * m1 + H[3, 1] * m2 + H[6, 1] * m3)
        + m2 * (H[0, 4] * m1 + H[3, 4] * m2 + H[6, 4] * m3)
        + m3 * (H[0, 7] * m1 + H[3, 7] * m2 + H[6, 7] * m3)
    )

    h[1, 1] += (
        m1 * (H[1, 1] * m1 + H[4, 1] * m2 + H[7, 1] * m3)
        + m2 * (H[1, 4] * m1 + H[4, 4] * m2 + H[7, 4] * m3)
        + m3 * (H[1, 7] * m1 + H[4, 7] * m2 + H[7, 7] * m3)
    )

    h[2, 1] += (
        m1 * (H[2, 1] * m1 + H[5, 1] * m2 + H[8, 1] * m3)
        + m2 * (H[2, 4] * m1 + H[5, 4] * m2 + H[8, 4] * m3)
        + m3 * (H[2, 7] * m1 + H[5, 7] * m2 + H[8, 7] * m3)
    )

    h[0, 2] += (
        m1 * (H[0, 2] * m1 + H[3, 2] * m2 + H[6, 2] * m3)
        + m2 * (H[0, 5] * m1 + H[3, 5] * m2 + H[6, 5] * m3)
        + m3 * (H[0, 8] * m1 + H[3, 8] * m2 + H[6, 8] * m3)
    )

    h[1, 2] += (
        m1 * (H[1, 2] * m1 + H[4, 2] * m2 + H[7, 2] * m3)
        + m2 * (H[1, 5] * m1 + H[4, 5] * m2 + H[7, 5] * m3)
        + m3 * (H[1, 8] * m1 + H[4, 8] * m2 + H[7, 8] * m3)
    )

    h[2, 2] += (
        m1 * (H[2, 2] * m1 + H[5, 2] * m2 + H[8, 2] * m3)
        + m2 * (H[2, 5] * m1 + H[5, 5] * m2 + H[8, 5] * m3)
        + m3 * (H[2, 8] * m1 + H[5, 8] * m2 + H[8, 8] * m3)
    )

    return f, h

@wp.func
def evaluate_volumetric_neo_hookean_force_and_hessian(
    tet_id: int,
    v_order: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    # ============ Get Vertices ============
    v0 = pos[tet_indices[tet_id, 0]]
    v1 = pos[tet_indices[tet_id, 1]]
    v2 = pos[tet_indices[tet_id, 2]]
    v3 = pos[tet_indices[tet_id, 3]]

    # ============ Compute rest volume from Dm_inv ============
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    # ============ Deformation Gradient ============
    Ds = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0)
    F = Ds * Dm_inv

    # ============ Flatten F to vec9 ============
    f = vec9(
        F[0, 0],
        F[1, 0],
        F[2, 0],
        F[0, 1],
        F[1, 1],
        F[2, 1],
        F[0, 2],
        F[1, 2],
        F[2, 2],
    )

    # ============ Useful Quantities ============
    J = wp.determinant(F)
    # Guard against division by zero in lambda (Lamé's first parameter)
    # For numerical stability, ensure lmbd has a reasonable minimum magnitude
    lmbd_safe = wp.sign(lmbd) * wp.max(wp.abs(lmbd), 1e-6)
    alpha = 1.0 + mu / lmbd_safe
    # Compute cofactor (adjugate) matrix directly for numerical stability when J ≈ 0
    cof = compute_cofactor(F)

    cof_vec = vec9(
        cof[0, 0],
        cof[1, 0],
        cof[2, 0],
        cof[0, 1],
        cof[1, 1],
        cof[2, 1],
        cof[0, 2],
        cof[1, 2],
        cof[2, 2],
    )

    # ============ Stress ============
    P_vec = rest_volume * (mu * f + lmbd * (J - alpha) * cof_vec)

    # ============ Hessian ============
    H = (
        mu * wp.identity(n=9, dtype=float)
        + lmbd * wp.outer(cof_vec, cof_vec)
        + compute_cofactor_derivative(F, lmbd * (J - alpha))
    )
    H = rest_volume * H

    # ============ Assemble Pointwise Force ============
    if v_order == 0:
        m = wp.vec3(
            -(Dm_inv[0, 0] + Dm_inv[1, 0] + Dm_inv[2, 0]),
            -(Dm_inv[0, 1] + Dm_inv[1, 1] + Dm_inv[2, 1]),
            -(Dm_inv[0, 2] + Dm_inv[1, 2] + Dm_inv[2, 2]),
        )
    elif v_order == 1:
        m = wp.vec3(Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2])
    elif v_order == 2:
        m = wp.vec3(Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2])
    else:
        m = wp.vec3(Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2])

    force, hessian = assemble_tet_vertex_force_and_hessian(P_vec, H, m[0], m[1], m[2])

    # ============ Damping ============
    if damping > 0.0:
        inv_dt = 1.0 / dt

        v0_prev = pos_prev[tet_indices[tet_id, 0]]
        v1_prev = pos_prev[tet_indices[tet_id, 1]]
        v2_prev = pos_prev[tet_indices[tet_id, 2]]
        v3_prev = pos_prev[tet_indices[tet_id, 3]]

        Ds_dot = (
            wp.matrix_from_cols(
                (v1 - v1_prev) - (v0 - v0_prev),
                (v2 - v2_prev) - (v0 - v0_prev),
                (v3 - v3_prev) - (v0 - v0_prev),
            )
            * inv_dt
        )
        F_dot = Ds_dot * Dm_inv

        f_dot = vec9(
            F_dot[0, 0],
            F_dot[1, 0],
            F_dot[2, 0],
            F_dot[0, 1],
            F_dot[1, 1],
            F_dot[2, 1],
            F_dot[0, 2],
            F_dot[1, 2],
            F_dot[2, 2],
        )

        P_damp = damping * (H * f_dot)

        f_damp = wp.vec3(
            -(P_damp[0] * m[0] + P_damp[3] * m[1] + P_damp[6] * m[2]),
            -(P_damp[1] * m[0] + P_damp[4] * m[1] + P_damp[7] * m[2]),
            -(P_damp[2] * m[0] + P_damp[5] * m[1] + P_damp[8] * m[2]),
        )
        force = force + f_damp
        hessian = hessian * (1.0 + damping * inv_dt)

    return force, hessian



@wp.kernel
def solve_elasticity(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)


    if tet_indices:
        # solve tet elasticity
        num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
        for adj_tet_counter in range(num_adj_tets):
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                particle_adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hookean_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

    # # fmt: off
    # if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
    #     wp.printf(
    #         "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
    #         particle_index,
    #         f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
    #     )

    # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        particle_displacements[particle_index] = h_inv * f



@wp.kernel
def apply_particle_displacements(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    displacements: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()
    pid = particle_ids_in_color[t_id]
    pos[pid] = pos[pid] + displacements[pid]


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia_out: wp.array(dtype=wp.vec3),
    displacements_out: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE or inv_mass[particle] == 0:
        inertia_out[particle] = pos_prev[particle]
        if displacements_out:
            displacements_out[particle] = wp.vec3(0.0, 0.0, 0.0)
        return
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    inertia = pos[particle] + vel_new * dt
    inertia_out[particle] = inertia
    if displacements_out:
        displacements_out[particle] = vel_new * dt


def initialize_particles(model, state_in: State, state_out: State, dt: float):
    wp.launch(
        kernel=forward_step,
        inputs=[
            dt,
            model.gravity,
            model.particle_q_prev,
            state_in.particle_q,
            state_in.particle_qd,
            model.particle_inv_mass,
            state_in.particle_f,
            model.particle_flags,
        ],
        outputs=[
            model.inertia,
            model.particle_displacements,
        ],
        dim=model.particle_count,
        device=model.device,
    )


def solve_particle_iteration(
    model, state_in: State, state_out: State, contacts: Contacts | None, dt: float, iter_num: int
):
    """Solve one VBD iteration for particles."""

    # Early exit if no particles
    if model.particle_count == 0:
        return

    # Zero out forces and hessians
    model.particle_forces.zero_()
    model.particle_hessians.zero_()

    # Iterate over color groups (Gauss-Seidel: apply displacements after each color)
    for color in range(len(model.particle_color_groups)):
        color_group = model.particle_color_groups[color]
        wp.launch(
            kernel=solve_elasticity,
            dim=color_group.size,
            inputs=[
                dt,
                color_group,
                model.particle_q_prev,
                state_in.particle_q,
                model.particle_mass,
                model.inertia,
                model.particle_flags,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                model.particle_adjacency,
                model.particle_forces,
                model.particle_hessians,
            ],
            outputs=[
                model.particle_displacements,
            ],
            device=model.device,
        )
        # Apply Newton step to positions (VBD Gauss-Seidel update)
        wp.launch(
            kernel=apply_particle_displacements,
            dim=color_group.size,
            inputs=[color_group, state_in.particle_q, model.particle_displacements],
            device=model.device,
        )

    wp.copy(state_out.particle_q, state_in.particle_q)


@wp.kernel
def update_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


def finalize_particles(model, state_out: State, dt: float):
    """Finalize particle velocities after VBD iterations."""
    # Early exit if no particles
    if model.particle_count == 0:
        return

    wp.launch(
        kernel=update_velocity,
        inputs=[dt, model.particle_q_prev, state_out.particle_q, state_out.particle_qd],
        dim=model.particle_count,
        device=model.device,
    )




@wp.func
def evaluate_neo_hookean_energy(
    tet_id: int,
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    mu: float,
    lmbd: float,
) -> float:
    """Compute scalar Neo-Hookean energy for a single tet."""
    v0 = pos[tet_indices[tet_id, 0]]
    v1 = pos[tet_indices[tet_id, 1]]
    v2 = pos[tet_indices[tet_id, 2]]
    v3 = pos[tet_indices[tet_id, 3]]

    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    Ds = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0)
    F = Ds * Dm_inv

    J = wp.determinant(F)
    lmbd_safe = wp.sign(lmbd) * wp.max(wp.abs(lmbd), 1e-6)
    alpha_nh = 1.0 + mu / lmbd_safe
    # I1 = tr(F^T F) = ||F||_F^2
    I1 = wp.ddot(F, F)

    # Stable Neo-Hookean: E = V * (mu/2 * (I1 - 3) + lmbd/2 * (J - alpha)^2)
    energy = rest_volume * (0.5 * mu * (I1 - 3.0) + 0.5 * lmbd * (J - alpha_nh) * (J - alpha_nh))
    return energy


@wp.func
def evaluate_vertex_merit_energy(
    v_id: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
) -> float:
    """Compute total merit energy (inertia + elastic) for a vertex."""
    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # Inertia energy: 0.5 * m * ||x - y||^2 / dt^2
    diff = pos[v_id] - inertia[v_id]
    me_inertia = 0.5 * mass[v_id] * wp.dot(diff, diff) * dt_sqr_reciprocal

    # Elastic energy: sum over adjacent tets
    me_elastic = 0.0
    num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, v_id)
    for adj_tet_counter in range(num_adj_tets):
        nei_tet_index, vertex_order = get_vertex_adjacent_tet_id_order(
            particle_adjacency, v_id, adj_tet_counter
        )
        mu = tet_materials[nei_tet_index, 0]
        lmbd = tet_materials[nei_tet_index, 1]
        if mu > 0.0 or lmbd > 0.0:
            me_elastic += evaluate_neo_hookean_energy(
                nei_tet_index,
                pos,
                tet_indices,
                tet_poses[nei_tet_index],
                mu,
                lmbd,
            )

    return me_inertia + me_elastic


@wp.func
def backtracing_line_search_vbd(
    v_id: int,
    dx: wp.vec3,
    E0: float,
    alpha: float,
    c: float,
    tau: float,
    max_num_iters: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
) -> float:
    """Backtracking line search with Armijo (first Wolfe) condition for VBD."""
    m = wp.dot(dx, dx)  # dx.squaredNorm()

    org_pos = pos[v_id]

    best_alpha = 0.0
    best_energy = E0

    for _i in range(max_num_iters):
        pos[v_id] = alpha * dx + org_pos

        e = evaluate_vertex_merit_energy(
            v_id, dt, pos, mass, inertia,
            tet_indices, tet_poses, tet_materials, particle_adjacency,
        )

        if e < best_energy:
            best_alpha = alpha
            best_energy = e

        # first Wolfe condition
        if e < E0 - alpha * c * m:
            break
        else:
            alpha = alpha * tau

    # restore vertex to the best position found
    pos[v_id] = best_alpha * dx + org_pos

    return best_energy


class SolverVBDMuscle:
    def __init__(self, model: Model, iterations: int = 20):
        self.model = model
        self.iterations = iterations

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ):
        model = self.model

        initialize_particles(model, state_in, state_out, dt)

        for iter_num in range(self.iterations):
            solve_particle_iteration(model, state_in, state_out, contacts, dt, iter_num)

        finalize_particles(model, state_out, dt)









# ---------------------------------------------------------------------------- #
#                                 ModelBuilder                                 #
# ---------------------------------------------------------------------------- #



Vec2 = list[float] | tuple[float, float] | wp.vec2
"""A 2D vector represented as a list or tuple of 2 floats."""
Vec3 = list[float] | tuple[float, float, float] | wp.vec3
"""A 3D vector represented as a list or tuple of 3 floats."""
Vec4 = list[float] | tuple[float, float, float, float] | wp.vec4
"""A 4D vector represented as a list or tuple of 4 floats."""
Quat = list[float] | tuple[float, float, float, float] | wp.quat
"""A quaternion represented as a list or tuple of 4 floats (in XYZW order)."""
Mat33 = list[float] | wp.mat33
"""A 3x3 matrix represented as a list of 9 floats or a ``warp.mat33``."""
Transform = tuple[Vec3, Quat] | wp.transform
"""A 3D transformation represented as a tuple of 3D translation and rotation quaternion (in XYZW order)."""

class ModelBuilder:
    
    def __init__(self):
        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_max_velocity = 1e5
        self.particle_color_groups: list[np.ndarray] = []
        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        self.world_gravity: list[tuple[float, float, float]] = []

        self.default_particle_radius = 0.1


    def finalize(self, device: str = "cuda:0") -> Model:
        n = self.particle_count
        m = Model(device)
        m.particle_count = n

        # --- particle state ---
        m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, device=device)
        m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, device=device)

        # --- particle properties ---
        ms = np.array(self.particle_mass, dtype=np.float32)
        inv_ms = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)
        m.particle_mass = wp.array(ms, dtype=wp.float32, device=device)
        m.particle_inv_mass = wp.array(inv_ms, dtype=wp.float32, device=device)
        m.particle_flags = wp.array([int(f) for f in self.particle_flags], dtype=wp.int32, device=device)

        # --- tet mesh ---
        if len(self.tet_indices) > 0:
            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32, device=device)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, device=device)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, device=device)
        else:
            m.tet_indices = None
            m.tet_poses = None
            m.tet_materials = None

        # --- adjacency (vertex → tet) ---
        m.particle_adjacency = self._build_adjacency(n, device)

        # --- graph coloring for parallel VBD ---
        m.particle_color_groups = self._build_graph_coloring(n, device)

        # --- gravity ---
        if self.world_gravity:
            gravity_vecs = [wp.vec3(*g) for g in self.world_gravity]
        else:
            gravity_vecs = [wp.vec3(0.0, -9.81, 0.0)]
        m.gravity = wp.array(gravity_vecs, dtype=wp.vec3, device=device)

        # --- scratch buffers ---
        m.particle_q_prev = wp.zeros(n, dtype=wp.vec3, device=device)
        m.inertia = wp.zeros(n, dtype=wp.vec3, device=device)
        m.particle_displacements = wp.zeros(n, dtype=wp.vec3, device=device)
        m.particle_forces = wp.zeros(n, dtype=wp.vec3, device=device)
        m.particle_hessians = wp.zeros(n, dtype=wp.mat33, device=device)

        m.requires_grad = False
        return m

    def _build_adjacency(self, n: int, device: str) -> ParticleForceElementAdjacencyInfo:
        """Build vertex-tet adjacency info from tet_indices."""
        adj = ParticleForceElementAdjacencyInfo()

        # tet adjacency
        adj_lists = [[] for _ in range(n)]
        for tet_id, (i, j, k, l) in enumerate(self.tet_indices):
            adj_lists[i].append((tet_id, 0))
            adj_lists[j].append((tet_id, 1))
            adj_lists[k].append((tet_id, 2))
            adj_lists[l].append((tet_id, 3))

        tet_flat = []
        tet_offsets = [0]
        for v in range(n):
            for tet_id, order in adj_lists[v]:
                tet_flat.append(tet_id)
                tet_flat.append(order)
            tet_offsets.append(len(tet_flat))

        adj.v_adj_tets = wp.array(tet_flat if tet_flat else [0], dtype=wp.int32, device=device)
        adj.v_adj_tets_offsets = wp.array(tet_offsets, dtype=wp.int32, device=device)

        # empty placeholders for faces/edges/springs (not used in this example)
        empty_flat = wp.zeros(1, dtype=wp.int32, device=device)
        empty_offsets = wp.zeros(n + 1, dtype=wp.int32, device=device)
        adj.v_adj_faces = empty_flat
        adj.v_adj_faces_offsets = empty_offsets
        adj.v_adj_edges = empty_flat
        adj.v_adj_edges_offsets = wp.clone(empty_offsets)
        adj.v_adj_springs = empty_flat
        adj.v_adj_springs_offsets = wp.clone(empty_offsets)

        return adj

    def _build_graph_coloring(self, n: int, device: str) -> list:
        """Greedy graph coloring: vertices sharing a tet get different colors."""
        neighbors = [set() for _ in range(n)]
        for i, j, k, l in self.tet_indices:
            verts = [i, j, k, l]
            for a in range(4):
                for b in range(a + 1, 4):
                    neighbors[verts[a]].add(verts[b])
                    neighbors[verts[b]].add(verts[a])

        colors = [-1] * n
        for v in range(n):
            used = {colors[nb] for nb in neighbors[v] if colors[nb] >= 0}
            c = 0
            while c in used:
                c += 1
            colors[v] = c

        num_colors = (max(colors) + 1) if n > 0 else 0
        groups = [[] for _ in range(num_colors)]
        for v, c in enumerate(colors):
            groups[c].append(v)

        print(f"Graph coloring: {n} vertices -> {num_colors} colors")
        return [wp.array(g, dtype=wp.int32, device=device) for g in groups]


    @property
    def particle_count(self):
        return len(self.particle_q)

    def add_particle(self, pos, vel, mass, radius=None, flags=None) -> int:
        """Adds a single particle to the model. Returns the particle index."""
        idx = len(self.particle_q)
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        self.particle_radius.append(radius if radius is not None else self.default_particle_radius)
        self.particle_flags.append(flags if flags is not None else ParticleFlags.ACTIVE)
        return idx
    

    
    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        particle_radius: float | None = None,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
            tri_ke: Stiffness for surface mesh triangles. Defaults to 0.0.
            tri_ka: Area stiffness for surface mesh triangles. Defaults to 0.0.
            tri_kd: Damping for surface mesh triangles. Defaults to 0.0.
            tri_drag: Drag coefficient for surface mesh triangles. Defaults to 0.0.
            tri_lift: Lift coefficient for surface mesh triangles. Defaults to 0.0.
            add_surface_mesh_edges: Whether to create zero-stiffness bending edges on the
                generated surface mesh. These edges improve collision robustness for VBD solver. Defaults to True.
            edge_ke: Bending edge stiffness used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            edge_kd: Bending edge damping used when ``add_surface_mesh_edges`` is True. Defaults to 0.0.
            particle_radius: particle's contact radius (controls rigidbody-particle contact distance)

        Note:
            The generated surface triangles and optional edges are for collision purposes.
            Their stiffness and damping values default to zero so they do not introduce additional
            elastic forces. Set the triangle stiffness parameters above to non-zero values if you
            want the surface to behave like a thin skin.
        """
        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = wp.quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m, particle_radius)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)


    def add_tetrahedron(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        k_mu: float = 1.0e3,
        k_lambda: float = 1.0e3,
        k_damp: float = 0.0,
        custom_attributes: dict[str] | None = None,
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness
            custom_attributes: Dictionary of custom attribute names to values.

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configuration

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume
    






# ---------------------------------------------------------------------------- #
#                                  PLY output                                  #
# ---------------------------------------------------------------------------- #

def extract_surface_triangles(tet_indices_np: np.ndarray) -> list[tuple[int, int, int]]:
    """Extract surface triangles from a tetrahedral mesh.

    A face shared by exactly one tet is a surface face.
    """
    face_count: dict[tuple, tuple] = {}
    for tet in tet_indices_np:
        v0, v1, v2, v3 = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        faces = [
            (v0, v2, v1),
            (v1, v2, v3),
            (v0, v1, v3),
            (v0, v3, v2),
        ]
        for f in faces:
            key = tuple(sorted(f))
            if key not in face_count:
                face_count[key] = f
            else:
                face_count[key] = None  # shared face — mark for removal
    return [f for f in face_count.values() if f is not None]


def save_ply(filename: str, positions: np.ndarray, surface_faces: list[tuple[int, int, int]]):
    """Save mesh surface to PLY file."""
    with open(filename, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(positions)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write(f"element face {len(surface_faces)}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for p in positions:
            fp.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in surface_faces:
            fp.write(f"3 {f[0]} {f[1]} {f[2]}\n")


# ---------------------------------------------------------------------------- #
#                                    example                                   #
# ---------------------------------------------------------------------------- #
class Example:
    def __init__(self, device: str):
        self.device = device

        builder = ModelBuilder()
        dim_x, dim_y, dim_z = 12, 4, 4
        cell_size = 0.1
        k_damp = 1e-1

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=k_damp,
            fix_left=True,
        )
        self.model = builder.finalize(device=device)

        self.solver = SolverVBDMuscle(self.model)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.nsubsteps = 1
        self.dt = 1.0 / 60.0

        # Pre-compute surface faces for PLY output
        tet_indices_np = np.array(builder.tet_indices, dtype=np.int32)
        self.surface_faces = extract_surface_triangles(tet_indices_np)

        # Output directory
        self.output_dir = "output/ply"
        os.makedirs(self.output_dir, exist_ok=True)

    def step(self, dt):
        for _ in range(self.nsubsteps):
            self.solver.step(self.state_in, self.state_out, control=None, contacts=None, dt=dt)
            self.state_in, self.state_out = self.state_out, self.state_in

    def save_frame(self, frame: int):
        """Save current particle positions to PLY."""
        positions = self.state_in.particle_q.numpy()
        filename = os.path.join(self.output_dir, f"frame_{frame:04d}.ply")
        save_ply(filename, positions, self.surface_faces)

    def run(self):
        num_steps = 300
        save_interval = 1

        print("=== VBD Simulation ===")
        print(f"  particles: {self.model.particle_count}")
        print(f"  colors:    {len(self.model.particle_color_groups)}")
        print(f"  steps:     {num_steps}")
        print(f"  dt:        {self.dt:.4f}s")
        print()

        self.save_frame(0)

        for istep in range(num_steps):
            t0 = time.time()
            self.step(self.dt)
            elapsed = time.time() - t0

            if (istep + 1) % save_interval == 0:
                self.save_frame(istep + 1)
                pos = self.state_in.particle_q.numpy()
                pos_min = pos.min(axis=0)
                pos_max = pos.max(axis=0)
                print(
                    f"Step {istep + 1:4d}/{num_steps} | "
                    f"dt={elapsed:.4f}s | "
                    f"y=[{pos_min[1]:.3f}, {pos_max[1]:.3f}]"
                )

        self.save_frame(num_steps)
        print(f"\nDone. Frames saved to {self.output_dir}/")


if __name__ == "__main__":
    wp.init()
    example = Example(device="cuda:0")
    example.run()
