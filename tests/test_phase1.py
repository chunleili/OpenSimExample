"""Phase 1 verification: DGF curves, gradient FD tests, Hessian FD tests.

Run with: uv run python tests/test_phase1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vbd_muscle.dgf_curves import (
    active_force_length, active_force_length_deriv,
    passive_force_length, passive_force_length_deriv,
    force_velocity,
)
from vbd_muscle.fem import (
    compute_rest_data, compute_deformation_gradient,
    get_b_vec, vertex_gradient_from_pk1,
)
from vbd_muscle.constitutive import (
    neo_hookean_energy, neo_hookean_pk1,
    fiber_pk1, total_pk1, total_energy,
    vertex_hessian_fd,
)
from vbd_muscle.mesh import generate_box_mesh
from vbd_muscle.activation import activation_dynamics

PASS = "PASS"
FAIL = "FAIL"


def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


# --------------------------------------------------------------------------
# 1. DGF curve derivative verification
# --------------------------------------------------------------------------

def test_dgf_derivatives():
    print("\n=== DGF curve derivative tests ===")
    ok = True
    eps = 1e-7

    l_vals = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6]

    for l in l_vals:
        fd = (active_force_length(l + eps) - active_force_length(l - eps)) / (2 * eps)
        an = active_force_length_deriv(l)
        err = abs(fd - an) / (abs(an) + 1e-12)
        ok &= report(f"f_L'({l:.1f})", err < 1e-4, f"rel_err={err:.2e}")

    for l in l_vals:
        fd = (passive_force_length(l + eps) - passive_force_length(l - eps)) / (2 * eps)
        an = passive_force_length_deriv(l)
        err = abs(fd - an) / (abs(an) + 1e-12)
        ok &= report(f"f_PE'({l:.1f})", err < 1e-4, f"rel_err={err:.2e}")

    return ok


# --------------------------------------------------------------------------
# 2. DGF curve sanity checks
# --------------------------------------------------------------------------

def test_dgf_values():
    print("\n=== DGF curve value sanity checks ===")
    ok = True

    # f_L(1.0) should be close to 1.0 (peak at optimal length)
    fl1 = active_force_length(1.0)
    ok &= report("f_L(1.0) ~ 1.0", abs(fl1 - 1.0) < 0.05, f"val={fl1:.4f}")

    # f_L should be small at extremes
    ok &= report("f_L(0.4) ~ 0", active_force_length(0.4) < 0.05)
    ok &= report("f_L(1.8) ~ 0", active_force_length(1.8) < 0.05)

    # f_PE(1.0) should be small (OpenSim-aligned curve has small offset)
    fp1 = passive_force_length(1.0)
    ok &= report("f_PE(1.0) ~ 0", fp1 < 0.05, f"val={fp1:.6f}")

    # f_PE(1.6) should be ~1.0
    fp16 = passive_force_length(1.6)
    ok &= report("f_PE(1.6) ~ 1.0", abs(fp16 - 1.0) < 0.15, f"val={fp16:.4f}")

    # f_V(0) = 1.0 (isometric)
    fv0 = force_velocity(0.0)
    ok &= report("f_V(0) = 1.0", abs(fv0 - 1.0) < 0.01, f"val={fv0:.4f}")

    # f_V(-1) ~ 0 (max shortening)
    fvm1 = force_velocity(-1.0)
    ok &= report("f_V(-1) ~ 0", abs(fvm1) < 0.05, f"val={fvm1:.4f}")

    # f_V(1) ~ 1.794 (max eccentric)
    fv1 = force_velocity(1.0)
    ok &= report("f_V(1) ~ 1.794", abs(fv1 - 1.794) < 0.05, f"val={fv1:.4f}")

    return ok


# --------------------------------------------------------------------------
# 3. Neo-Hookean PK1 gradient test (single tet)
# --------------------------------------------------------------------------

def test_neo_hookean_gradient():
    print("\n=== Neo-Hookean PK1 gradient test ===")
    ok = True

    # Reference tet (regular tetrahedron, ~10 cm scale)
    X = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ])
    tets = np.array([[0, 1, 2, 3]])
    Dm_inv, volumes = compute_rest_data(X, tets)
    V = volumes[0]
    B = Dm_inv[0]

    mu, kappa = 5000.0, 500000.0

    # Apply non-trivial deformation
    x = X.copy()
    x[1, 0] *= 1.15
    x[2, 1] *= 0.92
    x[3, 2] *= 1.08
    x[1, 1] += 0.005  # shear

    eps = 1e-7

    for local_idx in range(4):
        b_vec = get_b_vec(B, local_idx)
        F = compute_deformation_gradient(x, B)
        P = neo_hookean_pk1(F, mu, kappa)
        g_an = vertex_gradient_from_pk1(P, V, b_vec)

        # FD gradient from energy
        g_fd = np.zeros(3)
        for d in range(3):
            x_plus = x.copy()
            x_plus[local_idx, d] += eps
            F_plus = compute_deformation_gradient(x_plus, B)
            E_plus = V * neo_hookean_energy(F_plus, mu, kappa)

            x_minus = x.copy()
            x_minus[local_idx, d] -= eps
            F_minus = compute_deformation_gradient(x_minus, B)
            E_minus = V * neo_hookean_energy(F_minus, mu, kappa)

            g_fd[d] = (E_plus - E_minus) / (2 * eps)

        rel_err = np.linalg.norm(g_an - g_fd) / (np.linalg.norm(g_fd) + 1e-30)
        ok &= report(
            f"Neo-Hookean grad vertex {local_idx}",
            rel_err < 1e-4,
            f"rel_err={rel_err:.2e}")

    return ok


# --------------------------------------------------------------------------
# 4. Total PK1 gradient test (Neo-Hookean + fiber)
# --------------------------------------------------------------------------

def test_total_gradient():
    print("\n=== Total PK1 (Neo-Hookean + fiber) gradient test ===")
    ok = True

    X = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ])
    tets = np.array([[0, 1, 2, 3]])
    Dm_inv, volumes = compute_rest_data(X, tets)
    V = volumes[0]
    B = Dm_inv[0]

    d0 = np.array([0.0, 0.0, 1.0])
    mu, kappa = 5000.0, 500000.0
    sigma0 = 300000.0
    activation = 0.8

    x = X.copy()
    x[1, 0] *= 1.12
    x[2, 1] *= 0.95
    x[3, 2] *= 1.06
    x[2, 2] += 0.003

    eps = 1e-7

    for local_idx in range(4):
        b_vec = get_b_vec(B, local_idx)
        F = compute_deformation_gradient(x, B)
        P = total_pk1(F, d0, mu, kappa, sigma0, activation)
        g_an = vertex_gradient_from_pk1(P, V, b_vec)

        g_fd = np.zeros(3)
        for d in range(3):
            x_plus = x.copy()
            x_plus[local_idx, d] += eps
            F_plus = compute_deformation_gradient(x_plus, B)
            E_plus = V * total_energy(F_plus, d0, mu, kappa, sigma0, activation)

            x_minus = x.copy()
            x_minus[local_idx, d] -= eps
            F_minus = compute_deformation_gradient(x_minus, B)
            E_minus = V * total_energy(F_minus, d0, mu, kappa, sigma0, activation)

            g_fd[d] = (E_plus - E_minus) / (2 * eps)

        rel_err = np.linalg.norm(g_an - g_fd) / (np.linalg.norm(g_fd) + 1e-30)
        ok &= report(
            f"Total grad vertex {local_idx}",
            rel_err < 1e-4,
            f"rel_err={rel_err:.2e}")

    return ok


# --------------------------------------------------------------------------
# 5. Hessian FD consistency test
# --------------------------------------------------------------------------

def test_hessian_consistency():
    """Verify FD Hessian (eps=1e-7) vs FD Hessian (eps=1e-6)."""
    print("\n=== Hessian FD consistency test ===")
    ok = True

    X = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ])
    tets = np.array([[0, 1, 2, 3]])
    Dm_inv, volumes = compute_rest_data(X, tets)

    d0 = np.array([0.0, 0.0, 1.0])
    mu, kappa = 5000.0, 500000.0
    sigma0 = 300000.0
    activation = 0.8

    x = X.copy()
    x[1, 0] *= 1.1
    x[3, 2] *= 1.05

    for local_idx in range(4):
        H1 = vertex_hessian_fd(
            x, Dm_inv[0], volumes[0], local_idx,
            d0, mu, kappa, sigma0, activation, eps=1e-7)
        H2 = vertex_hessian_fd(
            x, Dm_inv[0], volumes[0], local_idx,
            d0, mu, kappa, sigma0, activation, eps=1e-6)
        rel_err = np.linalg.norm(H1 - H2) / (np.linalg.norm(H1) + 1e-30)
        ok &= report(
            f"Hessian consistency vertex {local_idx}",
            rel_err < 1e-2,
            f"rel_err={rel_err:.2e}")

    return ok


# --------------------------------------------------------------------------
# 6. Mesh generation test
# --------------------------------------------------------------------------

def test_box_mesh():
    print("\n=== Box mesh generation test ===")
    ok = True

    nodes, tets = generate_box_mesh(0.01, 0.01, 0.1, 2, 2, 10)
    n_verts = (2 + 1) * (2 + 1) * (10 + 1)
    n_elems = 6 * 2 * 2 * 10

    ok &= report("Vertex count", len(nodes) == n_verts,
                  f"got {len(nodes)}, expected {n_verts}")
    ok &= report("Element count", len(tets) == n_elems,
                  f"got {len(tets)}, expected {n_elems}")

    # Check all tets have positive volume
    Dm_inv, volumes = compute_rest_data(nodes, tets)
    ok &= report("All volumes > 0", np.all(volumes > 0),
                  f"min_vol={volumes.min():.2e}")

    # Check total volume
    total_vol = volumes.sum()
    expected_vol = 0.01 * 0.01 * 0.1
    ok &= report("Total volume correct",
                  abs(total_vol - expected_vol) / expected_vol < 1e-10,
                  f"got {total_vol:.6e}, expected {expected_vol:.6e}")

    return ok


# --------------------------------------------------------------------------
# 7. Activation dynamics test
# --------------------------------------------------------------------------

def test_activation_dynamics():
    print("\n=== Activation dynamics test ===")
    ok = True

    dt = 0.001  # 1 ms
    a = 0.01  # start low

    # Step excitation to 1.0, record activation rise
    a_vals = [a]
    for _ in range(200):
        a = activation_dynamics(1.0, a, dt)
        a_vals.append(float(a))

    # Activation should approach 1.0
    ok &= report("Activation rises to ~1.0",
                  abs(a_vals[-1] - 1.0) < 0.01,
                  f"final_a={a_vals[-1]:.4f}")

    # Should reach ~63% in about tau_a / factor
    # tau_a = 15 ms, so ~15 steps should give significant rise
    ok &= report("Activation > 0.5 after 30ms",
                  a_vals[30] > 0.5,
                  f"a(30ms)={a_vals[30]:.4f}")

    # Deactivation: step excitation to 0
    a = 1.0
    for _ in range(200):
        a = activation_dynamics(0.0, a, dt)
    ok &= report("Deactivation to ~0",
                  a < 0.05,
                  f"final_a={a:.4f}")

    return ok


# --------------------------------------------------------------------------
# 8. VBD solver basic test
# --------------------------------------------------------------------------

def test_vbd_basic():
    """Test VBD solver on a small mesh with known rest state."""
    print("\n=== VBD solver basic test ===")
    ok = True

    from vbd_muscle.solver import VBDSolver
    from vbd_muscle.mesh import assign_fiber_directions

    # Small box mesh
    nodes, tets = generate_box_mesh(0.01, 0.01, 0.05, 2, 2, 5)
    fiber_dirs = assign_fiber_directions(nodes, tets)

    solver = VBDSolver(
        nodes, tets, fiber_dirs,
        mu=5000.0, kappa=500000.0, sigma0=0.0,  # no fiber for this test
        density=1060.0, damping=0.01, dt=0.001,
        n_iterations=5, gravity=np.array([0.0, 0.0, 0.0]),  # no gravity
    )
    solver.mesh_info()

    # At rest, no forces -> should stay put
    x_before = solver.get_positions().copy()
    solver.step(activation=0.0)
    x_after = solver.get_positions()
    max_disp = np.max(np.linalg.norm(x_after - x_before, axis=1))
    ok &= report("Rest state stable", max_disp < 1e-6,
                  f"max_disp={max_disp:.2e}")

    return ok


# --------------------------------------------------------------------------
# 9. Fiber PK1 stress vs DGF analytical
# --------------------------------------------------------------------------

def test_fiber_pk1_stress():
    """Verify fiber_pk1 Cauchy stress matches DGF analytical curves.

    For uniaxial incompressible deformation F = diag(1/sqrt(lam), 1/sqrt(lam), lam)
    with d0 = [0,0,1], the Cauchy stress zz component should equal:
        sigma_zz = sigma0 * (activation * f_L(lam) + f_PE(lam)) * lam
    The extra lam factor comes from the PK1 -> Cauchy conversion (P @ F^T).
    """
    print("\n=== Fiber PK1 stress vs DGF analytical ===")
    ok = True

    d0 = np.array([0.0, 0.0, 1.0])
    sigma0 = 300000.0

    lam_values = [0.5, 0.7, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
    activations = [0.0, 0.5, 1.0]

    for act in activations:
        for lam in lam_values:
            # Incompressible uniaxial deformation
            lat = 1.0 / np.sqrt(lam)
            F = np.diag([lat, lat, lam])
            J = np.linalg.det(F)  # should be ~1.0

            P_fiber = fiber_pk1(F, d0, sigma0, act)

            # Cauchy stress: sigma = (1/J) * P @ F^T
            cauchy = (1.0 / J) * P_fiber @ F.T
            sigma_zz = cauchy[2, 2]

            # DGF analytical (extra lam from PK1->Cauchy conversion)
            fL = active_force_length(lam)
            fPE = passive_force_length(lam)
            sigma_expected = sigma0 * (act * fL + fPE) * lam

            err = abs(sigma_zz - sigma_expected) / (abs(sigma_expected) + 1.0)
            ok &= report(
                f"fiber stress a={act:.1f} lam={lam:.2f}",
                err < 1e-10,
                f"got={sigma_zz:.2f}, expected={sigma_expected:.2f}, rel_err={err:.2e}")

    return ok


# --------------------------------------------------------------------------
# 10. Reaction force under fully-constrained uniform deformation
# --------------------------------------------------------------------------

def test_reaction_force_uniform():
    """Verify reaction forces under fully-prescribed uniform deformation.

    All vertices are directly placed at x_i = F * X_i (no solver run).
    The sum of elastic gradients at z=0 face vertices should equal
    the PK1 traction on the reference z=0 face: F_z = -P_zz * A_ref.
    (Negative because the outward normal of z=0 face is [0,0,-1].)
    """
    print("\n=== Reaction force (fully constrained uniform deformation) ===")
    ok = True

    from vbd_muscle.solver import VBDSolver
    from vbd_muscle.mesh import assign_fiber_directions

    l_opt = 0.10
    sigma0 = 300000.0
    mu = 5000.0
    kappa = 100 * mu
    F0 = 1000.0
    PCSA = F0 / sigma0
    side = np.sqrt(PCSA)

    nodes, tets = generate_box_mesh(side, side, l_opt, 3, 3, 6)
    fiber_dirs = assign_fiber_directions(nodes, tets)
    A_ref = side ** 2  # reference cross-section area

    lam_values = [0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    for lam in lam_values:
        solver = VBDSolver(
            nodes, tets, fiber_dirs,
            mu=mu, kappa=kappa, sigma0=sigma0,
            density=1060.0, damping=0.01, dt=0.001,
            n_iterations=1,
            gravity=np.array([0.0, 0.0, 0.0]),
        )

        # Directly set ALL vertex positions to uniform deformation
        # (set_prescribed_positions does NOT update solver.x)
        lat = 1.0 / np.sqrt(max(lam, 0.01))
        center_x, center_y = side / 2.0, side / 2.0
        for vi in range(len(nodes)):
            solver.x[vi] = np.array([
                center_x + (nodes[vi, 0] - center_x) * lat,
                center_y + (nodes[vi, 1] - center_y) * lat,
                nodes[vi, 2] * lam,
            ])

        # Compute reaction forces at z=0 face
        z_min_verts = np.where(np.abs(nodes[:, 2]) < 1e-10)[0]
        rf = solver.compute_reaction_forces(z_min_verts, activation=1.0)
        total_Fz = sum(f[2] for f in rf.values())

        # Analytical: PK1 stress for this uniform F
        F_mat = np.diag([lat, lat, lam])
        d0 = np.array([0.0, 0.0, 1.0])
        P = total_pk1(F_mat, d0, mu, kappa, sigma0, 1.0)

        # Expected z-force on z=0 face:
        # By FEM divergence, sum of elastic gradients at z=0 face =
        # integral of P @ N dA over the reference z=0 face, where N=[0,0,-1].
        # So F_z_expected = P_zz * (-1) * A_ref = -P_zz * A_ref
        F_expected = -P[2, 2] * A_ref

        rel_err = abs(total_Fz - F_expected) / (abs(F_expected) + 1.0)
        ok &= report(
            f"reaction force lam={lam:.2f}",
            rel_err < 1e-6,
            f"Fz={total_Fz:.2f} N, expected={F_expected:.2f} N, rel_err={rel_err:.2e}")

    return ok


# ==========================================================================

def main():
    print("=" * 60)
    print("Phase 1 Verification Tests")
    print("=" * 60)

    all_ok = True
    all_ok &= test_dgf_values()
    all_ok &= test_dgf_derivatives()
    all_ok &= test_neo_hookean_gradient()
    all_ok &= test_total_gradient()
    all_ok &= test_hessian_consistency()
    all_ok &= test_fiber_pk1_stress()
    all_ok &= test_reaction_force_uniform()
    all_ok &= test_box_mesh()
    all_ok &= test_activation_dynamics()
    all_ok &= test_vbd_basic()

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
