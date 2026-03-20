"""Phase 3.1 Level 0: Single muscle F-L / activation dynamics validation.

Compares FEM volumetric muscle force response against DGF analytical curves
and (optionally) OpenSim Level 0 output.

Usage:
    uv run python run_level0_fem.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from vbd_muscle.dgf_curves import active_force_length, passive_force_length
from vbd_muscle.mesh import generate_box_mesh, assign_fiber_directions
from vbd_muscle.solver import VBDSolver
from vbd_muscle.activation import activation_dynamics

# -----------------------------------------------------------------------
# Muscle parameters (matching validation_level0_single_muscle.py)
# -----------------------------------------------------------------------
MUSCLE_PARAMS = {
    "max_isometric_force": 1000.0,       # N
    "optimal_fiber_length": 0.10,        # m
    "tendon_slack_length": 0.20,         # m (not used in fiber-only test)
    "pennation_angle_at_optimal": 0.0,   # rad
    "fiber_damping": 0.01,
    "max_contraction_velocity": 10.0,    # l_opt/s
}

sigma0 = 300000.0   # Pa, peak isometric stress
F0 = MUSCLE_PARAMS["max_isometric_force"]  # N
PCSA = F0 / sigma0  # m^2
l_opt = MUSCLE_PARAMS["optimal_fiber_length"]
side = np.sqrt(PCSA)  # square cross-section side length

mu = 5000.0          # Pa, muscle shear modulus
kappa = 100 * mu     # Pa, bulk modulus (near-incompressible)


# =======================================================================
# Part 1: Analytical DGF Curves (reference)
# =======================================================================

def plot_analytical_curves():
    """Plot DGF analytical force-length curves."""
    lam_range = np.linspace(0.4, 1.8, 200)
    fL = active_force_length(lam_range)
    fPE = passive_force_length(lam_range)
    f_total = fL + fPE  # activation = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Active + passive curves
    ax = axes[0]
    ax.plot(lam_range, fL, 'b-', label='$f_L$ (active)', linewidth=2)
    ax.plot(lam_range, fPE, 'r-', label='$f_{PE}$ (passive)', linewidth=2)
    ax.plot(lam_range, f_total, 'k--', label='Total ($a=1$)', linewidth=2)
    ax.set_xlabel(r'Normalized fiber length $\lambda$')
    ax.set_ylabel('Normalized force')
    ax.set_title('DGF Analytical Force-Length Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.8)

    # Activation levels
    ax = axes[1]
    for a_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        f = a_val * fL + fPE
        ax.plot(lam_range, f, label=f'a={a_val:.2f}', linewidth=1.5)
    ax.set_xlabel(r'Normalized fiber length $\lambda$')
    ax.set_ylabel('Normalized force')
    ax.set_title('Force-Length at Different Activations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.8)

    plt.tight_layout()
    plt.savefig('output/output_level0_analytical_curves.png', dpi=150)
    print("Saved output/output_level0_analytical_curves.png")
    return fig



# =======================================================================
# Part 3: VBD quasi-static F-L validation
# =======================================================================

def vbd_force_length(lam_values, activation=1.0, nx=2, ny=2, nz=8,
                     n_iterations=100):
    """Run VBD quasi-static simulations at different fiber stretches.

    Uses incremental loading: stretches are applied in sorted order,
    reusing the previous converged state as initial guess.

    Args:
        lam_values: Array of target stretch ratios.
        activation: Muscle activation level.
        nx, ny, nz: Mesh resolution.
        n_iterations: VBD static solve iterations per stretch.
    Returns:
        forces: Axial reaction force at each stretch, shape (len(lam_values),).
    """
    nodes, tets = generate_box_mesh(side, side, l_opt, nx, ny, nz)
    fiber_dirs = assign_fiber_directions(nodes, tets)

    # Identify boundary vertices
    z_coords = nodes[:, 2]
    z_min_verts = np.where(np.abs(z_coords - 0.0) < 1e-10)[0]
    z_max_verts = np.where(np.abs(z_coords - l_opt) < 1e-10)[0]

    # Create a single solver instance and reuse across stretches
    solver = VBDSolver(
        nodes, tets, fiber_dirs,
        mu=mu, kappa=kappa, sigma0=sigma0,
        density=1060.0, damping=0.01, dt=0.001,
        n_iterations=20,
        gravity=np.array([0.0, 0.0, 0.0]),
    )

    forces = np.zeros(len(lam_values))

    for i, lam in enumerate(lam_values):
        # Create fresh solver for each stretch to avoid state leaking
        solver = VBDSolver(
            nodes, tets, fiber_dirs,
            mu=mu, kappa=kappa, sigma0=sigma0,
            density=1060.0, damping=0.01, dt=0.001,
            n_iterations=20,
            gravity=np.array([0.0, 0.0, 0.0]),
        )

        # Set initial guess: uniform axial stretch, incompressible lateral
        lat = 1.0 / np.sqrt(max(lam, 0.1))  # lateral scale
        center_x = side / 2.0
        center_y = side / 2.0
        for vi in range(len(nodes)):
            solver.x[vi, 2] = nodes[vi, 2] * lam
            solver.x[vi, 0] = center_x + (nodes[vi, 0] - center_x) * lat
            solver.x[vi, 1] = center_y + (nodes[vi, 1] - center_y) * lat

        # Fix only z-coordinate at both end faces (x,y free for Poisson)
        # z=0 face: fix z=0
        z_min_prescribed = np.zeros((len(z_min_verts), 1))
        solver.set_fixed_dof(z_min_verts, [2], z_min_prescribed)

        # z=L face: fix z = lam * l_opt
        z_max_prescribed = np.full((len(z_max_verts), 1), lam * l_opt)
        solver.set_fixed_dof(z_max_verts, [2], z_max_prescribed)

        # Fix one corner vertex fully to prevent rigid body translation
        solver.set_fixed_vertices([z_min_verts[0]])

        # Quasi-static solve
        converged, n_iters = solver.solve_static(
            activation=activation,
            n_iterations=n_iterations,
            tol=1e-8,
        )

        # Measure reaction force at z=0
        rf = solver.compute_reaction_forces(z_min_verts, activation=activation)
        total_force_z = sum(f[2] for f in rf.values())
        forces[i] = total_force_z

        status = "converged" if converged else f"not converged ({n_iters} iters)"
        print(f"  lambda={lam:.2f}: F_z={total_force_z:.2f} N [{status}]")

    return forces


def run_vbd_fl_validation():
    """Run VBD F-L validation and compare with DGF."""
    print("\n--- VBD Force-Length Validation ---")
    print(f"  Mesh: side={side*100:.2f} cm, length={l_opt*100:.1f} cm")
    print(f"  PCSA={PCSA*1e4:.2f} cm^2, F0={F0:.0f} N, sigma0={sigma0/1e3:.0f} kPa")

    lam_values = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
    forces = vbd_force_length(lam_values, activation=1.0, nx=2, ny=2, nz=5,
                              n_iterations=200)

    # DGF reference
    fL = active_force_length(lam_values)
    fPE = passive_force_length(lam_values)
    dgf_forces = F0 * (fL + fPE)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam_values, forces / F0, 'bo-', label='VBD (quasi-static)', linewidth=2)
    ax.plot(lam_values, dgf_forces / F0, 'r--', label='DGF reference', linewidth=2)
    ax.set_xlabel(r'Normalized fiber length $\lambda$')
    ax.set_ylabel(r'Normalized force ($F / F_0$)')
    ax.set_title('VBD vs DGF Force-Length Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/output_level0_vbd_fl.png', dpi=150)
    print("Saved output/output_level0_vbd_fl.png")

    # RMSE
    rmse = np.sqrt(np.mean((forces / F0 - dgf_forces / F0) ** 2))
    print(f"\n  RMSE (normalized): {rmse:.4f}")
    print(f"  Target: < 0.05")

    return fig


# =======================================================================
# Part 4: Activation dynamics validation
# =======================================================================

def run_activation_dynamics_validation():
    """Compare activation dynamics step response."""
    print("\n--- Activation Dynamics Validation ---")

    dt = 0.001  # 1 ms
    t_max = 0.3  # 300 ms
    n_steps = int(t_max / dt)

    times = np.arange(n_steps + 1) * dt
    activations = np.zeros(n_steps + 1)
    activations[0] = 0.01

    # Step excitation from 0.01 to 1.0 at t=0
    for i in range(n_steps):
        activations[i + 1] = activation_dynamics(1.0, activations[i], dt)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times * 1000, activations, 'b-', linewidth=2, label='Activation')
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Activation')
    ax.set_title('Activation Dynamics (Step Input e: 0.01 -> 1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/output_level0_activation.png', dpi=150)
    print("Saved output/output_level0_activation.png")

    # Report timing
    idx_50 = np.searchsorted(activations, 0.5)
    idx_90 = np.searchsorted(activations, 0.9)
    idx_99 = np.searchsorted(activations, 0.99)
    print(f"  Time to 50%: {times[idx_50]*1000:.1f} ms")
    print(f"  Time to 90%: {times[idx_90]*1000:.1f} ms")
    print(f"  Time to 99%: {times[idx_99]*1000:.1f} ms")

    return fig


# =======================================================================
# Main
# =======================================================================

def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("Level 0 Validation: Single Muscle Force-Length")
    print("=" * 60)

    print(f"\nParameters:")
    print(f"  F0 = {F0:.0f} N")
    print(f"  l_opt = {l_opt*100:.1f} cm")
    print(f"  PCSA = {PCSA*1e4:.2f} cm^2")
    print(f"  sigma0 = {sigma0/1e3:.0f} kPa")
    print(f"  mu = {mu/1e3:.1f} kPa, kappa = {kappa/1e3:.0f} kPa")
    print(f"  Cross-section side = {side*100:.2f} cm")

    # Part 1: Analytical DGF curves
    print("\n--- Part 1: Analytical DGF Curves ---")
    plot_analytical_curves()

    # Part 2: VBD quasi-static (slow in pure Python, optional)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-vbd', action='store_true',
                        help='Skip VBD quasi-static validation (slow)')
    args, _ = parser.parse_known_args()

    if not args.skip_vbd:
        print("\n--- Part 2: VBD Quasi-Static F-L ---")
        run_vbd_fl_validation()
    else:
        print("\n--- Part 2: Skipped (use without --skip-vbd to run) ---")

    # Part 3: Activation dynamics
    run_activation_dynamics_validation()

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
