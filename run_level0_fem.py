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

    # Part 2: Activation dynamics
    run_activation_dynamics_validation()

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
