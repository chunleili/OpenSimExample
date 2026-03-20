"""Constitutive models: Neo-Hookean isotropic + Hill-type fiber.

Energy density: W(F, a) = W_iso(F) + W_fiber(F, a)
  W_iso  = (mu/2)(J^{-2/3} I1 - 3) + (kappa/2)(J - 1)^2
  W_fiber: Hill-type via DGF analytical curves
"""

import numpy as np
from .dgf_curves import (
    active_force_length,
    active_force_length_deriv,
    passive_force_length,
    passive_force_length_deriv,
)
from .fem import compute_deformation_gradient, get_b_vec, vertex_gradient_from_pk1


# ---------------------------------------------------------------------------
# Neo-Hookean (deviatoric-volumetric split)
# ---------------------------------------------------------------------------

def neo_hookean_energy(F, mu, kappa):
    """Modified Neo-Hookean energy density W_iso(F)."""
    J = np.linalg.det(F)
    J_safe = max(J, 1e-10)
    I1 = np.sum(F ** 2)  # tr(F^T F)
    I1_bar = J_safe ** (-2.0 / 3.0) * I1
    return 0.5 * mu * (I1_bar - 3.0) + 0.5 * kappa * (J_safe - 1.0) ** 2


def neo_hookean_pk1(F, mu, kappa):
    """1st Piola-Kirchhoff stress for modified Neo-Hookean.

    P = mu J^{-2/3} (F - I1/3 F^{-T}) + kappa J(J-1) F^{-T}
    """
    J = np.linalg.det(F)
    if J < 1e-10:
        # Element is inverted/degenerate: use regularized version
        # Return a penalty force pushing toward positive volume
        J_clamped = max(J, 1e-10)
        return kappa * (J_clamped - 1.0) * F  # simplified penalty
    I1 = np.sum(F ** 2)
    try:
        Finv_T = np.linalg.inv(F).T
    except np.linalg.LinAlgError:
        return kappa * (J - 1.0) * F
    P = (mu * J ** (-2.0 / 3.0) * (F - (I1 / 3.0) * Finv_T)
         + kappa * J * (J - 1.0) * Finv_T)
    return P


# ---------------------------------------------------------------------------
# Hill-type fiber
# ---------------------------------------------------------------------------
# https://github.com/opensim-org/opensim-core/blob/aa40605bae4ecb81c0bae23638f2cd90f20202e5/OpenSim/Actuators/DeGrooteFregly2016Muscle.h#L579
def fiber_pk1(F, d0, sigma0, activation, scale=1.0, e0=0.6,
              fv_multiplier=1.0, fiber_damping=0.0, norm_fiber_velocity=0.0):
    """1st PK stress for Hill-type fiber (aligned with OpenSim calcFiberForce).

    Fiber force decomposition (OpenSim convention):
      active       = sigma0 * a * f_L(l~) * f_V(v~)
      con_passive  = sigma0 * f_PE(l~)
      noncon_passive = sigma0 * fiber_damping * v~
      total        = active + con_passive + noncon_passive

    PK1 stress:
      P_fiber = (total / l~) * (F d0)(d0^T)

    where l~ = ||F d0|| is the normalized fiber length (assuming reference
    mesh is built at optimal fiber length, so stretch ratio = l / l_opt).

    Args:
        F: Deformation gradient, shape (3, 3).
        d0: Reference fiber direction (unit vector), shape (3,).
        sigma0: Peak isometric stress (Pa).
        activation: Muscle activation, scalar in [0, 1].
        scale: Active force-length width scale.
        e0: Passive fiber strain at one normalized force.
        fv_multiplier: Force-velocity multiplier f_V(v~), default 1.0 (isometric).
        fiber_damping: Damping coefficient (OpenSim default 0.0).
        norm_fiber_velocity: Normalized fiber velocity v~ in [-1, 1].
    """
    Fd0 = F @ d0
    l_tilde = np.linalg.norm(Fd0)
    l_tilde = max(l_tilde, 1e-8)

    fL = float(active_force_length(l_tilde, scale))
    fPE = float(passive_force_length(l_tilde, e0=e0))

    # OpenSim decomposition: active + conservative passive + non-conservative passive
    active = activation * fL * fv_multiplier
    con_passive = fPE
    noncon_passive = fiber_damping * norm_fiber_velocity
    f_total = sigma0 * (active + con_passive + noncon_passive)

    return (f_total / l_tilde) * np.outer(Fd0, d0)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def total_pk1(F, d0, mu, kappa, sigma0, activation,
              scale=1.0, e0=0.6):
    """Total 1st PK stress (Neo-Hookean + fiber)."""
    P = neo_hookean_pk1(F, mu, kappa)
    if sigma0 > 0:
        P += fiber_pk1(F, d0, sigma0, activation, scale, e0)
    # Guard against NaN/Inf from degenerate elements
    if not np.all(np.isfinite(P)):
        P = np.where(np.isfinite(P), P, 0.0)
    return P


def total_energy(F, d0, mu, kappa, sigma0, activation,
                 scale=1.0, e0=0.6):
    """Total energy density (for gradient verification via FD).

    The fiber energy is computed by numerical integration of the fiber stress:
      W_fiber = sigma0 * integral_1^l~ [a*f_L(s) + f_PE(s)] ds
    """
    W = neo_hookean_energy(F, mu, kappa)

    if sigma0 > 0:
        Fd0 = F @ d0
        l_tilde = max(np.linalg.norm(Fd0), 1e-8)

        from scipy.integrate import quad

        def integrand(s):
            return float(sigma0 * (activation * active_force_length(s, scale)
                                   + passive_force_length(s, e0=e0)))

        W_fiber, _ = quad(integrand, 1.0, l_tilde)
        W += W_fiber

    return W


# ---------------------------------------------------------------------------
# Per-vertex Hessian (finite-difference of gradient)
# ---------------------------------------------------------------------------

def vertex_hessian_fd(x_elem, Dm_inv, volume, local_idx,
                      d0, mu, kappa, sigma0, activation,
                      scale=1.0, e0=0.6, eps=1e-7):
    """Compute per-vertex 3x3 Hessian via finite differences of gradient.

    H[:,d] = (g(x + eps*e_d) - g(x)) / eps   for d = 0, 1, 2.
    """
    b_vec = get_b_vec(Dm_inv, local_idx)

    # Reference gradient
    F0 = compute_deformation_gradient(x_elem, Dm_inv)
    P0 = total_pk1(F0, d0, mu, kappa, sigma0, activation, scale, e0)
    g0 = vertex_gradient_from_pk1(P0, volume, b_vec)

    H = np.zeros((3, 3))
    for d in range(3):
        x_pert = x_elem.copy()
        x_pert[local_idx, d] += eps
        F_pert = compute_deformation_gradient(x_pert, Dm_inv)
        P_pert = total_pk1(F_pert, d0, mu, kappa, sigma0, activation,
                           scale, e0)
        g_pert = vertex_gradient_from_pk1(P_pert, volume, b_vec)
        col = (g_pert - g0) / eps
        if np.all(np.isfinite(col)):
            H[:, d] = col

    return H


def project_spd(H):
    """Project a 3x3 matrix to symmetric positive semi-definite."""
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        return np.eye(3) * 1e-6  # fallback for NaN/Inf
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return np.eye(3) * 1e-6
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
