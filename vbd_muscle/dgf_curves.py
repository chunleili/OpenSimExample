"""DeGrooteFregly2016 muscle curve functions and analytical derivatives.

Reference: De Groote et al. (2016), "Evaluation of Direct Collocation Optimal
Control Problem Formulations for Solving the Muscle Redundancy Problem"
"""

import numpy as np

# Active force-length: 3 Gaussian terms [b1, b2, b3, b4]
_B = np.array([
    [0.815, 1.055, 0.162, 0.063],
    [0.433, 0.717, -0.030, 0.200],
    [0.1, 1.0, 0.354, 0.0],
])

# Force-velocity constants [d1, d2, d3, d4]
_D = np.array([-0.3211, -8.149, -0.374, 0.8825])


def active_force_length(l_norm, scale=1.0):
    """Active force-length multiplier f_L(l~).

    Args:
        l_norm: Normalized fiber length (l / l_opt).
        scale: Active force width scale (>= 1.0).
    """
    l_norm = np.asarray(l_norm, dtype=float)
    x = (l_norm - 1.0) / scale + 1.0
    total = np.zeros_like(x)
    for i in range(3):
        b1, b2, b3, b4 = _B[i]
        s = b3 + b4 * x
        # Guard against zero denominator
        s = np.where(np.abs(s) < 1e-12, 1e-12, s)
        z = (x - b2) / s
        total += b1 * np.exp(-0.5 * z ** 2)
    return total


def active_force_length_deriv(l_norm, scale=1.0):
    """Derivative df_L / dl~."""
    l_norm = np.asarray(l_norm, dtype=float)
    x = (l_norm - 1.0) / scale + 1.0
    dx_dl = 1.0 / scale
    total = np.zeros_like(x)
    for i in range(3):
        b1, b2, b3, b4 = _B[i]
        s = b3 + b4 * x
        s = np.where(np.abs(s) < 1e-12, 1e-12, s)
        z = (x - b2) / s
        gauss = b1 * np.exp(-0.5 * z ** 2)
        # dz/dx = (b3 + b4*b2) / s^2
        dz_dx = (b3 + b4 * b2) / s ** 2
        total += gauss * (-z) * dz_dx * dx_dl
    return total


def passive_force_length(l_norm, kPE=4.0, e0=0.6):
    """Passive force-length multiplier f_PE(l~).

    Args:
        l_norm: Normalized fiber length.
        kPE: Exponential shape parameter (default 4.0).
        e0: Passive fiber strain at one normalized force (default 0.6).
    """
    l_norm = np.asarray(l_norm, dtype=float)
    exponent = np.clip(kPE * (l_norm - 1.0) / e0, -50.0, 50.0)
    num = np.exp(exponent) - 1.0
    denom = np.exp(kPE) - 1.0
    return num / denom


def passive_force_length_deriv(l_norm, kPE=4.0, e0=0.6):
    """Derivative df_PE / dl~."""
    l_norm = np.asarray(l_norm, dtype=float)
    denom = np.exp(kPE) - 1.0
    exponent = np.clip(kPE * (l_norm - 1.0) / e0, -50.0, 50.0)
    return (kPE / e0) * np.exp(exponent) / denom


def force_velocity(v_norm):
    """Force-velocity multiplier f_V(v~).

    Args:
        v_norm: Normalized contraction velocity (v / V_max), in [-1, 1].
            Negative = shortening (concentric), positive = lengthening (eccentric).
    """
    v_norm = np.asarray(v_norm, dtype=float)
    d1, d2, d3, d4 = _D
    inner = d2 * v_norm + d3
    return d1 * np.log(inner + np.sqrt(inner ** 2 + 1.0)) + d4


def force_velocity_inverse(fv):
    """Inverse of force-velocity curve: v~ = f_V^{-1}(f_V)."""
    fv = np.asarray(fv, dtype=float)
    d1, d2, d3, d4 = _D
    return (np.sinh((fv - d4) / d1) - d3) / d2
