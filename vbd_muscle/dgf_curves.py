"""DeGrooteFregly2016 muscle curve functions and analytical derivatives.

Aligned with OpenSim DeGrooteFregly2016Muscle.h implementation.
Constants from https://simtk.org/projects/optcntrlmuscle (not the paper supplement).

Reference: De Groote et al. (2016), "Evaluation of Direct Collocation Optimal
Control Problem Formulations for Solving the Muscle Redundancy Problem"

Data Sources:
https://github.com/opensim-org/opensim-core/blob/aa40605bae4ecb81c0bae23638f2cd90f20202e5/OpenSim/Actuators/DeGrooteFregly2016Muscle.h#L950
https://static-content.springer.com/esm/art%3A10.1007%2Fs10439-016-1591-9/MediaObjects/10439_2016_1591_MOESM1_ESM.pdf
"""

import numpy as np

# ---------------------------------------------------------------------------
# Active force-length curve parameters (3 Gaussian-like terms)
# ---------------------------------------------------------------------------
b11 = 0.8150671134243542
b21 = 1.055033428970575
b31 = 0.162384573599574
b41 = 0.063303448465465

b12 = 0.433004984392647
b22 = 0.716775413397760
b32 = -0.029947116970696
b42 = 0.200356847296188

b13 = 0.1
b23 = 1.0
b33 = 0.353553390593274  # 0.5 * sqrt(0.5)
b43 = 0.0

# ---------------------------------------------------------------------------
# Passive force-length curve parameters
# ---------------------------------------------------------------------------
kPE = 4.0

# ---------------------------------------------------------------------------
# Force-velocity curve parameters
# ---------------------------------------------------------------------------
d1 = -0.3211346127989808
d2 = -8.149
d3 = -0.374
d4 = 0.8825327733249912


# ---------------------------------------------------------------------------
# Helpers (match calcGaussianLikeCurve / calcGaussianLikeCurveDerivative)
# ---------------------------------------------------------------------------

def _gaussian_like(x, b1, b2, b3, b4):
    """Gaussian-like function used in the active force-length curve.

    Note: the supplement has a typo — the denominator should be squared.
    """
    return b1 * np.exp(-0.5 * (x - b2) ** 2 / (b3 + b4 * x) ** 2)


def _gaussian_like_deriv(x, b1, b2, b3, b4):
    """Derivative of the Gaussian-like function with respect to x."""
    s = b3 + b4 * x
    return (b1 * np.exp(-(b2 - x) ** 2 / (2.0 * s ** 2))
            * (b2 - x) * (b3 + b2 * b4)) / s ** 3


# ---------------------------------------------------------------------------
# Active force-length
# ---------------------------------------------------------------------------

def active_force_length(l_norm, scale=1.0):
    """Active force-length multiplier f_L(l~).

    Args:
        l_norm: Normalized fiber length (l / l_opt).
        scale: Active force width scale (>= 1.0).
    """
    l_norm = np.asarray(l_norm, dtype=float)
    x = (l_norm - 1.0) / scale + 1.0
    return (_gaussian_like(x, b11, b21, b31, b41)
            + _gaussian_like(x, b12, b22, b32, b42)
            + _gaussian_like(x, b13, b23, b33, b43))


def active_force_length_deriv(l_norm, scale=1.0):
    """Derivative df_L / dl~."""
    l_norm = np.asarray(l_norm, dtype=float)
    x = (l_norm - 1.0) / scale + 1.0
    return (1.0 / scale) * (
        _gaussian_like_deriv(x, b11, b21, b31, b41)
        + _gaussian_like_deriv(x, b12, b22, b32, b42)
        + _gaussian_like_deriv(x, b13, b23, b33, b43))


# ---------------------------------------------------------------------------
# Passive force-length
# ---------------------------------------------------------------------------

def passive_force_length(l_norm, e0=0.6, min_norm_fiber_length=0.2):
    """Passive force-length multiplier f_PE(l~).

    Matches OpenSim: curve passes through zero at min_norm_fiber_length,
    so it stays non-negative over the allowed fiber length range.

    Args:
        l_norm: Normalized fiber length.
        e0: Passive fiber strain at one normalized force (default 0.6).
        min_norm_fiber_length: Minimum normalized fiber length (default 0.2).
    """
    l_norm = np.asarray(l_norm, dtype=float)
    offset = np.exp(kPE * (min_norm_fiber_length - 1.0) / e0)
    denom = np.exp(kPE) - offset
    exponent = np.clip(kPE * (l_norm - 1.0) / e0, -50.0, 50.0)
    return (np.exp(exponent) - offset) / denom


def passive_force_length_deriv(l_norm, e0=0.6, min_norm_fiber_length=0.2):
    """Derivative df_PE / dl~."""
    l_norm = np.asarray(l_norm, dtype=float)
    offset = np.exp(kPE * (min_norm_fiber_length - 1.0) / e0)
    denom = np.exp(kPE) - offset
    exponent = np.clip(kPE * (l_norm - 1.0) / e0, -50.0, 50.0)
    return (kPE / e0) * np.exp(exponent) / denom


# ---------------------------------------------------------------------------
# Force-velocity
# ---------------------------------------------------------------------------

def force_velocity(v_norm):
    """Force-velocity multiplier f_V(v~).

    Args:
        v_norm: Normalized contraction velocity (v / V_max), in [-1, 1].
            Negative = shortening (concentric), positive = lengthening (eccentric).
    """
    v_norm = np.asarray(v_norm, dtype=float)
    inner = d2 * v_norm + d3
    return d1 * np.log(inner + np.sqrt(inner ** 2 + 1.0)) + d4


def force_velocity_inverse(fv):
    """Inverse of force-velocity curve: v~ = f_V^{-1}(f_V)."""
    fv = np.asarray(fv, dtype=float)
    return (np.sinh((fv - d4) / d1) - d3) / d2
