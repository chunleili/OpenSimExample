"""Activation dynamics for muscle excitation -> activation conversion.

Uses DGF smooth tanh-based time constant switching.
"""

import numpy as np


def activation_dynamics(excitation, activation, dt,
                        tau_a=0.015, tau_d=0.060, smoothing=10.0):
    """Advance activation by one time step (implicit Euler).

    da/dt = (e - a) / tau(e, a)
    tau = tau_a + (tau_d - tau_a) * 0.5 * (1 - tanh(smoothing * (e - a)))

    Args:
        excitation: Neural excitation signal, scalar or array in [0, 1].
        activation: Current activation, same shape as excitation.
        dt: Time step (s).
        tau_a: Activation time constant (s), default 0.015.
        tau_d: Deactivation time constant (s), default 0.060.
        smoothing: Tanh smoothing factor for switching.
    Returns:
        New activation value(s), clamped to [0, 1].
    """
    e = np.asarray(excitation, dtype=float)
    a = np.asarray(activation, dtype=float)

    # Smooth time constant
    sigma = 0.5 * (1.0 + np.tanh(smoothing * (e - a)))
    tau = tau_a * sigma + tau_d * (1.0 - sigma)

    # Implicit Euler: a_new = (a + dt*e/tau) / (1 + dt/tau)
    a_new = (a + dt * e / tau) / (1.0 + dt / tau)
    return np.clip(a_new, 0.0, 1.0)
