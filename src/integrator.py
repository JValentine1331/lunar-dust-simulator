# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

import numpy as np


def rk4_step(derivative_func, state, t, dt, params):
    """
    Advances the system forward one timestep
    using 4th-order Runge-Kutta integration.

    Parameters
        derivative_func : (function): Function that computes derivatives of the system state.
        state : (np.ndarray): Current system state vector.
        t : (float): Current simulation time.
        dt : (float): Simulation timestep.
        params : (dict): Dictionary containing simulation parameters.

    Returns
        np.ndarray: Updated system state vector after one RK4 step.
    """

    # First slope estimate
    k1 = dt * derivative_func(state, t, params)

    # Second slope estimate
    k2 = dt * derivative_func(state + 0.5 * k1, t + 0.5 * dt, params)

    # Third slope estimate
    k3 = dt * derivative_func(state + 0.5 * k2, t + 0.5 * dt, params)

    # Fourth slope estimate
    k4 = dt * derivative_func(state + k3, t + dt, params)

    # Weighted RK4 update
    new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return new_state