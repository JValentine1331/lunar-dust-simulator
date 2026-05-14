# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

import numpy as np

from src.physics import total_a, grav_and_lorentz_force

def particle_derivatives1(state, t, params):
    """
    Computes time derivatives for a charged lunar dust particle.

    State Vector: state = [x, y, vx, vy]

    where:
        x  = x-position
        y  = y-position
        vx = x-velocity
        vy = y-velocity

    Parameters
        state : (np.ndarray): Current particle state vector
        t : (float): Current simulation time

    params : (dict)
        Dictionary containing:
            charge
            mass
            electric_field

    Returns
        np.ndarray: Time derivative of state vector:
        [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """

    # Unpack State Vector
    x, y, vx, vy = state

    # Retrieve Simulation Parameters
    charge = params["charge"]
    mass = params["mass"]
    efield = params["efield"]

    # Compute Acceleration
    acceleration = total_a(charge, mass, efield)
    ax, ay = acceleration

    # Construct Derivative Vector
    derivatives = np.array([
        vx,   # dx/dt
        vy,   # dy/dt
        ax,   # dvx/dt
        ay    # dvy/dt
    ])

    return derivatives

def particle_derivatives2(state, t, params):
    
    # Unpack State Vector
    x, y, vx, vy = state

    # Retrieve Simulation Parameters
    charge = params["charge"]
    mass = params["mass"]
    efield = params["efield"]

    # Compute Acceleration
    force = grav_and_lorentz_force(charge, mass, efield)
    a = force / mass
    ax, ay = a
    
    # Construct Derivative Vector
    derivatives = np.array([
        vx,   # dx/dt
        vy,   # dy/dt
        ax,   # dvx/dt
        ay    # dvy/dt
    ])
    
    return derivatives
