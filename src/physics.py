# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

import numpy as np

#Physical Constants

lunar_g = 1.62 #m / s^2

#Physical interaction functions

def g():
    """
    Returns lunar gravitational acceleration vector.
    Negative y-direction represents downward motion.
    """
    return np.array([0.0, -lunar_g])

def electrostatic_a(charge, mass, efield):
    """
    Computes acceleration due to electrostatic force.
    Parameters:
       charge (float): Particle charge in Coulombs
       mass (float): Particle mass in kg
       electric_field (np.ndarray): Electric field vector [Ex, Ey]

    Returns:
       np.ndarray: Electrostatic acceleration vector
    """
    return (charge / mass) * efield

def total_a(charge, mass, efield):
    """
    Computes total particle acceleration from
    gravity + electrostatic effects.
    """
    g_a = g()
    e_a = electrostatic_a(charge, mass, efield)
    
    return g_a + e_a

def grav_and_lorentz_force(charge, mass, efield):
    """
    Computes the total force from gravitational and electric components
    Parameters:
        mass : (float): mass of particle
        charge : (float): Particle charge in Coulombs
        efield : (np.ndarray): Electric field vector [Ex, Ey]

    Returns:
        (np.ndarray): Total force vector [Fx, Fy]
    """
    
    gf = mass * g()
    ef = charge * efield
    return gf + ef

