# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""
import numpy as np
from src.physics import total_a, grav_and_lorentz_force
from src.simulation import particle_derivatives1, particle_derivatives2
from src.integrator import rk4_step

# Initial Particle State

state = np.array([
    0.0,   # x-position
    0.0,   # y-position
    1.0,   # x-velocity
    5.0    # y-velocity
])

# Simulation Parameters
params = {"charge": 1e-6, "mass": 1e-3, "efield": np.array([0.0, 5.0])}

# Time Settings
t = 0.0
dt = 0.1

# Advance One RK4 Step

print(particle_derivatives1(state, t, params))
print("line")
print(particle_derivatives2(state, t, params))
new_state1 = rk4_step(
    particle_derivatives1,
    state,
    t,
    dt,
    params)

new_state2 = rk4_step(
    particle_derivatives2, 
    state,
    t, 
    dt, 
    params)

# Display Results

print("Updated State Vector:")
print(new_state1, new_state2)