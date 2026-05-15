# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""
import numpy as np
from src.physics import total_a, grav_and_lorentz_force
from src.simulation import particle_derivatives1, particle_derivatives2
from src.integrator import rk4_step
from src.monte_carlo import particle_population_generator
from src.visualization import plot_trajectories, plot_landing_distribution

# Monte Carlo Population Parameters
NUM_PARTICLES = 100

SPEED_MEAN = 5.0 #m/s
SPEED_STD = 1.0 #m/s

ANGLE_MIN = 60 #degrees
ANGLE_MAX = 120 #degress

CHARGE_MEAN = 1e-6 #Coulombs
CHARGE_STD = 2e-7 #Coulombs

EFIELD = np.array([0.0, 500.0]) #N/Coulomb

MASS = 1e-3 #kg

SEED = 4223

#Time Integration Parameters
DT = 0.05
NUM_STEPS = 1000

#Generate Initial Population
states, charges, masses = particle_population_generator(
    num_particles=NUM_PARTICLES,
    speed_avg=SPEED_MEAN,
    speed_std=SPEED_STD,
    angle_min_deg=ANGLE_MIN,
    angle_max_deg=ANGLE_MAX,
    charge_avg=CHARGE_MEAN,
    charge_std=CHARGE_STD,
    mass=MASS,
    seed=SEED)

#Initialize Trajectory Storage

# Shape:
# [particle_index, timestep_index, state_variable]


trajectories = np.zeros((NUM_PARTICLES,
                         NUM_STEPS,
                         4))

#Storing Initial states
trajectories[:, 0, :] = states

# Track whether particles have landed
landed = np.zeros(NUM_PARTICLES, dtype=bool)

#Time evolve the system
for i in range(1, NUM_STEPS):
    
    current_t = i * DT
    
    for particle in range(NUM_PARTICLES):
        
        # Skip particles that already landed
        if landed[particle]:
            trajectories[particle, i, :] = trajectories[particle, i - 1, :]
            continue
        
        current_state = trajectories[particle, i - 1,:]
        
        params = {"charge": charges[particle],
                  "mass": masses[particle],
                  "efield": EFIELD}
        
        next_state = rk4_step(particle_derivatives2, 
                              current_state, 
                              current_t, 
                              DT, 
                              params)
        
        # Surface Collision Condition
        if next_state[1] < 0:
        
            # Clamp particle to surface
            next_state[1] = 0

            # Zero vertical velocity
            next_state[3] = 0
            
            landed[particle] = True
            
        trajectories[particle, i, :] = next_state
        
#Display Simulation Info
print("\nTrajectory Array Shape:")
print(trajectories.shape)

print("\nExample Final Particle State:")
print(trajectories[0, -1, :])

plot_trajectories(trajectories, charges)
plot_landing_distribution(trajectories)