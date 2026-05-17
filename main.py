# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""
import numpy as np
from src.physics import total_a, grav_and_lorentz_force, electric_field
from src.simulation import particle_derivatives1, particle_derivatives2
from src.integrator import rk4_step
from src.monte_carlo import particle_population_generator
from src.visualization import plot_trajectories, plot_landing_distribution, histogram, parameter_study_plot
from src.analysis import landing_distance, max_heights, airborne_durations, summarize_metric

#Monte Carlo Population Parameters
NUM_PARTICLES = 100

SPEED_MEAN = 5.0 #m/s
SPEED_STD = 1.0 #m/s

ANGLE_MIN = 60 #degrees
ANGLE_MAX = 120 #degress

CHARGE_MEAN = 1e-6 #Coulombs
CHARGE_STD = 2e-6 #Coulombs

#E field parameter study
E0_values = [500,1000,1500,2000, 3000]

mean_heights = []
mean_flight_times = []
landed_fractions = []

#E0 = 1500.0 #Coulombs
L = 10.0 #m

MASS = 1e-3 #kg

SEED = 500

#Time Integration Parameters
DT = 0.0005
NUM_STEPS = 10000

for E0 in E0_values:
    print(f"\nRunning Simulation for E0 = {E0}")

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
                      "E0": E0,
                      "L": L}
            
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
        
    #Display Trajectories
    plot_trajectories(trajectories, charges, E0=E0)
    plot_landing_distribution(trajectories, E0=E0)
    
    #Statistical Analysis
    landing_x = landing_distance(trajectories)
    max_height_values = max_heights(trajectories)
    flight_times = airborne_durations(trajectories, DT)
    
    #Parameter Study Metrics
    mean_heights.append(np.nanmean(max_height_values))
    mean_flight_times.append(np.nanmean(flight_times))
    
    landed_fraction = (np.sum(~np.isnan(flight_times)) / NUM_PARTICLES)
    landed_fractions.append(landed_fraction)
        
    summarize_metric(landing_x, "Landing Distance (m)")
    summarize_metric(max_height_values, "Maximum Height (m)")
    summarize_metric(flight_times, "Airborne Duration (s)")
    
    #Display Simulation Info
    print("\nTrajectory Array Shape:")
    print(trajectories.shape)
    
    print("\nExample Final Particle State:")
    print(trajectories[0, -1, :])
    
    print("\nNumber of landed particles:")
    print(np.sum(~np.isnan(flight_times)))
    
    print("\nNumber of non-landed particles:")
    print(np.sum(np.isnan(flight_times)))
    
    print("\nFlight Times:")
    print(flight_times[~np.isnan(flight_times)])
    
    #Histogram Visualizations
    histogram(landing_x, xlabel="Landing Distances (m)",
              title=f'Landing Distance Distribution (E0={E0})', 
              filename=f"landing_distance_histogram_{E0}.png")

    histogram(max_height_values, xlabel="Maximum Height (m)", 
              title=f"Maximum Height Distribution (E0={E0})", 
              filename=f"max_height_histogram_{E0}.png")

    histogram(flight_times, xlabel="Airborne Duration (s)",
              title=f"Airborne Duration Distribution (E0={E0})",
              filename=f"airborne_duration_histogram_{E0}.png")

#Parameter Study Visualizations
parameter_study_plot(E0_values,
                     mean_heights,
                     xlabel="Surface Electric Field E0 (N/C)",
                     ylabel="Mean Maximum Height (m)",
                     title="Mean Maximum Height vs Electric Field Strength", 
                     filename="mean_height_vs_E0.png")

parameter_study_plot(E0_values,
                     mean_flight_times,
                     xlabel="Surface Electric Field E0 (N/C)",
                     ylabel="Mean Airborne Duration (s)",
                     title="Mean Airborne Duration vs Electric Field Strength",
                     filename="flight_time_vs_E0.png")

parameter_study_plot(E0_values,
                     landed_fractions,
                     xlabel="Surface Electric Field E0 (N/C)", 
                     ylabel="Landed Fraction",
                     title="Landed Fraction vs Electric Field Strength",
                     filename="landed_fraction_vs_E0.png")
