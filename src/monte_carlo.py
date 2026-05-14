# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

import numpy as np

def particle_population_generator(
        num_particles,
        speed_avg,
        speed_std,
        angle_min_deg,
        angle_max_deg,
        charge_avg,
        charge_std,
        mass,
        seed=None):
    
    """
    Generates a Monte Carlo population of charged dust particles.

    Parameters
        num_particles : (int): Number of particles to generate
        speed_mean : (float): Mean launch speed (m/s)
        speed_std : (float): Standard deviation of launch speed
        angle_min_deg : (float):Minimum launch angle in degrees
        angle_max_deg : (float): Maximum launch angle in degrees
        charge_mean : (float): Mean particle charge (C)
        charge_std : (float): Standard deviation of particle charge
        mass : (float): Particle mass (kg)
        seed : (int or None): Random seed for reproducibility

    Returns
        states : (np.ndarray): Particle state vectors: [x, y, vx, vy]
        charges : (np.ndarray): Particle charge values
        masses : (np.ndarray): Particle mass values
    """

    
    #Random Seed
    if seed is not None:
        np.random.seed(seed)
        
    #Generate Initial speeds and angles
    speeds = np.random.normal(loc=speed_avg, scale=speed_std, size=num_particles)
    
    #Take absolute value to remove negative inital speeds
    speeds = np.abs(speeds)
    
    angles_deg = np.random.uniform(low=angle_min_deg, 
                                   high=angle_max_deg,
                                   size=num_particles)
    
    #convert to radians
    angles_rad = np.radians(angles_deg)
    
    #resolve speeds into velocities
    vx, vy = speeds * np.cos(angles_rad), speeds * np.sin(angles_rad)
    
    #initlize positions
    x0, y0 = np.zeros(num_particles), np.zeros(num_particles)

    #generate charges
    charges = np.random.normal(loc=charge_avg,
                               scale=charge_std,
                               size=num_particles)
    
    #generate masses
    masses = np.full(num_particles, mass)
    
    #State matrix
    states = np.column_stack((x0,
                              y0,
                              vx,
                              vy))
    
    return states, charges, masses