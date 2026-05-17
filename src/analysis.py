# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:43:38 2026

@author: jessv
"""

import numpy as np

def landing_distance(trajectories):
    """
    Computes final x-position for each particle.

    Parameters
    trajectories : (np.ndarray): Shape: [particle, timestep, state_variable]

    Returns
        (np.ndarray): Final landing x-positions
    """
    
    return trajectories[:,-1,0]

def max_heights(trajectories):
    """
    Computes maximum height reached by each particle.

    Parameters
        trajectories : (np.ndarray)

    Returns
        (np.ndarray): Maximum y-position for each particle
    """
    
    return np.max(trajectories[:,:,1], axis=1)

def airborne_durations(trajectories, dt):
    """
    Estimates airborne duration for each particle.

    Parameters
        trajectories : (np.ndarray)

        dt : (float): Simulation timestep

    Returns
        (np.ndarray): Estimated airborne durations
    """
    
    num_particles = trajectories.shape[0]
    durations = np.full(num_particles, np.nan)
    
    for i in range(num_particles):
        y = trajectories[i,:,1]
        
        #ignore particles that never leave the ground
        if np.max(y) < 0.01:
            durations[i] = np.nan
            continue
        else:
            pass
        
        indices = np.where(y <= 0)[0]
        
        if len(indices > 1):
            landing_step = indices[0] + 1 
            durations[i] = landing_step * dt
            
        else:
            durations[i] = np.nan
            
        return durations
    
def summarize_metric(values, label):
    """
    Prints summary statistics for a metric.

    Parameters
        values : (np.ndarray)

        label : (str): Metric name
    """
   
    print(f"\n--- {label} ---")

    print(f"Mean: {np.nanmean(values):.3f}")
    print(f"Std Dev: {np.nanstd(values):.3f}")
    print(f"Min: {np.nanmin(values):.3f}")
    print(f"Max: {np.nanmax(values):.3f}")
