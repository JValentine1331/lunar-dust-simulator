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
    Computes airborne durations using explicit
    launch and landing event detection.

    Parameters
        trajectories : np.ndarray
    
        dt : float
            Simulation timestep

    Returns
    np.ndarray
        Airborne durations
    """
    
    num_particles = trajectories.shape[0]

    durations = np.full(num_particles, np.nan)

    MIN_HEIGHT = 0.01

    for particle_idx in range(num_particles):

        y = trajectories[particle_idx, :, 1]

        launched = False
        launch_step = None
        landing_step = None

        # ---------------------------------
        # Detect launch + landing events
        # ---------------------------------

        for step in range(1, len(y)):

            # Detect first meaningful lofting
            if not launched and y[step] > MIN_HEIGHT:

                launched = True
                launch_step = step

            # Detect landing AFTER launch
            elif launched:

                if y[step - 1] > 0 and y[step] <= 0:

                    landing_step = step
                    break

        # ---------------------------------
        # Compute duration
        # ---------------------------------

        if launch_step is not None and landing_step is not None:

            durations[particle_idx] = (
                landing_step - launch_step
            ) * dt

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
