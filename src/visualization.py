# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectories(trajectories, charges):
    """
    Plots particle trajectories in x-y space.

    Parameters
        trajectories : (np.ndarray): 
            Shape:
                [particle, timestep, state_variable]

            where:
                state_variable = [x, y, vx, vy]
    """
    
    num_particles = trajectories.shape[0]
    
    # Normalize charge values for colormap
    norm = plt.Normalize(vmin=np.min(charges), vmax=np.max(charges))
    cmap = plt.cm.plasma
    
    for particle in range(num_particles):
        
        x = trajectories[particle, : , 0]
        y = trajectories[particle, : , 1]
        
        color = cmap(norm(charges[particle]))
        plt.plot(x, y, color=color, alpha=0.7)
    
    plt.xlabel("x-position (m)")
    plt.ylabel("y-position (m)")
    plt.title("Monte Carlo Lunar Dust Trajectories")
    plt.grid()
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Particle Charge (C)")
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)

    # Save figure
    plt.savefig("results/trajectory_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_landing_distribution(trajectories):
    """
    Plots final landing positions of particles.
    """

    final_x = trajectories[:, -1, 0]
    final_y = trajectories[:, -1, 1]

    plt.figure(figsize=(10, 4))

    plt.scatter(final_x, final_y, alpha=0.7)

    plt.axhline(y=0, color='black', linestyle='--')

    plt.xlabel("Final x-position (m)")
    plt.ylabel("Final y-position (m)")
    plt.title("Particle Landing Distribution")
    plt.grid(True)

    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    # Save figure
    plt.savefig("results/landing_distribution.png", dpi=300, bbox_inches='tight')

    plt.show()
    
def histogram(values, xlabel, title, filename, bins=20):
    """
    Plots and saves histogram of simulation metrics.

    Parameters
        values : (np.ndarray): Data values
        xlabel : (str): x-axis label
        title : (str): Plot title
        filename : (str): Output filename
        bins : (int): Number of histogram bin
    """
    #Remove NaN values
    values = values[~np.isnan(values)]
    
    plt.figure(figsize=(8,5))
    plt.hist(values, bins=bins, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    
    #create results directory and store
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}",dpi=300,bbox_inches='tight')
    plt.show()