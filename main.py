# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:50:32 2026

@author: jessv
"""

from src.physics import total_a, grav_and_lorentz_force
import numpy as np

electric_field = np.array([0.0, 5.0])

acc = total_a(1e-6, 1e-3, electric_field)
net_F = grav_and_lorentz_force(1e-6, 1e-3, electric_field)

print("Acceleration Vector:")
print(acc)
print(net_F)