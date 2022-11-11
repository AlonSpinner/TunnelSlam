import numpy as np

def spherical_to_cartesian(z : np.array):
    #z = [r,theta,psi] ~ [range, yaw, pitch]
    x = z[0] * np.cos(z[2]) * np.cos(z[1])
    y = z[0] * np.cos(z[2]) * np.sin(z[1])
    z = z[0] * np.sin(z[2])
    return np.array([x,y,z])