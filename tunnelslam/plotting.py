import matplotlib.pyplot as plt
import numpy as np
import symforce.geo as geo

def plotPose3(ax : plt.Axes, p : geo.Pose3, color = 'r'):
        u = p.R.to_rotation_matrix()[:,0]
        t = p.t
        graphics = ax.quiver(t[0],t[1],t[2],u[0],u[1],u[2], color = color)
        return graphics

def axis_equal(ax : plt.Axes):
        bounds = np.array([np.diff(ax.get_xlim()),
                                np.diff(ax.get_ylim()),
                                np.diff(ax.get_zlim())]).flatten()
        ax.set_box_aspect(bounds) #this produces