from symforce import geo
from symforce import typing as T
from symforce import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3

import symforce
from symforce import logger
if symforce.get_backend() != "symengine":
    logger.warning("The 3D Localization example is very slow on the sympy backend")

from symforce.opt.optimizer import Optimizer
from symforce.values import Values
from symforce.opt.factor import Factor


#Create data:

x1 = geo.Pose3.identity()
x2 = geo.Pose3(R = geo.Rot3(geo.Quaternion(geo.V3([0,-np.sin(np.pi/4),0]),np.cos(np.pi/4))), t = geo.V3(10,10,5))
p0 = x1.t
m0 = x1.R.to_rotation_matrix()[:,0]
p1 = x2.t
m1 = x2.R.to_rotation_matrix()[:,0]

f_spline = lambda s: (2*s**3-3*s**2+1) * p0 \
        + (s**3 - 2*s**2 + s) * m0 * 10\
        + (-2*s**3 + 3*s**2) * p1 \
        + (s**3 - s**2) * m1* 10

N = 100 #amount of points on curve
std = 0.4
t = np.linspace(0,1,N)
z, spline_gt = np.zeros((N,3)), np.zeros((N,3))
for i, ti in enumerate(t):
    spline_gt[i] = f_spline(ti)
    z[i] = f_spline(ti) + np.random.normal(0.0,std,3)

def measurement_residual(
    s : T.Scalar,
    p0: geo.V3,
    m0: geo.V3,
    p1: geo.V3,
    m1: geo.V3,
    z: geo.V3, 
    sqrtInfo: geo.V3 #diagonal of sqrt information matrix
    ) -> geo.V3:
    splinePoint = (2*s**3-3*s**2+1) * p0 \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * p1 \
        + (s**3 - s**2) * m1

    e = z-splinePoint
    return geo.M.diag(sqrtInfo) * e

# -----------------------------------------------------------------------------
# Build Values
# -----------------------------------------------------------------------------
values = Values()
values["p0"] = p0
values["m0"] = m0
values["p1"] = p1
values["m1"] = m1
values["z"] = [zi for zi in z]
values["meas_sqrtInfo"] = geo.V3(std**2 * np.ones(3))
values["s"] = [np.random.uniform(0.0 ,1.0) for _ in range(len(z))]

# -----------------------------------------------------------------------------
# Create Factors from values
# -----------------------------------------------------------------------------
factors = []

# measurements
for i in range(len(z)):
    factors.append(
            Factor(residual = measurement_residual,
            keys = [
                    f"s[{i}]",
                    "p0",
                    "m0",
                    "p1",
                    "m1",
                    f"z[{i}]",
                    "meas_sqrtInfo",
            ]))
# -----------------------------------------------------------------------------
# optimize
# -----------------------------------------------------------------------------
optimized_keys = [f"s[{i}]" for i in range(len(z))]
optimizer = Optimizer(
factors=factors,
optimized_keys=optimized_keys,
# Return problem stats for every iteration
debug_stats=True,
# Customize optimizer behavior
params=Optimizer.Params(verbose=True, enable_bold_updates = False)
)
result = optimizer.optimize(values)
optVals = result.optimized_values

opt_s = np.clip(np.array(optVals["s"]),0,1)
spline_opt = np.zeros((len(opt_s),3))
for i, s in enumerate(opt_s):
    spline_opt[i] = f_spline(s)

#plot 
fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-5,15), ylim = (-5,15), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))
ax.scatter3D(spline_gt[:,0], spline_gt[:,1], spline_gt[:,2])
ax.scatter3D(z[:,0], z[:,1], z[:,2])
ax.scatter3D(spline_opt[:,0], spline_opt[:,1], spline_opt[:,2])
plotPose3(ax,x1)
plotPose3(ax,x2)
plt.show()