import re
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
gt_s = np.linspace(0,1,N)
z, spline_gt = np.zeros((N,3)), np.zeros((N,3))
for i, si in enumerate(gt_s):
    spline_gt[i] = f_spline(si)
    z[i] = f_spline(si) + np.random.normal(0.0,std,3)

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

s_init = [np.random.uniform(0.0 ,1.0) for _ in range(len(z))]
values["s"] = s_init 

# -----------------------------------------------------------------------------
# Create Factors from values
# -----------------------------------------------------------------------------
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

# K = 1.5
# eps = 1e-6
# lower_lim, upper_lim = 0.0, 1.0
# def boundry_factor(
#     s : T.Scalar) -> geo.V1:
#     if s < lower_lim + eps:
#         return geo.V1(K * abs(s - lower_lim))
#     elif s > upper_lim - eps:
#         return geo.V1(K * abs(s - upper_lim))
#     else:
#         return geo.V1(0.0)

K = 0.05
eps = 1e-2
lower_lim, upper_lim = 0.0, 1.0
def boundry_factor(
    s : T.Scalar) -> geo.V1:
    dmin = abs(lower_lim - s)
    dmax = abs(upper_lim-s)
    e = K*(1/(dmax+eps) * 1/(dmin+eps))
    return geo.V1(e)

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

#boundries for s
for i in range(len(s_init)):
    factors.append(
            Factor(residual = boundry_factor,
            keys = [
                    f"s[{i}]"
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

opt_s = np.array(optVals["s"])
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

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(gt_s)
ax.plot(opt_s)
ax.legend(["s -  ground truth","s - optimized"])

plt.show()