import symforce
symforce.set_epsilon_to_number()

from symforce import geo
from symforce import typing as T
from symforce import symbolic as sm
import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3

from symforce import logger

from symforce.opt.optimizer import Optimizer
from symforce.values import Values
from symforce.opt.factor import Factor

#Create data:

p0 = geo.Pose3.identity()
p1 = geo.Pose3(R = geo.Rot3(geo.Quaternion(geo.V3([0,-np.sin(np.pi/4),0]),np.cos(np.pi/4))), t = geo.V3(10,10,5))
m0 = p0.R.to_rotation_matrix()[:,0]
m1 = p1.R.to_rotation_matrix()[:,0]

f_spline = lambda s: (2*s**3-3*s**2+1) * p0.t \
        + (s**3 - 2*s**2 + s) * m0 * 10\
        + (-2*s**3 + 3*s**2) * p1.t \
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
values["p1"] = p1
values["z"] = [zi for zi in z]
values["meas_sqrtInfo"] = geo.V1(1/(3*std**2))

s_init = [np.random.uniform(0.0 ,1.0) for _ in range(len(z))]
values["s"] = s_init 

# -----------------------------------------------------------------------------
# Create Factors from values
# -----------------------------------------------------------------------------
DT = 1.0 #constant 
def measurement_residual(
    s : T.Scalar,
    p0: geo.Pose3,
    p1: geo.Pose3,
    z: geo.V3, 
    sqrtInfo: geo.V1 #diagonal of sqrt information matrix
    ) -> geo.V1:

    twist = geo.V6((p0.inverse() * p1).to_tangent())/DT
    m0 = twist[3:,0]
    m1 = m0 #assume constant velocity

    splinePoint = (2*s**3-3*s**2+1) * p0.t \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * p1.t \
        + (s**3 - s**2) * m1

    e = (z-splinePoint).norm()
    return sqrtInfo * e

K = 10.0
eps = 0.01
lower_lim, upper_lim = 0.0, 1.0
def boundry_factor(
    s : T.Scalar) -> geo.V1:
    if s < lower_lim + eps:
        return geo.V1(K * abs(s - lower_lim))
    elif s > upper_lim - eps:
        return geo.V1(K * abs(s - upper_lim))
    else:
        return geo.V1(0.0)

factors = []

# measurements
for i in range(len(z)):
    factors.append(
            Factor(residual = measurement_residual,
            keys = [
                    f"s[{i}]",
                    "p0",
                    "p1",
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
params=Optimizer.Params(verbose=True,
                        enable_bold_updates = False,
                        iterations = 100,
                        use_diagonal_damping = True,
                        use_unit_damping = True,
                        initial_lambda = 10)
)
result = optimizer.optimize(values)
optVals = result.optimized_values

opt_s = np.array(optVals["s"])
spline_opt = np.zeros((len(opt_s),3))
for i, s in enumerate(opt_s):
    spline_opt[i] = f_spline(s)

# plot 
fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-5,15), ylim = (-5,15), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))
ax.scatter3D(spline_gt[:,0], spline_gt[:,1], spline_gt[:,2])
ax.scatter3D(z[:,0], z[:,1], z[:,2])
ax.scatter3D(spline_opt[:,0], spline_opt[:,1], spline_opt[:,2])
plotPose3(ax,p0)
plotPose3(ax,p1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(gt_s)
ax.plot(opt_s)
ax.plot(s_init)
ax.legend(["s -  ground truth","s - optimized", "s - initial"])

plt.show()