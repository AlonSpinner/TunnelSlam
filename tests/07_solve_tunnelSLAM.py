import symforce
symforce.set_epsilon_to_number()

import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3
from tunnelslam.factors import cov2sqrtInfo, radial_residual, measurement_residual,\
        odometry_residual, pose3prior_residual, pose3prior_residual
import pickle
import os

from symforce import geo
from symforce import logger
from symforce import symbolic as sm
from symforce import typing as T
from symforce.opt.optimizer import Optimizer
from symforce.values import Values
from symforce.opt.factor import Factor

np.random.seed(1)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
#load lm measurements 
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','meas_lm_hist.pickle')
file = open(filename, 'rb')
meas_lm_hist = pickle.load(file)
for meas in meas_lm_hist:
        meas["projections"] = [geo.V3(proj) for proj in meas["projections"]]
file.close()

#load odom measurements
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','meas_odom_hist.pickle')
file = open(filename, 'rb')
meas_odom_hist = pickle.load(file)
meas_odom_hist = [geo.Pose3.from_storage(o) for o in meas_odom_hist]
file.close()

# -----------------------------------------------------------------------------
# Build Values
# -----------------------------------------------------------------------------
values = Values()

#initial guesses for poses from dead reckoning
x_initial = [[] for _ in range(len(meas_odom_hist)+1)]
dr_x = geo.Pose3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([5,0,0])))
x_initial[0] = dr_x
for k,o in enumerate(meas_odom_hist):
        dr_x = dr_x.compose(o)
        x_initial[k+1] = dr_x
values["x"] = x_initial
values["x0"] = x_initial[0]

#initial gusses for landmarks from dead reckoning
measurements_indices = [] #just collect these two terms nicely
measurements_projections = []
for i, zi in enumerate(meas_lm_hist):
        measurements_indices.extend(zi["indices"])
        measurements_projections.extend(zi["projections"])
measurements_indices = np.asarray(measurements_indices)

initial_landmark_guesses = [geo.V3() for _ in range(max(measurements_indices[:,1])+1)]
for id,proj in zip(measurements_indices,measurements_projections):
       initial_landmark_guesses[id[1]] = x_initial[id[0]] * proj #plant a dead reckoning projection in list
values["l"] = initial_landmark_guesses

odom_cov = np.eye(6);  odom_cov[3,3] = 0.01
meas_cov = np.diag([0.1,np.radians(1),np.radians(1)])
prior_cov = np.eye(6);  odom_cov[3,3] = 0.0001
values["r"] = 1.0
values["odom_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(odom_cov)))
values["meas_sqrtInfo"] = geo.V3(np.diag(cov2sqrtInfo(meas_cov)))
values["prior_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(prior_cov)))
values["epsilon"] = sm.numeric_epsilon
values["odom"] = meas_odom_hist

da = [] #data assosication
z = []
for i, zi in enumerate(meas_lm_hist):
        da.append(zi["indices"])
        z.append(zi["values"])
values["z"] = z
values["da"] = da

# -----------------------------------------------------------------------------
# Create Factors from values
# -----------------------------------------------------------------------------
factors = []
#prior
factors.append(
        Factor(residual = pose3prior_residual,
        keys = [
                f"x[0]",
                "x0",
                "prior_sqrtInfo",
                "epsilon"
        ]))

# measurements
for i, dai in enumerate(values["da"]):
        for j in range(len(dai)):
                factors.append(
                        Factor(residual = measurement_residual,
                        keys = [
                                f"x[{dai[j][0]}]",
                                f"l[{dai[j][1]}]",
                                f"z[{i}][{j}]",
                                "meas_sqrtInfo",
                        ]))
                factors.append(
                Factor(residual = radial_residual,
                keys = [
                        f"x[{dai[j][0]}]",
                        f"l[{dai[j][1]}]",
                        "r",
                ]))

# odometrey
for k in range(len(values["odom"])):
        factors.append(
                Factor(residual = odometry_residual,
                keys = [
                        f"x[{k}]",
                        f"x[{k+1}]",
                        f"odom[{k}]",
                        "odom_sqrtInfo",
                        "epsilon",
                ]))
# -----------------------------------------------------------------------------
# optimize
# -----------------------------------------------------------------------------
optimized_keys_x = [f"x[{k}]" for k in range(len(meas_odom_hist)+1)]
#landmarks are a little annoying. find all indicies of lms measured, and then run on the unique set
measurements_indices = []
for dai in (values["da"]):
        measurements_indices.extend(dai)
measurements_indices = np.asarray(measurements_indices)
optimized_keys_lm = [f"l[{k}]" for k in np.unique(measurements_indices[:,1])]
optimized_keys = optimized_keys_x + optimized_keys_lm

optimizer = Optimizer(
factors=factors,
optimized_keys=optimized_keys,
# Return problem stats for every iteration
debug_stats=True,
# Customize optimizer behavior
params=Optimizer.Params(verbose=True, initial_lambda=1e4, lambda_down_factor=1 / 2.0),
)
result = optimizer.optimize(values)

optVals = result.optimized_values

landmarks = np.array(optVals["l"])
poses = [geo.Pose3.from_storage(xi.to_storage()) for xi in optVals["x"]]
# -----------------------------------------------------------------------------
# plot
# -----------------------------------------------------------------------------
fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-15,10), ylim = (-5,5), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))

#optimized values
landmarks = np.array(optVals["l"])
opt_lm_id = np.unique(measurements_indices[:,1])
poses = [geo.Pose3.from_storage(xi.to_storage()) for xi in optVals["x"]]
opt_graphics = ax.scatter3D(landmarks[opt_lm_id,0], landmarks[opt_lm_id,1], landmarks[opt_lm_id,2], \
        c = 'orange', marker = 'd')
for x in poses:
        plotPose3(ax,x,'orange')

#load lm ground truth
#obtain landmarks
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','landmarks.pickle')
file = open(filename, 'rb')
gt_landmarks = pickle.load(file)
file.close()
#load gt_x_hist
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','gt_x_hist.pickle')
file = open(filename, 'rb')
gt_x_hist = pickle.load(file)
gt_x_hist = [geo.Pose3.from_storage(o) for o in gt_x_hist]
file.close()

#add ground truth to plot
for x in gt_x_hist:
        plotPose3(ax,x,'red')
gt_graphics = ax.scatter3D(gt_landmarks[opt_lm_id,0], gt_landmarks[opt_lm_id,1], gt_landmarks[opt_lm_id,2])
#add initial landmark guesses
initial_landmark_guesses = np.asarray([lm.to_numpy() for lm in initial_landmark_guesses])
initial_graphics = ax.scatter3D(initial_landmark_guesses[opt_lm_id,0], initial_landmark_guesses[opt_lm_id,1], \
        initial_landmark_guesses[opt_lm_id,2],c = 'gray', marker = 'x')

ax.legend([opt_graphics, gt_graphics ,initial_graphics],["optimized","ground truth","initial"])
plt.show()