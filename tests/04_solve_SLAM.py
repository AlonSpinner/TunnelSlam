import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3
import pickle
import os

import symforce
from symforce import geo
from symforce import logger
from symforce import sympy as sm
from symforce import typing as T
if symforce.get_backend() != "symengine":
    logger.warning("The 3D Localization example is very slow on the sympy backend")

from symforce.opt.optimizer import Optimizer
from symforce.values import Values

np.random.seed(1)

# -----------------------------------------------------------------------------
# Load measurements
# -----------------------------------------------------------------------------
#load lm measurements 
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','meas_lm_hist.pickle')
file = open(filename, 'rb')
meas_lm_hist = pickle.load(file)
file.close()

#load odom measurements
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','meas_odom_hist.pickle')
file = open(filename, 'rb')
meas_odom_hist = pickle.load(file)
meas_odom_hist = [geo.Pose3_SE3.from_storage(o) for o in meas_odom_hist]
file.close()

# -----------------------------------------------------------------------------
# Build Values
# -----------------------------------------------------------------------------
values = Values()

#initial guesses for poses from dead reckoning
x_initial = [[] for _ in range(len(meas_odom_hist)+1)]
dr_x = geo.Pose3_SE3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([5,0,0])))
x_initial[0] = dr_x
for k,o in enumerate(meas_odom_hist):
        dr_x = dr_x.compose(o)
        x_initial[k+1] = dr_x
values["x"] = x_initial

#initial gusses for landmarks from dead reckoning
# for k, meas_lm_k in enumerate(meas_lm_hist):
#         x_initial[k] * meas_lm_hist
# for meas_k in meas_lm_hist:
#         combined.extend(meas_k["indexes"])
# [np.asarray(_['indexes'],dtype='int') for _ in meas_lm_hist]
#values["l"] = 
values["odom_cov"] = 0.1*np.eye(1)
values["meas_cov"] = np.diag([0.1,np.radians(1),np.radians(1)])
values["epsilon"] = sm.default_epsilon

# -----------------------------------------------------------------------------
# Create Factors
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Instantiate, optimize
# -----------------------------------------------------------------------------