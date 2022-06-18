from tunnelslam.factors import measurement_residual,radial, odometry_residual
import numpy as np

import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")
from symforce import geo, sympy
from symforce import typing as T

x = geo.Pose3_SE3.symbolic("x")
lm = geo.V3.symbolic("lm")
r = sympy.Symbol("r")
e = radial(x,lm,r)


x1 = geo.Pose3_SE3.symbolic("x1")
x2 = geo.Pose3_SE3.symbolic("x2")
meas = geo.Pose3_SE3.from_storage(np.ones(7)) #some constant
diagonal_sigmas = geo.V6(np.ones(6)) #some constant
e = odometry_residual(x1,x2,meas, diagonal_sigmas, epsilon = sympy.default_epsilon)
