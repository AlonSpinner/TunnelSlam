from tunnelslam.factors import measurement_residual,radial, odometry_residual
import numpy as np
from symforce import geo, sympy
from symforce import typing as T

x = geo.Pose3_SE3.symbolic("x")
lm = geo.V3.symbolic("lm")
r = sympy.Symbol("r")
cost = radial(x,lm,r)

x = geo.Pose3_SE3.symbolic("x")
lm = geo.V3.symbolic("lm")
z = geo.V3(np.ones(3))
info = geo.Matrix33(np.eye(3))
cost = measurement_residual(x,lm,z,info)

x1 = geo.Pose3_SE3.symbolic("x1")
x2 = geo.Pose3_SE3.symbolic("x2")
meas = geo.Pose3_SE3.from_storage(np.ones(7)) #some constant
diagonal_sigmas = geo.V6(np.ones(6)) #some constant
cost = odometry_residual(x1,x2,meas, diagonal_sigmas, epsilon = sympy.default_epsilon)
