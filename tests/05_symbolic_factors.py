from tunnelslam.factors import measurement_residual,radial_residual, odometry_residual
import numpy as np
from symforce import geo, sympy
from symforce import typing as T

x = geo.Pose3_SE3.symbolic("x")
lm = geo.V3.symbolic("lm")
r = sympy.Symbol("r")
cost = radial_residual(x,lm,r)

x = geo.Pose3_SE3.symbolic("x")
lm = geo.V3.symbolic("lm")
z = geo.V3(np.ones(3))
sqrtInfo = geo.V3(np.ones(3))
cost = measurement_residual(x,lm,z,sqrtInfo)

x1 = geo.Pose3_SE3.symbolic("x1")
x2 = geo.Pose3_SE3.symbolic("x2")
meas = geo.Pose3_SE3.identity() #some constant
sqrtInfo = geo.V6(np.ones(6))
cost = odometry_residual(x1,x2,meas, sqrtInfo, epsilon = sympy.default_epsilon)
