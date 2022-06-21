from symforce import sympy as sm
from symforce import geo

'''
goal here is to create a semi factor showing what we want to do

f(s,x1,x2,lm,r) -> residual
s - running parameter
xi - poses
lm - landmark
r - tunnel radius

we want to build a hermite cubic spline from 2 poses to interpolate over the storage space.
then we want to estimate the distance between the landmark and the spline, call it r_hat
residual = r - rhat

to construct the spline, we will use hermite cubic splines: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
they are basicaly the same as ferguson curves
slopes will be the poses x directions.

below we empoly a toy problem
'''
s = sm.Symbol("s")
lm = geo.V3.symbolic("lm")
x0 = geo.Pose3.symbolic("x0")
x1 = geo.Pose3.symbolic("x1")
r =sm.Symbol("r")

#lets assume m derivatives were calculated for now...
p0 = x0.t
m0 = x0.R.to_rotation_matrix()[:,0]
p1 = x1.t
m1 = x1.R.to_rotation_matrix()[:,0]

#lets create a curve with some running parameter u
Lambda = (2*s**3-3*s**2+1) * p0 \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * p1 \
        + (s**3 - s**2) * m1

rhat = (Lambda-lm).norm()
res = rhat - r

print(res)
