from symforce import sympy as sm
from symforce import geo
import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3

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
# -----------------------------------------------------------------------------
# symbolic formulation
# -----------------------------------------------------------------------------
s = sm.Symbol("s")
lm = geo.V3.symbolic("lm")
x1 = geo.Pose3.symbolic("x1")
x2 = geo.Pose3.symbolic("x2")
r =sm.Symbol("r")

#lets assume m derivatives were calculated for now...
p0 = x1.t
m0 = x1.R.to_rotation_matrix()[:,0]
p1 = x2.t
m1 = x2.R.to_rotation_matrix()[:,0]

#lets create a curve with some running parameter u
splinePoint = (2*s**3-3*s**2+1) * p0 \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * p1 \
        + (s**3 - s**2) * m1

rhat = (lm-splinePoint).norm()
e = rhat - r

#problem to think about:
#How do we decide to which of the segments x1-x2 should we optimize?
#We can't create a long segment chain as the factor is defined over finite set of variables
#We can introduce discrete variables, or check on N segments and take the minimal residual

# -----------------------------------------------------------------------------
# lets draw a spline
# -----------------------------------------------------------------------------

#lets plug in some numbers and plot
x1_a = geo.Pose3.identity()
x2_a = geo.Pose3(R = geo.Rot3(geo.Quaternion(geo.V3([0,-np.sin(np.pi/4),0]),np.cos(np.pi/4))), t = geo.V3(10,10,5))
p0_a = x1_a.t
m0_a = x1_a.R.to_rotation_matrix()[:,0]
p1_a = x2_a.t
m1_a = x2_a.R.to_rotation_matrix()[:,0]

splinePoint_a = (2*s**3-3*s**2+1) * p0_a \
        + (s**3 - 2*s**2 + s) * m0_a * 10\
        + (-2*s**3 + 3*s**2) * p1_a \
        + (s**3 - s**2) * m1_a* 10

t = np.linspace(0,1,100)
si = []
for ti in t:
        si.append(splinePoint_a.subs(s,ti))
curvePoints = np.asarray(si,dtype = "float")

#plot 
fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-5,15), ylim = (-5,15), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))
ax.scatter3D(curvePoints[:,0], curvePoints[:,1], curvePoints[:,2])
plotPose3(ax,x1_a)
plotPose3(ax,x2_a)
plt.show()



