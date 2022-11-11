import symforce
from symforce import geo
from dataclasses import dataclass
import numpy as np
x1 = geo.Pose3(R=geo.Rot3.identity(),
               t=geo.Vector3([1,0,0]))
M = geo.M33(np.array([[0,-1,0]
                      ,[1,0,0],
                       [0,0,1]]))
x2 = geo.Pose3(R=geo.Rot3.from_rotation_matrix(M),
               t=geo.Vector3([4,0,0]))
dt = 1.0
dx = (x1.inverse() * x2).to_tangent()
zeta = geo.V6(dx) / dt
print(zeta.evalf())

X2_test = x1.retract(zeta * dt).evalf().to_homogenous_matrix()
X2 = x2.to_homogenous_matrix()
print((X2_test - X2))