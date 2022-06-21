import sympy
import numpy as np
import matplotlib.pyplot as plt
from symforce import sympy as sm

s = sympy.Symbol('s')
p0 = 0 #starting point is [0,p0]
m0 = 0
p1 = 0 #ending point is [1,p1]
m1 = -5 

Lambda = (2*s**3-3*s**2+1) * p0 \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * p1 \
        + (s**3 - s**2) * m1


f = sympy.lambdify(s, Lambda)
t = np.linspace(0,1,100)
plt.plot(t,f(t))
plt.show()

# Lambda2D = sympy.Array([s,Lambda])
# q = sympy.Array([sympy.Symbol(f"{v}") for v in ["q0", "q1"]])

# #find sqrd norm
# sqrdnorm = (q[0] - Lambda2D[0])** 2 + (q[1] - Lambda2D[1]) ** 2 #couldnt find vector operations.. whatever
# sol = sympy.solve(sqrdnorm.diff(s),s)
# sympy.exp(f"{sqrdnorm.diff(s)} == 0")

