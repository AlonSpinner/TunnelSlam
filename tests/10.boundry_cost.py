import numpy as np
import matplotlib.pyplot as plt
from symforce import geo
from symforce import typing as T

K = 10
eps = 0.01
lower_lim, upper_lim = 0.0, 1.0
def boundry_factor(
    s : T.Scalar) -> geo.V1:
    if s < lower_lim:
        return geo.V1(K * abs(s - lower_lim))
    elif s > upper_lim:
        return geo.V1(K * abs(s - upper_lim))
    else:
        return geo.V1(0.0)

s = np.linspace(-0.2,1.2,10000)
f = np.array([boundry_factor(s_i) for s_i in s])
plt.plot(s,f)
plt.show()



