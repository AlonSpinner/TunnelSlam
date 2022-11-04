import numpy as np
import matplotlib.pyplot as plt

K = 0.001
eps = 1e-4
s = np.linspace(-0.2,1.2,100)
dmin = np.abs(0.0 - s)
dmax = np.abs(1.0-s)
plt.plot((-np.log(s+eps))**6)
# plt.plot(K * (1/(dmax+eps) * 1/(dmin+eps)))
plt.show()

