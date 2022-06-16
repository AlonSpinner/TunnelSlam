import numpy as np
import matplotlib.pyplot as plt

f = [0, 2, 3, 0]

Lambda = lambda s:(1-s)**3 * f[0] \
        + 3*s*(1-s)**2 * s * f[1] \
        + 3*s**2 * (1-s) * f[2] \
        + s**3 * f[3]

t = np.linspace(0,1,100)

plt.plot(t,Lambda(t))
plt.scatter(np.linspace(0,1,4),f, c = 'red')
plt.show()
