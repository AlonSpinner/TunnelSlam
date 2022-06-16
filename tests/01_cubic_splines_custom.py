import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.CubicSplineCalculator import CubicSplineCalculator

x = [0, 2 ,3 ,4]
y = [0, 2, 3, 0]
spline = CubicSplineCalculator(x,y)

t = np.linspace(0,4,100)
f = [spline.interpolation(ti) for ti in t]
plt.plot(t,f)
plt.scatter(x,y,c = 'red')
plt.show()
