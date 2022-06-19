import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")
from symforce import geo
import sympy 
import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3
import pickle
import os

epsilon = 1e-9

#Create symbolic curve
s = sympy.Symbol('s')
x0 = geo.Pose3.symbolic("x0")
x1 = geo.Pose3.symbolic("x1")

Lambda  = (1.0 - s) * sympy.Array(x0.to_storage()) \
        + s * sympy.Array(x1.to_storage())
Lambda_prime = Lambda.diff(s)

#paramters of surface
r = 1.0

x0a = geo.Pose3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([10,0,0])))
x1a = geo.Pose3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([-10,0,0])))
Lambda_actual = Lambda.subs([x0,x1],[x0a,x1a])
Lambda_prime_actual = Lambda_prime.subs([x0,x1],[x0a,x1a])

#create points and poses
landmarks = []
for u in np.linspace(0,1,30):
        #extract pose
        q = np.asarray(Lambda_actual.subs(s, u),dtype='float')#.evalf()
        p = geo.Pose3.from_storage(q)
        
        #perturb pose only on roll
        tangent_perturbation = np.zeros(6); tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
        p_perturb = p.retract(tangent_perturbation, epsilon = epsilon)
        #calculate landmark in world coordinates
        lm = p_perturb * geo.Vector3(np.array([0,r,0]))        
        #store
        landmarks.append(lm)

landmarks = np.array(landmarks)

fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-10,10), ylim = (-5,5), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))
ax.scatter3D(landmarks[:,0], landmarks[:,1], landmarks[:,2])

plotPose3(ax,x0a)
plotPose3(ax,x1a)

plt.show()

#save map of features
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','landmarks.pickle')
file = open(filename, 'wb')
pickle.dump(landmarks,file)
file.close()

