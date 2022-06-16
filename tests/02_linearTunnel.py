import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")
from symforce import geo
import sympy 
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-9

#Create symbolic curve
s = sympy.Symbol('s')
x0 = geo.Pose3_SE3.symbolic("x0")
x1 = geo.Pose3_SE3.symbolic("x1")

Lambda  = (1.0 - s) * sympy.Array(x0.to_storage()) \
        + s * sympy.Array(x1.to_storage())
Lambda_prime = Lambda.diff(s)


#paramters of surface
r = 1.0
r_sigma = 0.3

x0a = geo.Pose3_SE3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([0,0,0])))
x1a = geo.Pose3_SE3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([10,0,0])))
Lambda_actual = Lambda.subs([x0,x1],[x0a,x1a])
Lambda_prime_actual = Lambda_prime.subs([x0,x1],[x0a,x1a])

#create points and poses
landmarks = []
for u in np.linspace(0,1,100):
        #extract pose
        q = np.asarray(Lambda_actual.subs(s, u),dtype='float')#.evalf()
        p = geo.Pose3_SE3.from_storage(q)
        
        #perturb pose only on roll
        tangent_perturbation = np.zeros(6); tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
        p_perturb = p.retract(tangent_perturbation, epsilon = epsilon)
        #calculate landmark in world coordinates
        lm = p_perturb * geo.Vector3(np.array([0,np.random.uniform(r,r_sigma),0]))        
        #store
        landmarks.append(lm)

landmarks = np.array(landmarks)


def plotPose3(ax : plt.Axes, p : geo.Pose3_SE3, color = 'r'):
        u = p.R.to_rotation_matrix()[:,0]
        t = p.t
        ax.quiver(t[0],t[1],t[2],u[0],u[1],u[2], color = color)

fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-5,15), ylim = (-5,5), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.scatter3D(landmarks[:,0], landmarks[:,1], landmarks[:,2], 'gray')

plotPose3(ax,x0a)
plotPose3(ax,x1a)

plt.show()



