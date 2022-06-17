import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")
from symforce import geo
import numpy as np
import matplotlib.pyplot as plt
from tunnelslam.plotting import plotPose3
import pickle
import os

epsilon = 1e-9
np.random.seed(1)

#obtain landmarks
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path,'out','landmarks.pickle')
file = open(filename, 'rb')
landmarks = pickle.load(file)
file.close()

#create odometry measurements

odom = geo.Pose3_SE3(R=geo.Rot3.identity(), t= geo.Vector3(np.array([-2,0,0])))
x = geo.Pose3_SE3(R=geo.Rot3.identity(), t=geo.Vector3(np.array([5,0,0])))
odom_cov = 0.1*np.eye(1) * 0
meas_cov = np.diag([0.1,np.radians(1),np.radians(1)]) * 0

K = 5
gt_hist = [[] for k in range(K)]; gt_hist[0] = x
meas_odom_hist = [[] for k in range(K-1)]
meas_lm_hist = [[] for k in range(K-1)]
for k in range(0,K-1):
        #move
        x = x.compose(odom)
        
        #measure odom
        tangent_perturbation = np.zeros(6); tangent_perturbation[3] = np.random.multivariate_normal(np.zeros(1),odom_cov)
        meas_odom_hist[k] = (odom.retract(tangent_perturbation, epsilon = epsilon))

        zk_values = []
        zk_projections = []
        zk_indexes = []
        #measure landmarks
        for index,lm in enumerate(landmarks):
                rel_lm = np.asarray(x.inverse() * geo.Vector3(lm),dtype = "float")
                r = np.linalg.norm(rel_lm)
                theta = np.arctan2(rel_lm[0],rel_lm[1]) #yaw
                psi = np.arctan2(np.linalg.norm(rel_lm[:2]),rel_lm[2]) #pitch
                if 0 <= theta <= np.pi and r < 4.0:
                        z = np.random.multivariate_normal(np.array([r,theta,psi]),meas_cov)
                        zk_values.append(z)
                        zk_indexes.append(index)

                        #https://en.wikipedia.org/wiki/Spherical_coordinate_system
                        rel_lm_x = z[0] * np.sin(z[2]) * np.cos(z[1])
                        rel_lm_y = z[0] * np.sin(z[2]) * np.sin(z[1])
                        rel_lm_z = z[0] * np.cos(z[2])
                        # zk_projections.append(np.array([rel_lm_x,rel_lm_y,rel_lm_z]))
                        zk_projections.append(rel_lm)

        meas_lm_hist[k] = ({"values": zk_values, "projections": zk_projections, "indexes": zk_indexes})
        gt_hist[k+1] = x


#plot 
fig = plt.figure()
ax = plt.axes(projection='3d',
        xlim = (-15,10), ylim = (-5,5), zlim = (-5,5),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
ax.set_box_aspect(aspect = (1,1,1))
ax.scatter3D(landmarks[:,0], landmarks[:,1], landmarks[:,2])
#ground truth
for x in gt_hist:
        gt_graphics = plotPose3(ax,x)

dr_x = gt_hist[0]
#dead reckoning + projections
for k, o in enumerate(meas_odom_hist):
        dr_graphics = plotPose3(ax,dr_x,'gray')

        dr_x = dr_x.compose(o)

        for rel_lm in np.array(meas_lm_hist[k]["projections"]):
                lm = np.asarray(dr_x * geo.Vector3(rel_lm),dtype = "float")
                ax.scatter3D(lm[0],lm[1],lm[2], c = 'r', s = 50, facecolors = 'none')

ax.legend([gt_graphics,dr_graphics],['ground truth','dead reckoning'])
plt.show()

#save measurements
dir_path = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(dir_path,'out','meas_lm_hist.pickle')
file = open(filename, 'wb')
pickle.dump(meas_lm_hist,file)
file.close()


filename = os.path.join(dir_path,'out','meas_odom_hist.pickle')
file = open(filename, 'wb')
pickle.dump([o.to_storage() for o in meas_odom_hist],file)
file.close()



