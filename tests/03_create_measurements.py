import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")
from symforce import geo
from symforce import sympy as sm
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

odom = geo.Pose3(R=geo.Rot3.identity(), t= geo.V3(np.array([-2,0,0])))
x = geo.Pose3(R=geo.Rot3.identity(), t=geo.V3(np.array([5,0,0])))
odom_cov = 0.1*np.eye(1)
meas_cov = np.diag([1,np.radians(1),np.radians(1)])

# a little confusing here:
#
#           MEASURE      MEASURE
#x0 ->odom-> x1 -> odom ->x2 -> .....
# we have k motions, and k measurement stations, but k+1 poses
# we run on K measurement stations (starting at x1)
K = 5
gt_hist = [[] for k in range(K+1)]; gt_hist[0] = x
meas_odom_hist = [[] for k in range(K)]
meas_lm_hist = [[] for k in range(K)]
for k in range(K):
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
                #z  = [r,yaw,pitch] ~ [r,theta,psi]
                rel_lm = x.inverse() * geo.V3(lm)
                r = rel_lm.norm()
                theta = sm.atan2(rel_lm[1],rel_lm[0]) #yaw #arctan2(y,x)
                psi = sm.atan2(rel_lm[2],geo.V2(rel_lm[:2]).norm()) #pitch

                if -np.pi/2 <= theta <= np.pi/2 and r < 5.0:
                        z = np.random.multivariate_normal(np.array([r,theta,psi],dtype='float'),meas_cov)
                        zk_values.append(z)
                        zk_indexes.append(np.array([k+1,index])) #K+1 poses

                        #help with: https://mathworld.wolfram.com/SphericalCoordinates.html
                        rel_lm_x = z[0] * np.cos(z[2]) * np.cos(z[1])
                        rel_lm_y = z[0] * np.cos(z[2]) * np.sin(z[1])
                        rel_lm_z = z[0] * np.sin(z[2])
                        zk_projections.append(np.array([rel_lm_x,rel_lm_y,rel_lm_z]))

        meas_lm_hist[k] = ({"values": zk_values, "projections": zk_projections, "indices": zk_indexes})
        gt_hist[k+1] = x #K+1 poses


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

dr_x = gt_hist[0].retract([0,0,0,0,0,0.1],epsilon) #move first pose just slightly so we can see difference in this 3D -> 1D problem
dr_graphics = plotPose3(ax,dr_x,'gray')
#dead reckoning + projections
for k, o in enumerate(meas_odom_hist):
        dr_x = dr_x.compose(o)
        
        dr_graphics = plotPose3(ax,dr_x,'gray')

        for rel_lm in np.array(meas_lm_hist[k]["projections"]):
                lm = np.asarray(dr_x * geo.V3(rel_lm),dtype = "float")
                ax.scatter3D(lm[0],lm[1],lm[2], c = 'gray', marker = 'x')

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



