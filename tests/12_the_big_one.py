#------------ SIMULATE AND CREATE MESAUREMENTS ------------
import symforce
symforce.set_epsilon_to_number()
from symforce import geo
import numpy as np
import matplotlib.pyplot as plt
import tunnelslam.plotting as plotting

TUNNEL_RADIUS = 1.0
DT = 0.5 #constant time between keyframes
T_final = 30.0
twist = geo.V6([0.0,
                0.0,
                2 * np.pi/15, #full circle in 20 seconds
                2.0,
                0.0,
                0.1]) #[omega, linear velocity]

p = geo.Pose3()

#build tunnel and extract key poses - simulatation
history = {"gt_p": [p],
           "gt_l": [],
            "z" : [],
            "u" : []}
time_from_last_keypose = 0.0
for t in np.arange(0,T_final,DT/5):
    #move robot
    p = p * geo.Pose3.from_tangent(twist * DT)

    #perturb pose only on roll
    tangent_perturbation = np.zeros(6); 
    tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
    p_perturb = p.retract(tangent_perturbation)
    #calculate landmark in world coordinates
    lm = p_perturb * geo.Vector3(np.array([0,TUNNEL_RADIUS,0]))        
    
    #store
    history["gt_l"].append(lm)
    if time_from_last_keypose > DT:
        history["gt_p"].append(p)
        time_from_last_keypose = 0.0
    else:
        time_from_last_keypose += DT

#pass through tunnel - collect measurements

#plot 
fig = plt.figure()
ax = plt.axes(projection='3d', aspect='equal',
        xlim = (-10,10), ylim = (-5,15), zlim = (0,20),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
plotting.axis_equal(ax)
for p in history["gt_p"]:
    plotting.plotPose3(ax,p)
gt_landmarks = np.array(history["gt_l"]).astype(float)
ax.scatter(gt_landmarks[:,0], gt_landmarks[:,1], gt_landmarks[:,2])
plt.show()