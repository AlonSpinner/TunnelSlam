#------------ SIMULATE AND CREATE MESAUREMENTS ------------
import symforce
symforce.set_epsilon_to_number()
from symforce import geo
import numpy as np
import matplotlib.pyplot as plt
import tunnelslam.plotting as plotting

TUNNEL_RADIUS = 1.0
DT = 1.0 #constant time between keyframes
T_final = 75.0

R0 = geo.Rot3.from_yaw_pitch_roll(0,np.radians(10),0)
t0 = geo.V3()
p = geo.Pose3(R = R0, t = t0)

w_world = geo.V3([0.0,0.0,2 * np.pi/15])
w_robot = p.inverse() * w_world
v_robot = geo.V3([-2.0,0.0,0.0])

twist = geo.V6()
twist[:3] = w_robot
twist[3:] = v_robot

# twist = geo.V6([0.05,
#                 0.1,
#                 2 * np.pi/15, #full circle in 15 seconds
#                 2.0,
#                 0.0,
#                 0.0]) #[omega, linear velocity]

#build tunnel and extract key poses - simulatation
history = {"gt_p": [p],
           "gt_l": [],
            "z" : [],
            "u" : []}
time_from_last_keypose = 0.0
DT_TUNNELING = DT/3 
for t in np.arange(0,T_final,DT_TUNNELING):
    #move robot
    p = p * geo.Pose3.from_tangent(twist * DT_TUNNELING)

    #perturb pose only on roll
    tangent_perturbation = np.zeros(6); 
    tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
    p_perturb = p.retract(tangent_perturbation)
    #calculate landmark in world coordinates
    lm = p_perturb * geo.Vector3(np.array([0,TUNNEL_RADIUS,0]))        
    
    #store
    history["gt_l"].append(lm)
    if time_from_last_keypose >= DT:
        history["gt_p"].append(p)
        time_from_last_keypose = 0.0
    else:
        time_from_last_keypose += DT_TUNNELING

#pass through tunnel - collect measurements

#plot 
fig = plt.figure()
ax = plt.axes(projection='3d', aspect='equal',
        xlim = (-10,10), ylim = (-15,5), zlim = (0,20),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
plotting.axis_equal(ax)
for p in history["gt_p"]:
    plotting.plotPose3(ax,p)
gt_landmarks = np.array(history["gt_l"]).astype(float)
ax.scatter(gt_landmarks[:,0], gt_landmarks[:,1], gt_landmarks[:,2])
plt.show()