#------------ SIMULATE AND CREATE MESAUREMENTS ------------
import symforce
symforce.set_epsilon_to_number()
from symforce import geo
from symforce import symbolic as sm
import numpy as np
import matplotlib.pyplot as plt
import tunnelslam.plotting as plotting
from symforce.opt.optimizer import Optimizer
from symforce.values import Values
from symforce.opt.factor import Factor
from tunnelslam.factors import cov2sqrtInfo, measurement_residual, odometry_residual, pose3prior_residual

#-------------------------------------------------------------------------------------------
#-------------------------------- PROBLEM PARAMETERS ---------------------------------------
#-------------------------------------------------------------------------------------------

TUNNEL_RADIUS = 1.0
DT = 1.0 #constant time between keyframes
T_final = 75.0

R0 = geo.Rot3.from_yaw_pitch_roll(0,np.radians(10),0)
t0 = geo.V3()
x0 = geo.Pose3_SE3(R = R0, t = t0)

w_world = geo.V3([0.0,0.0,2 * np.pi/15]) #15 seconds for a full rotation
w_robot = x0.inverse() * w_world
v_robot = geo.V3([-3.0,0.0,0.0])

twist = geo.V6()
twist[:3] = w_robot
twist[3:] = v_robot
twist = np.array(twist,dtype = float)
#-------------------------------------------------------------------------------------------
#-------------------------build tunnel and extract ground truth key poses -------------------
#-------------------------------------------------------------------------------------------
history = {"gt_x": [x0],
           "gt_l": [],
            "z" : [],
            "da" : [],
            "u" : []}
time_from_last_keypose = 0.0
DT_TUNNELING = DT/3 
x = x0
for t in np.arange(0,T_final,DT_TUNNELING):
    #move robot
    # x = x * geo.Pose3_SE3.from_tangent(twist * DT_TUNNELING)
    x = x.retract(twist * DT_TUNNELING)

    #perturb pose only on roll
    tangent_perturbation = np.zeros(6); 
    tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
    p_perturb = x.retract(tangent_perturbation)
    #calculate landmark in world coordinates
    lm = p_perturb * geo.Vector3(np.array([0,TUNNEL_RADIUS,0]))        
    
    #store
    history["gt_l"].append(lm)
    if time_from_last_keypose >= DT:
        history["gt_x"].append(x)
        time_from_last_keypose = 0.0
    else:
        time_from_last_keypose += DT_TUNNELING

SENSOR_RANGE = 20.0
SENSOR_FOV_Y = np.radians(30)
SENSOR_FOV_X = np.radians(50)
SENSOR_COV = np.diag([0.1,np.radians(1),np.radians(1)])
ODOM_COV = 1e-3 * np.eye(6) #in [rad,rad,rad,m,m,m]**2
ODOM_COV[0,0] = (twist[0]/5 * DT)**2
ODOM_COV[2,2] = (twist[2]/5 * DT)**2
ODOM_COV[3,3] = (twist[3]/5 * DT)**2
landmarks = np.array(history["gt_l"]).astype(float)
#pass through tunnel - collect measurements
for k, x in enumerate(history["gt_x"]):
        zk = []
        dak = []
        #measure landmarks
        for index,lm in enumerate(landmarks):
                #z  = [r,yaw,pitch] ~ [r,theta,psi]
                rel_lm = x.inverse() * geo.V3(lm)
                r = float(rel_lm.norm())
                theta = float(sm.atan2(rel_lm[1],rel_lm[0])) #yaw #arctan2(y,x)
                psi = float(sm.atan2(rel_lm[2],geo.V2(rel_lm[:2]).norm()))#pitch

                if abs(theta) <= SENSOR_FOV_X \
                    and abs(psi) <= SENSOR_FOV_Y \
                     and r <= SENSOR_RANGE:
                        z = np.random.multivariate_normal(np.array([r,theta,psi]),SENSOR_COV)
                        zk.append(z)
                        dak.append((k,index))
        zk = np.array(zk).astype(float)
        dak = np.array(dak).astype(int)
        history["z"].append(zk)
        history["da"].append(dak)

        noisy_u = np.random.multivariate_normal(twist * DT,ODOM_COV)
        history["u"].append(noisy_u)

#-------------------------------------------------------------------------------------------
#------------------------------------ SOLVE SLAM -------------------------------------------
#-------------------------------------------------------------------------------------------
values = Values()
PRIOR_COV = np.eye(6) *1e-3

values["odom_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(ODOM_COV)))
values["meas_sqrtInfo"] = geo.V3(np.diag(cov2sqrtInfo(SENSOR_COV)))
values["prior_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(PRIOR_COV)))
values["epsilon"] = sm.numeric_epsilon

values["u"] = []
values["x"] = [x0]
values["z"] = []
values["da"] = []
factors = []
for i, (ui,zi,dai) in enumerate(zip(history["u"],history["z"],history["da"])):
    k = i + 1

    values["u"] += [ui]
    values["x"] += [values["x"][-1].retract(ui)]
    values["z"] += [zi]
    values["da"] += [(i, dai)]

    factors.append(
        Factor(residual = odometry_residual,
        keys = [
                f"x[{k-1}]",
                f"x[{k}]",
                f"u[{k-1}]",
                "odom_sqrtInfo",
                "epsilon",
        ]))
    for j in range(len(dai)):
        factors.append(
                Factor(residual = measurement_residual,
                keys = [
                        f"x[{dai[j][0]}]",
                        f"l[{dai[j][1]}]",
                        f"z[{i}][{j}]",
                        "meas_sqrtInfo",
                ]))


#build values:

#initalize all variables !!

history["dr_x"] = [x0]
for noisy_u in history["u"]:
    history["dr_x"] += [history["dr_x"][-1].retract(noisy_u)]

values["x"] = history["dr_x"]
values["z"] = history["z"]
# values["l"]
# values["s"]

factors = []
#prior
factors.append(
        Factor(residual = pose3prior_residual,
        keys = [
                f"x[0]",
                "x0",
                "prior_sqrtInfo",
                "epsilon"
        ]))

#-------------------------------------------------------------------------------------------
#----------------------------------------PLOT-----------------------------------------------
#-------------------------------------------------------------------------------------------

fig = plt.figure()
ax = plt.axes(projection='3d', aspect='equal',
        xlim = (-10,10), ylim = (-15,5), zlim = (0,40),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
plotting.axis_equal(ax)
for x in history["gt_x"]:
    plotting.plotPose3(ax,x)
for x in history["dr_x"]:
    plotting.plotPose3(ax,x,color = "green")
gt_landmarks = np.array(history["gt_l"]).astype(float)
ax.scatter(gt_landmarks[:,0], gt_landmarks[:,1], gt_landmarks[:,2])
for i, da in enumerate(history["da"]):
    for j, d in enumerate(da):
        ax.plot([history["gt_x"][i].t[0],gt_landmarks[d,0]],
                [history["gt_x"][i].t[1],gt_landmarks[d,1]],
                [history["gt_x"][i].t[2],gt_landmarks[d,2]],
                color = 'k', linewidth = 0.5, linestyle = '-')
plt.show()