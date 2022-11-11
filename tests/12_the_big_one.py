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
from tunnelslam.factors import cov2sqrtInfo, measurement_residual, \
                               odometry_residual, pose3prior_residual, ferguson_residual, boundry_factor
from tunnelslam.utils import spherical_to_cartesian

#-------------------------------------------------------------------------------------------
#-------------------------------- PROBLEM PARAMETERS ---------------------------------------
#-------------------------------------------------------------------------------------------

TUNNEL_RADIUS = 1.0
DT = 1.0 #constant time between keyframes
T_final = 75.0

R0 = geo.Rot3.from_yaw_pitch_roll(0,np.radians(10),0)
t0 = geo.V3()
X0 = geo.Pose3(R = R0, t = t0)

w_world = geo.V3([0.0,0.0,2 * np.pi/30]) #30 seconds for a full rotation
w_robot = X0.inverse() * w_world
v_robot = geo.V3([-3.0,0.0,0.0])

twist = geo.V6()
twist[:3] = w_robot
twist[3:] = v_robot
twist = np.array(twist,dtype = float)
#-------------------------------------------------------------------------------------------
#-------------------------build tunnel and extract ground truth key poses and data --------
#-------------------------------------------------------------------------------------------
history = {"gt_x": [X0],
           "gt_l": [],
            "z" : [],
            "da" : [],
            "u" : []}
time_from_last_keypose = 0.0
DT_TUNNELING = DT/3
x = X0
#build tunnel
for t in np.arange(0,T_final,DT_TUNNELING):
    #move robot
    x = x * geo.Pose3.from_tangent(twist * DT_TUNNELING)

    #perturb pose only on roll
    tangent_perturbation = np.zeros(6); 
    tangent_perturbation[0] = np.random.uniform(-np.pi,+np.pi)
    x_perturb = x * geo.Pose3.from_tangent(tangent_perturbation)
    #calculate landmark in world coordinates
    lm = x_perturb * geo.Vector3(np.array([0,TUNNEL_RADIUS,0]))        
    
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
SENSOR_COV = np.diag([2.0,np.radians(3),np.radians(3)])
ODOM_COV = 1e-3 * np.eye(6) #in [rad,rad,rad,m,m,m]**2
ODOM_COV[0,0] = (twist[0]/4 * DT)**2
ODOM_COV[2,2] = (twist[2]/4 * DT)**2
ODOM_COV[3,3] = (twist[3]/4 * DT)**2
landmarks = np.array(history["gt_l"]).astype(float)
#pass through tunnel - collect measurements. 
for k, x in enumerate(history["gt_x"]):
    if k == 0: continue
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
    history["z"] += [zk]
    history["da"] += [dak]

    noisy_u = np.random.multivariate_normal(twist * DT,ODOM_COV)
    history["u"] += [noisy_u]

#-------------------------------------------------------------------------------------------
#------------------------------------ DEAD RECKONING ---------------------------------------
#-------------------------------------------------------------------------------------------
estimation = {}
estimation["dr_x"] = [X0]

#dead reckoning for initial estimation of [x]
for k, uk in enumerate(history["u"]):
    estimation["dr_x"] += [estimation["dr_x"][-1] * geo.Pose3.from_tangent(uk)]

#dead reckoning for landmarks from first sighting projection, using dr_x
N_landmarks = 0
for dak in history["da"]:
    if len(dak) == 0: continue
    N_landmarks = max(N_landmarks, np.max(np.array(dak)[:,1]))
N_landmarks +=1 #dak measures indicies
estimation["dr_l"] = [False for _ in range(N_landmarks)] #initalizes with falses
for k, dr_x in enumerate(estimation["dr_x"]):
    if k == 0: continue
    i = k-1 #from x[1] we measure z[0]
    dak = history["da"][i]
    zk = history["z"][i]
    if len(dak) == 0: continue
    for i, d in enumerate(dak):
        if estimation["dr_l"][d[1]] == False:
            estimation["dr_l"][d[1]] = dr_x * geo.V3(spherical_to_cartesian(zk[i]))

#-------------------------------------------------------------------------------------------
#--------------------------------REGULAR SLAM OPTIMIZATION ---------------------------------
#-------------------------------------------------------------------------------------------
        
values = Values()
PRIOR_COV = np.eye(6) *1e-3
values["odom_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(ODOM_COV)))
values["meas_sqrtInfo"] = geo.V3(np.diag(cov2sqrtInfo(SENSOR_COV)))
values["prior_sqrtInfo"] = geo.V6(np.diag(cov2sqrtInfo(PRIOR_COV)))
values["epsilon"] = sm.numeric_epsilon
values["u"] = [geo.Pose3.from_tangent(uk) for uk in history["u"]]
values["z"] = history["z"]
values["da"] = history["da"]
values["x0"] = estimation["dr_x"][0]
values["x"] = estimation["dr_x"] #initalized from dead reckoning
values["l"] = estimation["dr_l"]

factors = []
# prior
factors.append(
        Factor(residual = pose3prior_residual,
        keys = [
                f"x[0]",
                "x0",
                "prior_sqrtInfo",
                "epsilon"
        ]))
# odometry
for k in range(len(values["u"])):
    factors.append(
    Factor(residual = odometry_residual,
    keys = [
            f"x[{k}]",
            f"x[{k+1}]",
            f"u[{k}]",
            "odom_sqrtInfo",
            "epsilon",
    ]))

#measurements
for i, dai in enumerate(values["da"]):
        for j in range(len(dai)):
                factors.append(
                        Factor(residual = measurement_residual,
                        keys = [
                                f"x[{dai[j][0]}]",
                                f"l[{dai[j][1]}]",
                                f"z[{i}][{j}]",
                                "meas_sqrtInfo",
                        ]))

optimized_keys_x = [f"x[{k}]" for k in range(len(values["x"]))]
optimized_keys_l = [f"l[{k}]" for k in range(len(values["l"])) if values["l"][k] != False]
optimized_keys = optimized_keys_x + optimized_keys_l

optimizer = Optimizer(
factors=factors,
optimized_keys=optimized_keys,
# Return problem stats for every iteration
debug_stats=True,
# Customize optimizer behavior
params=Optimizer.Params(verbose=False, enable_bold_updates = False)
)
result = optimizer.optimize(values)
optVals_slam = result.optimized_values

#-------------------------------------------------------------------------------------------
#--------------------------------ADDING MY FACTORS SLAM OPTIMIZATION ----------------------
#-------------------------------------------------------------------------------------------

#ferguson factors
values["TUNNEL_RADIUS"] = 0.5
values["DT"] = DT
values["s"] = [0.0 for _ in range(len(values["l"]))]
for i in range(len(values["l"])):
    l = values["l"][i]
    front_dist = np.inf; back_dist = np.inf
    front_index = -1; back_index = -1
    for k, x in enumerate(values["x"]):
        rel_lm = x.inverse() * geo.V3(l)
        d = float(rel_lm.norm())
        theta = float(sm.atan2(rel_lm[1],rel_lm[0])) #yaw #arctan2(y,x)
        if abs(theta) < np.pi/2 and d < front_dist:
            front_dist = d
            front_index = k
        else:
            if d < back_dist:
                back_dist = d
                back_index = k
    #initalize s
    values["s"][i] = front_dist / (front_dist + back_dist)
    #add factor

    factors.append(
            Factor(residual = ferguson_residual,
            keys = [
                    f"s[{i}]",
                    f"x[{back_index}]",
                    f"x[{front_index}]",
                    f"l[{i}]",
                    "TUNNEL_RADIUS",
                    "DT",
            ]))
#boundries factors
values["s_lower_limit"] = 0.0
values["s_upper_limit"] = 1.0
values["s_boundry_eps"] = 0.01
values["s_boundry_k"] = 10.0
for s in range(len(values["s"])):
    factors.append(
            Factor(residual = boundry_factor,
            keys = [
                    f"s[{i}]",
                    "s_lower_limit",
                    "s_upper_limit",
                    "s_boundry_eps",
                    "s_boundry_k",
            ]))
optimized_keys_s =  [f"s[{j}]" for j in range(len(values["s"]))]
optimized_keys = optimized_keys + optimized_keys_s

optimizer = Optimizer(
factors=factors,
optimized_keys=optimized_keys,
# Return problem stats for every iteration
debug_stats=True,
# Customize optimizer behavior
params=Optimizer.Params(verbose=False, enable_bold_updates = False)
)
result = optimizer.optimize(values)
optVals_tunnelslam = result.optimized_values
#-------------------------------------------------------------------------------------------
#----------------------------------------PLOT-----------------------------------------------
#-------------------------------------------------------------------------------------------

fig = plt.figure()
ax = plt.axes(projection='3d', aspect='equal',
        xlim = (-10,10), ylim = (-15,5), zlim = (0,40),
        xlabel = 'x', ylabel = 'y', zlabel = 'z')
plotting.axis_equal(ax)
#plot trajectory
for x in history["gt_x"]:
    plotting.plotPose3(ax,x, color = "red")
for x in estimation["dr_x"]:
    plotting.plotPose3(ax,x,color = "green")
for x in optVals_slam["x"]:
    plotting.plotPose3(ax,x,color = "orange")
for x in optVals_tunnelslam["x"]:
    plotting.plotPose3(ax,x,color = "blue")
#plot landmarks
gt_landmarks = np.array(history["gt_l"]).astype(float)
ax.scatter(gt_landmarks[:,0], gt_landmarks[:,1], gt_landmarks[:,2])

dr_landmarks = np.array(estimation["dr_l"],dtype = float)
ax.scatter(dr_landmarks[:,0], dr_landmarks[:,1], dr_landmarks[:,2],c = 'gray', marker = 'x')

for i, da in enumerate(history["da"]):
    for j, d in enumerate(da):
        x = np.array(history["gt_x"][i].t)
        l = np.array(history["gt_l"][d[1]])
        ax.plot([x[0],l[0]],[x[1],l[1]],[x[2],l[2]],
                color = 'k', linewidth = 0.5, linestyle = '-')
plt.show()