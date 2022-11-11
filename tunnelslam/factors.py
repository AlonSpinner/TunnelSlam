from symforce import geo
from symforce import typing as T
from symforce import symbolic as sm
import numpy as np

def measurement_residual(
    x: geo.Pose3,
    lm: geo.V3, 
    z: geo.V3, 
    sqrtInfo: geo.V3 #diagonal of sqrt information matrix
) -> geo.V3:
    """
    Residual from a relative translation mesurement of a 3D pose to a landmark.

    Args:
        x: 3D pose of the robot in the world frame
        lm: World location of the landmark
        z: measured [range,yaw,pitch]
        info: measurement information matrix
    """
    rel_lm = x.inverse() * lm
    r = rel_lm.norm()
    theta = sm.atan2(rel_lm[1],rel_lm[0]) #yaw #arctan2(y,x)
    psi = sm.atan2(rel_lm[2],geo.V2(rel_lm[:2]).norm()) #pitch
    h = geo.V3([r,theta,psi])
    e = z-h
    return geo.M.diag(sqrtInfo) * e


def odometry_residual(
    x1: geo.Pose3,
    x2: geo.Pose3,
    odom: geo.Pose3,
    sqrtInfo: geo.V6, #diagonal of sqrt information matrix
    epsilon: T.Scalar,
) -> geo.V6:
    """
    Residual on the relative pose between two timesteps of the robot.

    Args:
        x1: First pose
        x2: Second pose
        odom: Relative pose measurement between the poses
        sqrtInfo: Sqrt information matrix
        epsilon: Small number for singularity handling
    """
    predict = x1.inverse() * x2
    tangent_error = predict.local_coordinates(odom, epsilon = epsilon)
    return geo.M.diag(sqrtInfo) * geo.V6(tangent_error)

def odometry_residual_tangent(
    x1: geo.Pose3,
    x2: geo.Pose3,
    u: geo.V6,
    sqrtInfo: geo.V6, #diagonal of sqrt information matrix
    epsilon: T.Scalar,
) -> geo.V6:
    """
    Residual on the relative pose between two timesteps of the robot.

    Args:
        x1: First pose
        x2: Second pose
        odom: Relative pose measurement between the poses
        sqrtInfo: Sqrt information matrix
        epsilon: Small number for singularity handling
    """
    predict = x1.retract(u, epsilon = epsilon)
    tangent_error = predict.local_coordinates(x2, epsilon = epsilon)
    return geo.M.diag(sqrtInfo) * geo.V6(tangent_error)


def pose3prior_residual(
    x : geo.Pose3, 
    x0: geo.Pose3, 
    sqrtInfo: geo.V6,
    epsilon: T.Scalar):

    '''
    Args:
    x: pose
    x0: prior pose
    sqrtInfo: Sqrt information matrix
    epsilon: Small number for singularity handling
    '''

    predict = x0.inverse() * x
    tangent_error = predict.to_tangent(epsilon = epsilon)
    return geo.M.diag(sqrtInfo) * geo.V6(tangent_error)

def radial_residual(
    x: geo.Pose3, 
    lm: geo.V3, 
    r : T.Scalar
) -> geo.V1:
    """
        x: 3D pose of the robot in the world frame
        lm: World location of the landmark
    """
    rel_lm = x.inverse() * lm
    e = geo.V2(rel_lm[1:]).norm() - r
    return geo.V1(e)

def radial_residual2(
    s : T.Scalar,
    x1 : geo.Pose3,
    x2 : geo.Pose3,
    lm : geo.V3,
    r : T.Scalar,
    d : T.Scalar
    ):
    """
    s - running parameter s in [0,1] .. how do we constrain?
    x1 - pose at s = 0
    x2 - pose at s = 1
    lm - landmark
    r - tunnel radius
    d - 1 if this factor is relevant, 0 otherwise (that is, if lm is attached to this spline)
    """
    #lets assume m derivatives were calculated for now...
    p0 = x1.t
    m0 = x1.R.to_rotation_matrix()[:,0]
    p1 = x2.t
    m1 = x2.R.to_rotation_matrix()[:,0]

    #lets create a curve with some running parameter u
    splinePoint = (2*s**3-3*s**2+1) * p0 \
            + (s**3 - 2*s**2 + s) * m0 \
            + (-2*s**3 + 3*s**2) * p1 \
            + (s**3 - s**2) * m1

    rhat = (lm-splinePoint).norm()
    e = (rhat - r) * d
    return geo.V1(e)

def cov2sqrtInfo(M : np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(M).T