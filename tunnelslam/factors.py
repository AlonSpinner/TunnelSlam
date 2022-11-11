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

def ferguson_residual(
    s : T.Scalar,
    x1 : geo.Pose3,
    x2 : geo.Pose3,
    l : geo.V3,
    d : T.Scalar,
    DT : T.Scalar
    ):
    """
    Residual on the relative pose between two timesteps of the robot.

    Args:
        x1: First pose
        x2: Second pose
        odom: Relative pose measurement between the poses
        sqrtInfo: Sqrt information matrix
        epsilon: Small number for singularity handling
    """
    twist = geo.V6((x1.inverse() * x2).to_tangent())/DT
    m0 = twist[3:,0]
    m1 = m0 #assume constant velocity

    splinePoint = (2*s**3-3*s**2+1) * x1.t \
        + (s**3 - 2*s**2 + s) * m0 \
        + (-2*s**3 + 3*s**2) * x2.t \
        + (s**3 - s**2) * m1

    d_predict = (splinePoint - l).norm()
    e = geo.V1((d - d_predict)**2)
    return e

def boundry_factor(
    s : T.Scalar,
    lower_lim : T.Scalar,
    upper_lim : T.Scalar,
    eps : T.Scalar,
    k : T.Scalar):
    if s < lower_lim + eps:
        return geo.V1(k * abs(s - lower_lim))
    elif s > upper_lim - eps:
        return geo.V1(k * abs(s - upper_lim))
    else:
        return geo.V1(0.0)

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

def cov2sqrtInfo(M : np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(np.linalg.inv(M)).T