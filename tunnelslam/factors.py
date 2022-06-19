from symforce import geo
from symforce import typing as T
from symforce import sympy as sm
import numpy as np


def radial_residual(
    x: geo.Pose3, 
    lm: geo.V3, 
    r : T.Scalar
) -> T.Scalar:
    """
    Residual from a relative translation mesurement of a 3D pose to a landmark.

    Args:
        x: 3D pose of the robot in the world frame
        lm: World location of the landmark
    """
    rel_lm = x.inverse() * lm
    e = geo.V2(rel_lm[1:]).norm() - r
    return e

def measurement_residual(
    x: geo.Pose3,
    lm: geo.V3, 
    z: geo.V3, 
    sqsrtInfo: geo.V3 #diagonal of sqrt information matrix
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
    return geo.M.diag(sqsrtInfo) * e


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


def cov2sqrtInfo(M : np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(M).T
