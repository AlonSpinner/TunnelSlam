
from symforce import geo
from symforce import typing as T
from symforce import sympy as sm


def radial(
    x: geo.Pose3, lm: geo.V3, r : T.Scalar
) -> T.Scalar:
    """
    Residual from a relative translation mesurement of a 3D pose to a landmark.

    Args:
        x: 3D pose of the robot in the world frame
        lm: World location of the landmark
    """
    rel_lm = x.inverse() * lm
    e = geo.V2(rel_lm[1:]).norm() - r
    return e**2

def measurement_residual(
    x: geo.Pose3, lm: geo.V3, z: geo.V3, info: geo.Matrix33
) -> T.Scalar:
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
    return e.T * info * e


def odometry_residual(
    x1: geo.Pose3_SE3,
    x2: geo.Pose3_SE3,
    odom: geo.Pose3_SE3,
    diagonal_sigmas: geo.V6,
    epsilon: T.Scalar,
) -> geo.V6:
    """
    Residual on the relative pose between two timesteps of the robot.

    Args:
        world_T_a: First pose in the world frame
        world_T_b: Second pose in the world frame
        a_T_b: Relative pose measurement between the poses
        diagonal_sigmas: Diagonal standard deviation of the tangent-space error
        epsilon: Small number for singularity handling
    """
    predict = x1.inverse() * x2
    tangent_error = predict.local_coordinates(odom, epsilon=epsilon)
    return T.cast(geo.V6, geo.M.diag(diagonal_sigmas.to_flat_list()).inv() * geo.V6(tangent_error))