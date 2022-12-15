# TunnelSlam
In attempt to create a speciflied SLAM solution for a tunneling robot

Idea:

1) Create hermite cubic spline between every two key robot poses
2) Assume constant twist between these poses
3) each point-landmark is assumed to belong to a single hermite cubic spline, and given a curve parameter to be estimated
4) the distance from the landmark to the curve is to be constrained using the curve parameter
5) how: new factors with symforce

