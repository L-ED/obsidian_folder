
visual odometry returns transformation matrix between frames. IMU sensor returns angular velocity and acceleration in copter coordinate frame. To fuse measurements into one state, we can use loosely coupled and tightly coupled methods. Loosely coupled methods estimate motion from IMU and camera frames independently and then fuse to get output. Tightly coupled methods fuse raw data to get estimation and more accurate because visual info can correct IMU drift and inertial measurements can correct visual features.

VIO algs can be categorized by number of involved in estimation frames. Full smoothers keep whole trajectory for optimization. Sliding window approaches track only N latest frames for estimation of new state. Filters only estimate latest state

Different methods can be used for measurement uncertainty. Extended Kalman Filter(EKF) produce covariance matrix,  