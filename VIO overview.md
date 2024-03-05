
visual odometry returns transformation matrix between frames. IMU sensor returns angular velocity and acceleration in copter coordinate frame. To fuse measurements into one state, we can use loosely coupled and tightly coupled methods. Loosely coupled methods estimate motion from IMU and camera frames independently and then fuse to get output. Tightly coupled methods fuse raw data to get estimation and more accurate because visual info can correct IMU drift and inertial measurements can correct visual features.

VIO algs can be categorized by number of involved in estimation frames. Full smoothers keep whole trajectory for optimization. Sliding window approaches track only N latest frames for estimation of new state. Filters only estimate latest state

Different methods can be used for measurement uncertainty. Extended Kalman Filter(EKF) produce covariance matrix,  Smoothers and information filters use information matrix(covar inv). The number of times in which the measurement model is linearized is also an important criterion. While a standard EKF (in contrast to the iterated EKF) processes a measurement only once, a smoothing approach allows linearizing multiple times. While the terminology is vast, the underlying algorithms are tightly related. For instance, it can be shown that the iterated Extended Kalman filter equations are equivalent to the Gauss-Newton algorithm, commonly used for smoothing (Bell and Cathey 1993).

Filtering 

Efficient because of estimating only last state. Classic approach estimate pose and landmarks in state and filter complexity grows quadratically in the number of landmarks