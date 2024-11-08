import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.linalg import block_diag

class RobotLocalizationEKF:
    def __init__(self, initial_pose, initial_covariance):
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)
        self.ekf.x = np.array(initial_pose)  # [x, y, theta]
        self.ekf.P = initial_covariance  # Initial covariance matrix

        # Define motion noise
        self.ekf.R = np.diag([0.1, 0.1])  # Measurement noise
        self.ekf.Q = np.diag([0.1, 0.1, np.deg2rad(5)**2])  # Process noise

        # Define the state transition function
        self.ekf.F = np.eye(3)

        # Define the measurement function (e.g., lidar)
        self.ekf.H = np.zeros((2,3))
        self.ekf.H[0,0] = 1
        self.ekf.H[1,1] = 1

    def predict(self, control_input, dt):
        """
        Predict the next state based on control input.
        control_input: [v, omega] - linear and angular velocities
        dt: time step
        """
        v, omega = control_input
        theta = self.ekf.x[2]

        if omega == 0:
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0
        else:
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
            dtheta = omega * dt

        # State transition
        self.ekf.x = self.ekf.x + np.array([dx, dy, dtheta])

        # Jacobian of the motion model
        self.ekf.F = np.array([
            [1, 0, -v / omega * np.cos(theta) + v / omega * np.cos(theta + omega * dt)],
            [0, 1, -v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * dt)],
            [0, 0, 1]
        ])

        self.ekf.predict()

    def update(self, measurement):
        """
        Update the state with a new measurement.
        measurement: [x, y] position from lidar
        """
        self.ekf.update(measurement, HJacobian=self.h_jacobian, Hx=self.h_x)

    def h_jacobian(self, x):
        """
        Jacobian of the measurement function h(x).
        """
        return self.ekf.H

    def h_x(self, x):
        """
        Measurement function h(x).
        """
        return np.array([x[0], x[1]])

    def get_pose(self):
        return self.ekf.x

    def get_covariance(self):
        return self.ekf.P