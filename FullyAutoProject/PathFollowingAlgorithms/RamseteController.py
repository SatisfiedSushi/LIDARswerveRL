# ramsete_controller.py

import math
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List


class RamseteController:
    def __init__(self, b: float = 2.0, zeta: float = 0.7):
        """
        Initialize the Ramsete controller with specified gains.

        :param b: Ramsete parameter b (typically 2.0)
        :param zeta: Ramsete damping ratio zeta (typically 0.7)
        """
        self.b = b
        self.zeta = zeta
        self.desired_path: List[Tuple[float, float, float]] = []  # List of (x, y, theta)
        self.actual_path: List[Tuple[float, float, float]] = []   # List of (x, y, theta)
        logging.debug(f"Initialized RamseteController with b={self.b}, zeta={self.zeta}")

    def set_target(self, des_x: float, des_y: float, des_theta: float,
                  vel_des: float, omega_des: float):
        """
        Set the desired target state.

        :param des_x: Desired x position (meters)
        :param des_y: Desired y position (meters)
        :param des_theta: Desired orientation (radians)
        :param vel_des: Desired linear velocity (m/s)
        :param omega_des: Desired angular velocity (rad/s)
        """
        self.des_x = des_x
        self.des_y = des_y
        self.des_theta = des_theta
        self.vel_des = vel_des
        self.omega_des = omega_des
        self.desired_path.append((des_x, des_y, des_theta))
        logging.debug(f"Set target to (x={des_x:.4f}, y={des_y:.4f}, theta={math.degrees(des_theta):.2f}°), "
                      f"vel_des={vel_des:.4f} m/s, omega_des={math.degrees(omega_des):.2f}°/s")

    def compute_control(self, current_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Compute the Ramsete control commands based on the current pose.

        :param current_pose: Current pose as (x, y, theta)
        :return: Control commands as (linear_velocity, angular_velocity)
        """
        x, y, theta = current_pose
        # Assume that set_target has been called prior to this step
        x_d, y_d, theta_d = self.des_x, self.des_y, self.des_theta
        v_d, omega_d = self.vel_des, self.omega_des

        # Compute pose error
        error_x = x_d - x
        error_y = y_d - y
        error_theta = self._normalize_angle(theta_d - theta)

        # Transform the error into the robot's frame
        error_linear = [
            math.cos(theta) * error_x + math.sin(theta) * error_y,
            -math.sin(theta) * error_x + math.cos(theta) * error_y
        ]
        e_x, e_y = error_linear

        logging.debug(f"Pose Error - error_x: {e_x:.4f} m, error_y: {e_y:.4f} m, error_theta: {math.degrees(error_theta):.4f}°")

        # Ramsete Control Law
        k = 2 * self.zeta * math.sqrt(omega_d**2 + self.b * v_d**2)
        v = v_d * math.cos(error_theta) + k * e_x
        omega = omega_d + k * error_theta + self.b * v_d * math.sin(error_theta) * e_y

        logging.debug(f"Ramsete Control Commands - v: {v:.4f} m/s, omega: {math.degrees(omega):.4f}°/s")

        # Append to actual path
        self.actual_path.append((x, y, theta))

        return v, omega

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize an angle to the range [-pi, pi].

        :param angle: Angle in radians.
        :return: Normalized angle in radians.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def plot_trajectory(self):
        """
        Plot the desired and actual trajectories.
        """
        if not self.desired_path:
            logging.warning("No desired path to plot.")
            return

        desired_x = [point[0] for point in self.desired_path]
        desired_y = [point[1] for point in self.desired_path]
        actual_x = [point[0] for point in self.actual_path]
        actual_y = [point[1] for point in self.actual_path]

        plt.figure(figsize=(10, 8))
        plt.plot(desired_x, desired_y, 'r--', label='Desired Trajectory')
        plt.plot(actual_x, actual_y, 'b-', label='Actual Path')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Ramsete Trajectory Following')
        plt.legend()
        plt.grid(True)
        plt.show()
