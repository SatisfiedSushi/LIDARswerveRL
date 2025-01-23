import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

@dataclass
class TrajectoryPoint:
    x: float          # X position (meters)
    y: float          # Y position (meters)
    theta: float      # Orientation (radians)
    v: float          # Linear velocity (m/s)
    omega: float      # Angular velocity (rad/s)
    time_stamp: float # Time stamp (seconds)

# --------------------- Trajectory Class ---------------------

class Trajectory:
    def __init__(self, points: List[TrajectoryPoint]):
        self.points = points
        logging.debug(f"Trajectory initialized with {len(self.points)} points.")

    @staticmethod
    def create_straight_line(start_pose: Tuple[float, float, float],
                             end_pose: Tuple[float, float, float],
                             total_time: float,
                             dt: float) -> 'Trajectory':
        """Generates a straight-line trajectory from start_pose to end_pose."""
        start_x, start_y, start_theta = start_pose
        end_x, end_y, end_theta = end_pose
        num_points = int(total_time / dt) + 1
        points = []
        logging.debug(f"Creating straight line trajectory from {start_pose} to {end_pose} over {total_time}s with dt={dt}s")
        for i in range(num_points):
            t = i * dt
            ratio = t / total_time if total_time != 0 else 1.0
            ratio = min(ratio, 1.0)  # Clamp ratio to [0,1]
            x = start_x + ratio * (end_x - start_x)
            y = start_y + ratio * (end_y - start_y)
            theta = start_theta + ratio * (end_theta - start_theta)
            v = math.hypot(end_x - start_x, end_y - start_y) / total_time if total_time != 0 else 0.0
            omega = (end_theta - start_theta) / total_time if total_time != 0 else 0.0
            points.append(TrajectoryPoint(x, y, theta, v, omega, t))
            logging.debug(f"Straight Line - Point {i}: (x={x:.4f}, y={y:.4f}, theta={math.degrees(theta):.2f}°), "
                          f"v={v:.4f} m/s, omega={math.degrees(omega):.2f}°/s, t={t:.2f}s")
        logging.info(f"Straight line trajectory created with {num_points} points.")
        return Trajectory(points)

    @staticmethod
    def create_circular_arc(center: Tuple[float, float],
                            radius: float,
                            start_angle: float,
                            end_angle: float,
                            angular_velocity: float,
                            total_time: float,
                            dt: float) -> 'Trajectory':
        """Generates a circular arc trajectory."""
        center_x, center_y = center
        num_points = int(total_time / dt) + 1
        points = []
        logging.debug(f"Creating circular arc trajectory with center=({center_x}, {center_y}), radius={radius}, "
                      f"from {math.degrees(start_angle):.2f}° to {math.degrees(end_angle):.2f}° over {total_time}s with dt={dt}s")
        for i in range(num_points):
            t = i * dt
            theta = start_angle + angular_velocity * t
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            v = radius * angular_velocity
            omega = angular_velocity
            points.append(TrajectoryPoint(x, y, theta, v, omega, t))
            logging.debug(f"Circular Arc - Point {i}: (x={x:.4f}, y={y:.4f}, theta={math.degrees(theta):.2f}°), "
                          f"v={v:.4f} m/s, omega={math.degrees(omega):.2f}°/s, t={t:.2f}s")
        logging.info(f"Circular arc trajectory created with {num_points} points.")
        return Trajectory(points)

    @staticmethod
    def combine_trajectories(traj1: 'Trajectory', traj2: 'Trajectory') -> 'Trajectory':
        """Combines two trajectories sequentially."""
        combined_points = traj1.points.copy()
        # Offset the time stamps of traj2
        if len(traj1.points) > 0 and len(traj2.points) > 1:
            offset_time = traj1.points[-1].time_stamp + (traj2.points[1].time_stamp - traj2.points[0].time_stamp)
        else:
            offset_time = 0.0
        logging.debug(f"Combining trajectories with time offset={offset_time:.2f}s")
        for i, point in enumerate(traj2.points[1:], start=1):
            combined_points.append(TrajectoryPoint(point.x, point.y, point.theta,
                                                  point.v, point.omega, point.time_stamp + offset_time))
            logging.debug(f"Combined Trajectory - Point {i}: (x={point.x:.4f}, y={point.y:.4f}, "
                          f"theta={math.degrees(point.theta):.2f}°), v={point.v:.4f} m/s, "
                          f"omega={math.degrees(point.omega):.2f}°/s, t={point.time_stamp + offset_time:.2f}s")
        logging.info(f"Combined trajectory created with {len(combined_points)} points.")
        return Trajectory(combined_points)

    def get_desired_state(self, elapsed_time: float, tolerance: float=1e-2) -> Tuple[TrajectoryPoint, Tuple[float, float]]:
        """Retrieves the desired trajectory point and velocity at the given elapsed_time."""
        for point in self.points:
            if math.isclose(point.time_stamp, elapsed_time, abs_tol=tolerance) or point.time_stamp > elapsed_time:
                desired_point = point
                desired_velocity = (point.v, point.omega)
                logging.debug(f"Desired state at time {elapsed_time:.2f}s: (x={desired_point.x:.4f}, "
                              f"y={desired_point.y:.4f}, theta={math.degrees(desired_point.theta):.2f}°), "
                              f"v={desired_velocity[0]:.4f} m/s, omega={math.degrees(desired_velocity[1]):.2f}°/s")
                return desired_point, desired_velocity
        # If time exceeds trajectory, return last point
        last_point = self.points[-1]
        logging.debug(f"Elapsed time {elapsed_time:.2f}s exceeds trajectory. Using last point: (x={last_point.x:.4f}, "
                      f"y={last_point.y:.4f}, theta={math.degrees(last_point.theta):.2f}°), "
                      f"v={last_point.v:.4f} m/s, omega={math.degrees(last_point.omega):.2f}°/s")
        return last_point, (last_point.v, last_point.omega)