import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# --------------------- Logging Configuration ---------------------

# Configure logging to display messages with timestamps and log levels
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ramsete_simulation_debug.log"),
                        logging.StreamHandler()
                    ])

# --------------------- Data Classes ---------------------

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

# --------------------- Ramsete Controller Class ---------------------

class RamseteController:
    def __init__(self, b: float=2.0, zeta: float=0.7):
        """
        Initialize the Ramsete controller with specified gains.
        :param b: Ramsete parameter b (typically 2.0)
        :param zeta: Ramsete damping ratio zeta (typically 0.7)
        """
        self.b = b
        self.zeta = zeta
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
        logging.debug(f"Target set to (x={self.des_x:.4f}, y={self.des_y:.4f}, theta={math.degrees(self.des_theta):.2f}°), "
                      f"v_des={self.vel_des:.4f} m/s, omega_des={math.degrees(self.omega_des):.2f}°/s")

    def step(self, current_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Compute the control commands based on the current pose.
        :param current_pose: Current pose as (x, y, theta)
        :return: Control commands as (linear_velocity, angular_velocity)
        """
        x, y, theta = current_pose
        # Pose error in the robot's frame
        error_x = math.cos(theta) * (self.des_x - x) + math.sin(theta) * (self.des_y - y)
        error_y = -math.sin(theta) * (self.des_x - x) + math.cos(theta) * (self.des_y - y)
        error_theta = self.des_theta - theta
        # Normalize error_theta to [-pi, pi]
        error_theta = (error_theta + math.pi) % (2 * math.pi) - math.pi

        logging.debug(f"Pose Error - error_x: {error_x:.4f} m, error_y: {error_y:.4f} m, error_theta: {math.degrees(error_theta):.4f}°")

        # Ramsete Control Law
        k = 2 * self.zeta * math.sqrt(self.omega_des**2 + self.b * self.vel_des**2)
        v = self.vel_des * math.cos(error_theta) + k * error_x
        # Avoid division by zero when error_theta is very small
        if abs(error_theta) > 1e-6:
            omega = self.omega_des + k * error_theta + self.b * self.vel_des * math.sin(error_theta) * error_y / error_theta
        else:
            omega = self.omega_des + k * error_theta + self.b * self.vel_des * error_y

        # Convert omega to degrees/s for logging
        omega_deg = math.degrees(omega)

        logging.debug(f"Control Commands - v: {v:.4f} m/s, omega: {omega_deg:.4f}°/s")
        print(f'step')

        return v, omega

# --------------------- Environment Class ---------------------

class Environment:
    def __init__(self, initial_pose: Tuple[float, float, float]=(0.0, 0.0, 0.0)):
        """
        Initialize the simulation environment with the robot's initial pose.
        :param initial_pose: Initial pose as (x, y, theta)
        """
        self.x, self.y, self.theta = initial_pose
        logging.debug(f"Initialized Environment with pose: (x={self.x:.4f}, y={self.y:.4f}, theta={math.degrees(self.theta):.2f}°)")

    def get_current_pose(self) -> Tuple[float, float, float]:
        """
        Get the current pose of the robot.
        :return: Current pose as (x, y, theta)
        """
        return (self.x, self.y, self.theta)

    def apply_action(self, v: float, omega: float, dt: float):
        """
        Update the robot's pose based on control commands.
        :param v: Linear velocity (m/s)
        :param omega: Angular velocity (rad/s)
        :param dt: Time step (s)
        """
        # Kinematic model update
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        logging.debug(f"Applied Action - v: {v:.4f} m/s, omega: {math.degrees(omega):.4f}°/s, "
                      f"New Pose: (x={self.x:.4f}, y={self.y:.4f}, theta={math.degrees(self.theta):.2f}°)")

# --------------------- Trajectory Creation Function ---------------------

def create_test_trajectory() -> Trajectory:
    """
    Creates a combined trajectory consisting of a straight line followed by a circular arc.
    :return: Combined Trajectory object
    """
    # Straight line parameters
    start_pose = (0.0, 0.0, 0.0)             # (x, y, theta) in meters and radians
    end_pose = (5.0, 0.0, 0.0)               # Move 5 meters along x-axis
    straight_total_time = 5.0                 # seconds
    dt = 0.05                                 # time step in seconds

    # Create straight line trajectory
    straight_traj = Trajectory.create_straight_line(start_pose, end_pose, straight_total_time, dt)

    # Circular arc parameters
    center = (5.0, 5.0)                        # Center of the circle
    radius = 5.0                               # meters
    start_angle_deg = -90.0                    # Starting at -90 degrees
    end_angle_deg = 0.0                        # Ending at 0 degrees
    angular_velocity_deg = 36.0                # degrees per second
    angular_velocity = math.radians(angular_velocity_deg)  # radians per second
    circular_total_time = 7.5                  # seconds

    start_angle = math.radians(start_angle_deg)
    end_angle = math.radians(end_angle_deg)

    # Create circular arc trajectory
    circular_traj = Trajectory.create_circular_arc(center, radius, start_angle, end_angle,
                                                  angular_velocity, circular_total_time, dt)

    # Combine trajectories
    combined_traj = Trajectory.combine_trajectories(straight_traj, circular_traj)

    return combined_traj

# --------------------- Main Simulation Function ---------------------

def main():
    # Initialize Environment and Controller
    initial_pose = (0.0, 0.0, 0.0)  # (x, y, theta) in meters and radians
    env = Environment(initial_pose)
    controller = RamseteController(b=2.0, zeta=0.7)
    logging.info("Initialized Environment and Ramsete Controller.")

    # Create Trajectory
    trajectory = create_test_trajectory()
    logging.info("Test trajectory created.")

    # Simulation Parameters
    dt = 0.05  # Time step in seconds
    total_simulation_time = trajectory.points[-1].time_stamp + 2.0  # Extra time to ensure completion
    total_frames = int(total_simulation_time / dt) + 1
    logging.info(f"Total Simulation Time: {total_simulation_time:.2f} seconds")
    logging.info(f"Total Frames: {total_frames}")

    # Visualization Setup
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    # Determine plot limits based on trajectory
    all_x = [point.x for point in trajectory.points]
    all_y = [point.y for point in trajectory.points]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Ramsete Trajectory Following Simulation')

    # Plot desired trajectory
    desired_x = [point.x for point in trajectory.points]
    desired_y = [point.y for point in trajectory.points]
    ax.plot(desired_x, desired_y, 'r--', label='Desired Trajectory')
    logging.debug("Desired trajectory plotted.")

    # Initialize robot path
    path_x, path_y = [], []
    path_line, = ax.plot([], [], 'b-', label='Robot Path')
    logging.debug("Robot path initialized.")

    # Initialize robot marker
    robot_marker, = ax.plot([], [], 'ko', label='Robot')
    logging.debug("Robot marker initialized.")

    # Initialize robot orientation arrow using FancyArrowPatch
    robot_arrow = FancyArrowPatch((env.x, env.y),
                                  (env.x + 0.5 * math.cos(env.theta),
                                   env.y + 0.5 * math.sin(env.theta)),
                                  arrowstyle='->', mutation_scale=10, color='k')
    ax.add_patch(robot_arrow)
    logging.debug("Robot orientation arrow initialized.")

    ax.legend()

    # Define the update function for animation
    def update(frame):
        logging.debug(f"Update function called for frame {frame}")
        try:
            # Calculate elapsed time based on frame number
            elapsed_time = frame * dt

            # Log elapsed time
            logging.debug(f"Elapsed Time: {elapsed_time:.2f}s")

            # Check if the simulation should continue
            if elapsed_time > total_simulation_time:
                logging.info("Elapsed time exceeded total simulation time. Closing the figure.")
                plt.close(fig)
                return

            # Get desired state from trajectory
            desired_point, desired_velocity = trajectory.get_desired_state(elapsed_time, tolerance=dt/2)
            desired_pose = (desired_point.x, desired_point.y, desired_point.theta)

            # Set target in controller
            controller.set_target(desired_point.x, desired_point.y, desired_point.theta,
                                 desired_point.v, desired_point.omega)

            # Get current pose
            current_pose = env.get_current_pose()
            logging.debug(f"Current Pose: (x={current_pose[0]:.4f}, y={current_pose[1]:.4f}, theta={math.degrees(current_pose[2]):.2f}°)")

            # Compute control commands
            v, omega = controller.step(current_pose)
            logging.debug(f"Control Commands Computed: v={v:.4f} m/s, omega={math.degrees(omega):.2f}°/s")

            # Apply action to the environment
            env.apply_action(v, omega, dt)

            # Update path for visualization
            path_x.append(env.x)
            path_y.append(env.y)
            path_line.set_data(path_x, path_y)
            logging.debug(f"Robot Path Updated: {len(path_x)} points")

            # Update robot marker
            robot_marker.set_data([env.x], [env.y])
            logging.debug(f"Robot Marker Updated: (x={env.x:.4f}, y={env.y:.4f})")

            # Update robot's orientation arrow
            new_arrow_end = (env.x + 0.5 * math.cos(env.theta),
                             env.y + 0.5 * math.sin(env.theta))
            robot_arrow.set_positions((env.x, env.y), new_arrow_end)
            logging.debug(f"Robot Arrow Updated: End Position = ({new_arrow_end[0]:.4f}, {new_arrow_end[1]:.4f})")

            # Log current state
            logging.info(
                f"Time: {elapsed_time:.2f}s, Position: ({env.x:.4f}, {env.y:.4f}), Theta: {math.degrees(env.theta):.2f}°")
            logging.debug(f"Desired Pose: (x={desired_pose[0]:.4f}, y={desired_pose[1]:.4f}, "
                          f"theta={math.degrees(desired_pose[2]):.2f}°), Desired Velocity: v={desired_velocity[0]:.4f} m/s, "
                          f"omega={math.degrees(desired_velocity[1]):.2f}°/s")
            logging.debug(f"Control Commands - v: {v:.4f} m/s, omega: {math.degrees(omega):.2f}°/s")
        except Exception as e:
            logging.error(f"Exception in update function at frame {frame}: {e}")

        # Return updated artists
        return robot_marker, path_line, robot_arrow

    # Create animation with frames as an iterable
    ani = FuncAnimation(fig, update, frames=range(total_frames),
                        interval=dt * 1000, repeat=False, blit=False)

    logging.info("Starting simulation animation.")
    plt.show()

    # Log final state
    final_pose = env.get_current_pose()
    logging.info(f"Simulation completed.")
    logging.info(f"Final Position: ({final_pose[0]:.4f}, {final_pose[1]:.4f}), Theta: {math.degrees(final_pose[2]):.2f}°")
    logging.info(f"Trajectory followed with {len(trajectory.points)} points.")

if __name__ == "__main__":
    main()
