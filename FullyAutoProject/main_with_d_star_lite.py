# main_with_d_star_lite.py

import argparse
import logging
from multiprocessing import Process, Manager, shared_memory, Event
import time
import struct
import math
from typing import Tuple, List

import numpy as np
import torch
import os
import warnings
import traceback
from scipy.interpolate import CubicSpline  # Optional: For spline-based trajectories

from PathFollowingAlgorithms.RamseteController import RamseteController
from PathFollowingAlgorithms.RamseteUtils.Trajectory import Trajectory
from RectangleFittingReceiver import visualization_main  # Ensure correct spelling and existence
from OptimizedDEVRLEnvIntake import env  # Ensure this is your environment class/module
from PathPlanningAlgorithms.DStarLite import DStarLite  # Import your DStarLite implementation
from PathPlanningAlgorithms.DStarLiteUtils.OccupancyGridMap import OccupancyGridMap  # Ensure correct path
from PathPlanningAlgorithms.DStarLiteUtils.PriorityQueue import PriorityQueue, Priority  # Import necessary classes
from PathPlanningAlgorithms.DStarLiteUtils.utils import Vertex, Vertices  # Import Vertex and Vertices classes


def convert_pos(pos):
    """
    Converts a position from a Tensor or list to a tuple of integers.

    Parameters:
    - pos: A Tensor, list, or tuple containing two numeric values.

    Returns:
    - A tuple of two integers representing the (x, y) position.
    """
    if isinstance(pos, torch.Tensor):
        if pos.dim() == 1 and pos.numel() == 2:
            converted = (int(round(pos[0].item())), int(round(pos[1].item())))
            logging.debug(f"convert_pos: Tensor input {pos.tolist()} converted to {converted}")
            return converted
        else:
            raise ValueError(f"Expected a 1D Tensor with 2 elements, got shape {pos.shape}")
    elif isinstance(pos, (list, tuple, np.ndarray)):
        if len(pos) == 2:
            converted = (int(round(pos[0])), int(round(pos[1])))
            logging.debug(f"convert_pos: list/tuple/ndarray input {pos} converted to {converted}")
            return converted
        else:
            raise ValueError(f"Expected a list/tuple/ndarray with 2 elements, got {len(pos)} elements")
    else:
        raise TypeError(f"Unsupported type for position conversion: {type(pos)}")


def inflate_segments(obstacle_segments, robot_radius):
    """
    Inflate line segments outward by the robot's radius.

    Parameters:
    - obstacle_segments (torch.Tensor): Shape (M, 2, 2)
    - robot_radius (float): Radius to inflate the segments.

    Returns:
    - inflated_segments (torch.Tensor): Shape (M', 2, 2)
    """
    # Convert to numpy for easier manipulation
    segments = obstacle_segments.cpu().numpy()
    inflated = []

    for seg in segments:
        (x1, y1), (x2, y2) = seg
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue  # Skip zero-length segments

        # Compute the normal vector
        nx = -dy / length
        ny = dx / length

        # Inflate both sides
        offset_x = nx * robot_radius
        offset_y = ny * robot_radius

        inflated_seg1 = ((x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y))
        inflated_seg2 = ((x1 - offset_x, y1 - offset_y), (x2 - offset_x, y2 - offset_y))

        inflated.append(inflated_seg1)
        inflated.append(inflated_seg2)

    if not inflated:
        return torch.empty((0, 2, 2), dtype=torch.float32, device=obstacle_segments.device)

    inflated_segments = torch.tensor(inflated, dtype=torch.float32, device=obstacle_segments.device)
    return inflated_segments


def run_environment(lock, action_queue, env_info_queue, termination_event):
    """
    Initializes and runs the environment.

    Parameters:
    - lock: A multiprocessing lock for synchronization.
    - action_queue: A multiprocessing queue for receiving actions.
    - env_info_queue: A multiprocessing queue for sending environment info.
    - termination_event: An event to signal termination.
    """
    # Initialize logging for the environment process
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO to exclude DEBUG logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("env_process.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize the environment
    env_instance = env(render_mode="human", max_teleop_time=1000000, lock=lock)
    obs, info = env_instance.reset()

    try:
        while not termination_event.is_set():
            # Receive action
            if not action_queue.empty():
                action = action_queue.get()
                # Action tuple: (vx, vy, omega)
            else:
                action = (0.0, 0.0, 0.0)  # Default action (no movement)

            # Apply action to the environment
            obs, reward, terminated, truncated, info = env_instance.step(action, testing_mode=False)
            env_instance.render()

            # Send environment info to the main process
            env_info = env_instance.get_env_info()
            env_info_queue.put(env_info)

            if terminated or truncated:
                obs, info = env_instance.reset()
    except KeyboardInterrupt:
        logging.info("Environment process interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in environment process: {e}")
        logging.error(traceback.format_exc())
    finally:
        env_instance.close()
        logging.info("Environment process terminated gracefully.")
        print("Environment ended.")


def run_visualization(shm_name, shm_size, lock, destination_queue, obstacle_queue, path_queue, termination_event):
    """
    Runs the visualization process.

    Parameters:
    - shm_name: Name of the shared memory segment.
    - shm_size: Size of the shared memory segment.
    - lock: A multiprocessing lock for synchronization.
    - destination_queue: Queue for sending destination points to the main script.
    - obstacle_queue: Queue for sending detected obstacles to the main script.
    - path_queue: Queue for receiving the path from the main script.
    - termination_event: An event to signal termination.
    """
    visualization_main(
        env_shm_name=shm_name,
        shm_size=shm_size,
        lock=lock,
        destination_queue=destination_queue,
        obstacle_queue=obstacle_queue,
        path_queue=path_queue,
        termination_event=termination_event
    )


def compute_action_to_point(robot_x, robot_y, robot_theta, next_point):
    """
    Compute the action to move the swerve robot towards the next point.

    Parameters:
    - robot_x (float): Current x-position of the robot.
    - robot_y (float): Current y-position of the robot.
    - robot_theta (float): Current orientation of the robot in radians.
    - next_point (tuple): Tuple (x, y) representing the next target point.

    Returns:
    - action (tuple): Tuple (vx, vy, omega) where:
        - vx (float): Velocity along the x-axis.
        - vy (float): Velocity along the y-axis.
        - omega (float): Angular velocity (rotation rate).
    """
    # Compute the difference between the current position and the target point
    dx = next_point[0] - robot_x
    dy = next_point[1] - robot_y

    # Proportional controller gains
    k_v = 4.0  # Gain for velocity control (Adjusted for smoother movement)
    k_omega = 0.1  # Gain for angular velocity control

    # Compute desired velocities based on the difference
    vx = k_v * dx
    vy = k_v * dy

    # Limit velocities to maximum values to prevent abrupt movements
    max_v = 1.0  # Maximum velocity, adjust as needed
    vx = max(-max_v, min(max_v, vx))
    vy = max(-max_v, min(max_v, vy))

    # Compute desired orientation to face the target point (optional)
    target_angle = math.atan2(dy, dx)
    angle_diff = target_angle - robot_theta

    # Normalize the angle difference to the range [-pi, pi]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Determine if the robot needs to adjust its orientation
    orientation_threshold = math.radians(10)  # Threshold in radians
    if abs(angle_diff) > orientation_threshold:
        # Apply a small angular velocity to align with the target direction
        omega = k_omega * angle_diff
        # Limit angular velocity
        max_omega = 0.5  # Maximum angular velocity, adjust as needed
        omega = max(-max_omega, min(max_omega, omega))
    else:
        # Minimal or no rotation needed
        omega = 0.0

    # Construct the action tuple
    action = (vx, vy, omega)
    return action


def get_robot_state_from_shared_memory(shm, shm_size):
    """
    Get the robot's position and orientation from shared memory.

    Returns:
    - robot_x, robot_y, robot_theta_rad
    """
    buffer = shm.buf[:shm_size]
    offset = 0
    robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
    offset += struct.calcsize('ff')
    robot_theta_rad, = struct.unpack_from('f', buffer, offset)
    return robot_x, robot_y, robot_theta_rad


def str2bool(v):
    '''Fix the bool BUG for argparse: transfer string to bool'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1', 'T'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0', 'F'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate_spline_trajectory(trajectory, occupancy_map, num_samples=500) -> Tuple[List[Tuple[float, float, float]], bool]:
    """
    Generates a spline trajectory from the given trajectory points and checks for collisions.

    :param trajectory: Trajectory object generated from D*Lite path.
    :param occupancy_map: OccupancyGridMap object for collision checking.
    :param num_samples: Number of samples along the spline.
    :return: Tuple containing the spline path (list of (x, y, theta)) and a boolean indicating collision-free.
    """
    if not trajectory.points:
        logging.warning("No points available to generate spline.")
        return [], False

    x = [point.x for point in trajectory.points]
    y = [point.y for point in trajectory.points]
    time_pts = [point.time for point in trajectory.points]

    try:
        cs_x = CubicSpline(time_pts, x)
        cs_y = CubicSpline(time_pts, y)
        logging.info("Cubic spline generated successfully.")
    except Exception as e:
        logging.error(f"Failed to generate spline: {e}")
        logging.error(traceback.format_exc())
        return [], False

    # Define time samples for collision checking
    time_samples = np.linspace(time_pts[0], time_pts[-1], num=num_samples)
    spline_path = []
    collision_free = True

    for t in time_samples:
        x_spline = cs_x(t)
        y_spline = cs_y(t)
        theta_spline = math.atan2(cs_y.derivative()(t), cs_x.derivative()(t))
        pos = (int(round(x_spline)), int(round(y_spline)))

        if not occupancy_map.is_unoccupied(pos):
            logging.warning(f"Spline trajectory collides at position: {pos}")
            collision_free = False
            break

        spline_path.append((x_spline, y_spline, theta_spline))

    if collision_free:
        logging.info("Spline trajectory is collision-free.")
    else:
        logging.warning("Spline trajectory collides with obstacles.")

    return spline_path, collision_free


def run_main():
    """Main function to run the Rectangle Fitting Simulation."""
    # Configurations using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dvc', type=str, default='cuda', help='Running device of SEPSO_Down, cuda or cpu')
    parser.add_argument('--DPI', type=str2bool, default=True,
                        help='True for DPI(from OkayPlan), False for PI(from SEPSO)')
    parser.add_argument('--KP', type=str2bool, default=True, help='whether to use Kinematics_Penalty')
    parser.add_argument('--FPS', type=int, default=0, help='Render FPS, 0 for maximum speed')
    parser.add_argument('--Playmode', type=str2bool, default=False, help='Play with keyboard: UP, DOWN, LEFT, RIGHT')

    # Planner related:
    parser.add_argument('--Max_iterations', type=int, default=100,
                        help='maximum number of particle iterations for each planning')
    parser.add_argument('--N', type=int, default=300, help='number of particles in each group')
    parser.add_argument('--D', type=int, default=20, help='particle dimension: number of waypoints = D/2')
    parser.add_argument('--Quality', type=float, default=10,
                        help='planning quality: the smaller, the better quality, and the longer time')
    parser.add_argument('--robot_radius', type=float, default=1, help='Radius of the robot for collision avoidance')

    # Env related:
    parser.add_argument('--window_size', type=int, default=366, help='render window size, minimal: 366')
    parser.add_argument('--Start_V', type=int, default=6, help='velocity of the robot')
    opt = parser.parse_args()
    opt.dvc = torch.device(opt.dvc if torch.cuda.is_available() else 'cpu')
    opt.Search_range = (0., opt.window_size)
    if opt.window_size < 366:
        opt.window_size = 366  # 366 is the minimal window size
        print("\nThe minimal window size should be at least 366.\n")
    if opt.Playmode:
        print("\nUse UP/DOWN/LEFT/RIGHT to control the target point.\n")

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO to exclude DEBUG logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("robot_navigation.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize multiprocessing manager and queues
    manager = Manager()
    lock = manager.Lock()
    destination_queue = manager.Queue()
    obstacle_queue = manager.Queue()
    action_queue = manager.Queue()
    path_queue = manager.Queue()
    env_info_queue = manager.Queue()
    termination_event = manager.Event()

    # Define shared memory sizes based on environment's calculation
    num_cameras = 2
    rays_per_camera = 255  # Adjusted to match shm_size=12288
    num_robots = 1

    # Each robot has 3 floats (12 bytes)
    # Each camera has:
    # - 2 floats for camera_pos (8 bytes)
    # - 2 floats for look_at_pos (8 bytes)
    # - 1 int for number_of_rays (4 bytes)
    # - Each ray has 5 fields: 3 floats, 1 int, 1 float (total 24 bytes)
    shm_size = 12288  # Adjusted to match the environment process

    logging.info(f"Calculated shared memory size: {shm_size} bytes")

    shm_name = 'my_shared_memory'

    # Create or connect to shared memory
    try:
        shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        logging.info(f"Shared memory created with name: {shm.name}, size: {shm.size} bytes")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=shm_name)
        logging.info(f"Shared memory connected with name: {shm.name}, size: {shm.size} bytes")
    except Exception as e:
        logging.error(f"Failed to create shared memory: {e}")
        exit(1)

    # Initialize D*-Lite
    # For initial start and goal, set default positions; these can be updated later
    initial_start = (0, 0)  # Placeholder; will be updated from shared memory
    initial_goal = (15, 7)  # Example goal position; adjust as needed

    # Initialize Occupancy Grid Map
    occupancy_map = OccupancyGridMap(x_dim=opt.window_size, y_dim=opt.window_size, exploration_setting='8N')

    # Initialize D*-Lite planner
    dstar = DStarLite(map=occupancy_map, s_start=initial_start, s_goal=initial_goal)

    # Start the Environment process
    env_process = Process(target=run_environment, args=(lock, action_queue, env_info_queue, termination_event))
    env_process.start()
    logging.info("Environment process started.")

    # Give the Environment some time to initialize and write to shared memory
    time.sleep(1)

    # Start the Visualization process
    logging.info("Starting visualization process")
    visualization_process = Process(
        target=run_visualization,
        args=(shm_name, shm_size, lock, destination_queue, obstacle_queue, path_queue, termination_event)
    )
    visualization_process.start()
    logging.info("Visualization process started.")

    print(f'CUDA is available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.current_device()}')

    # Initialize trajectory-related variables
    current_path = []
    path_index = 0
    replanning_interval = 0.5  # seconds
    last_replanning_time = 0

    # Initialize current_destination
    current_destination = None  # Added initialization to prevent UnboundLocalError
    trajectory = None
    ramsete_controller = None

    try:
        while True:
            current_time = time.time()

            # Initialize 'collide' and 'arrive' with default values at the start of each iteration
            collide = False
            arrive = False

            # Handle new destinations from destination_queue
            while not destination_queue.empty():
                new_destination = destination_queue.get()
                if new_destination:
                    # Convert the new destination to integer grid coordinates
                    try:
                        current_destination = convert_pos(new_destination)
                        logging.info(f"New destination set via mouse click: {current_destination}")
                        # Assert to ensure conversion was successful
                        assert isinstance(current_destination, tuple) and len(current_destination) == 2 and all(
                            isinstance(c, int) for c in
                            current_destination), "Destination must be a tuple of two integers."
                    except Exception as e:
                        logging.error(f"Failed to convert new destination: {e}")
                        continue  # Skip setting the destination if conversion fails

                    # Update the goal in D*-Lite
                    try:
                        dstar.s_goal = current_destination
                        dstar.rhs[dstar.s_goal] = 0
                        dstar.U.insert(dstar.s_goal, dstar.calculate_key(dstar.s_goal))
                        dstar.compute_shortest_path()
                        current_path, g, rhs = dstar.move_and_replan(dstar.s_start)  # Get new path
                        path_index = 0

                        # Generate a trajectory from the current_path
                        trajectory = Trajectory.from_waypoints(current_path, initial_time=current_time, dt=0.05,
                                                               max_v=1.0)
                        logging.info(f"Generated trajectory with {len(trajectory.points)} points.")

                        # Initialize the Ramsete Controller
                        ramsete_controller = RamseteController(b=2.0, zeta=0.7)

                        # Send the D*Lite path to visualization with path type identifier
                        path_queue.put(('dstar', current_path))
                        logging.info(f"Generated new D*Lite path: {current_path}")

                        # Generate spline trajectory
                        spline_path, is_collision_free = generate_spline_trajectory(trajectory, occupancy_map, num_samples=500)

                        if is_collision_free:
                            # Send the spline path to visualization with path type identifier
                            path_queue.put(('ramsete', spline_path))
                            logging.info("Spline trajectory sent to visualization.")
                        else:
                            logging.warning("Spline trajectory collides with obstacles. Using original D*Lite path.")

                        # Pause until Enter is pressed
                        logging.info("Trajectory and Ramsete trajectory generated. Pausing execution.")
                        input("Path and Ramsete trajectory generated. Press Enter to start following the trajectory...")
                        logging.info("User pressed Enter. Starting trajectory following.")

                        # Update the last replanning time
                        last_replanning_time = current_time
                    except Exception as e:
                        logging.error(f"Failed to update D*-Lite with new goal: {e}")
                        logging.error(traceback.format_exc())

            # Get environment info from env_info_queue
            if not env_info_queue.empty():
                env_info_received = env_info_queue.get()
                # Extract necessary information
                # Ignore 'start_point' and 'target_point'
                obstacles = env_info_received.get('Obs_Segments', [])
                dynamic_segments = env_info_received.get('Flat_pdct_segments', [])
                arrive = env_info_received.get('Arrive', False)
                collide = env_info_received.get('Collide', False)

                # Temporary logging
                logging.info(
                    f"Received {len(obstacles)} static obstacles and {len(dynamic_segments)} dynamic obstacles.")

                # Initialize a Vertices object to hold changed edges
                changed_vertices = Vertices()

                # Update static obstacles
                for obstacle in obstacles:
                    # Each obstacle is assumed to be a Tensor with shape (2, 2) representing a line segment
                    for point in obstacle:
                        try:
                            pos = convert_pos(point)
                            logging.info(f"Processing obstacle at position: {pos}")
                            # Assert to ensure conversion was successful
                            assert isinstance(pos, tuple) and len(pos) == 2 and all(
                                isinstance(c, int) for c in pos), "Obstacle position must be a tuple of two integers."
                        except Exception as e:
                            logging.error(f"Failed to convert obstacle position: {e}")
                            continue  # Skip this point and continue with others

                        if occupancy_map.is_unoccupied(pos):
                            occupancy_map.set_obstacle(pos)
                            logging.info(f"Set obstacle at position: {pos}")
                            # Create a Vertex for the obstacle
                            v = Vertex(pos=pos)
                            # Get successors (neighbors) of the obstacle
                            succ = occupancy_map.succ(pos)
                            for u in succ:
                                cost = dstar.c(u, pos)
                                v.add_edge_with_cost(succ=u, cost=cost)
                            changed_vertices.add_vertex(v)

                # Update dynamic segments
                for segment in dynamic_segments:
                    for point in segment:
                        try:
                            pos = convert_pos(point)
                            logging.info(f"Processing dynamic obstacle at position: {pos}")
                            # Assert to ensure conversion was successful
                            assert isinstance(pos, tuple) and len(pos) == 2 and all(isinstance(c, int) for c in
                                                                                    pos), "Dynamic obstacle position must be a tuple of two integers."
                        except Exception as e:
                            logging.error(f"Failed to convert dynamic obstacle position: {e}")
                            continue  # Skip this point and continue with others

                        # Assuming dynamic_segments represent obstacles to be set
                        if occupancy_map.is_unoccupied(pos):
                            occupancy_map.set_obstacle(pos)
                            logging.info(f"Set dynamic obstacle at position: {pos}")
                            v = Vertex(pos=pos)
                            succ = occupancy_map.succ(pos)
                            for u in succ:
                                cost = dstar.c(u, pos)
                                v.add_edge_with_cost(succ=u, cost=cost)
                            changed_vertices.add_vertex(v)
                        else:
                            # If dynamic_segments represent obstacles to be removed
                            occupancy_map.remove_obstacle(pos)
                            logging.info(f"Removed dynamic obstacle at position: {pos}")
                            # Optionally, handle the removal in DStarLite
                            v = Vertex(pos=pos)
                            succ = occupancy_map.succ(pos)
                            for u in succ:
                                cost = dstar.c(u, pos)
                                v.add_edge_with_cost(succ=u, cost=cost)
                            changed_vertices.add_vertex(v)

                # Assign the changed edges to DStarLite
                dstar.new_edges_and_old_costs = changed_vertices

            # Check for collision or arrival
            if collide:
                logging.warning("Robot has collided with an obstacle!")
                # Handle collision (e.g., stop the robot, reset, etc.)
                action = (0.0, 0.0, 0.0)
                action_queue.put(action)
                current_destination = None  # Reset destination
                current_path = []
                trajectory = None
                ramsete_controller = None
                path_index = 0
                logging.info("Trajectory and controller reset due to collision.")
                continue  # Skip further processing in this loop

            if arrive:
                logging.info("Robot has arrived at the destination!")
                # Handle arrival (e.g., stop the robot, await new destination, etc.)
                action = (0.0, 0.0, 0.0)
                action_queue.put(action)
                current_destination = None  # Reset destination
                current_path = []
                trajectory = None
                ramsete_controller = None
                path_index = 0
                logging.info("Trajectory and controller reset due to arrival.")
                continue  # Skip further processing in this loop

            # Determine if it's time to replan
            if (current_time - last_replanning_time) >= replanning_interval:
                if current_destination and dstar is not None and trajectory is not None:
                    # Get the current robot position again
                    with lock:
                        robot_x, robot_y, robot_theta_rad = get_robot_state_from_shared_memory(shm, shm_size)
                        logging.info(f"Robot current position: ({robot_x}, {robot_y}, {robot_theta_rad})")

                    # Update the start position in D*-Lite
                    dstar.s_start = (int(round(robot_x)), int(round(robot_y)))  # Ensure integers
                    logging.info(f"Updated D*-Lite start position to: {dstar.s_start}")

                    # Replan the path
                    try:
                        current_path, g, rhs = dstar.move_and_replan(dstar.s_start)
                        path_index = 0

                        # Generate a new trajectory from the updated path
                        trajectory = Trajectory.from_waypoints(current_path, initial_time=current_time, dt=0.05,
                                                               max_v=1.0)
                        logging.info(f"Replanned trajectory with {len(trajectory.points)} points.")

                        # Re-initialize the Ramsete Controller
                        ramsete_controller = RamseteController(b=2.0, zeta=0.7)

                        # Send the new path to visualization with path type identifier
                        path_queue.put(('dstar', current_path))
                        logging.info(f"Replanned path: {current_path}")

                        # Generate spline trajectory
                        spline_path, is_collision_free = generate_spline_trajectory(trajectory, occupancy_map, num_samples=500)

                        if is_collision_free:
                            # Send the spline path to visualization with path type identifier
                            path_queue.put(('ramsete', spline_path))
                            logging.info("Replanned spline trajectory sent to visualization.")
                        else:
                            logging.warning("Replanned spline trajectory collides with obstacles. Using original D*Lite path.")

                        # Pause until Enter is pressed
                        logging.info("Replanned trajectory and Ramsete trajectory generated. Pausing execution.")
                        input("Replanned path and Ramsete trajectory generated. Press Enter to continue...")
                        logging.info("User pressed Enter. Continuing trajectory following.")

                        # Update the last replanning time
                        last_replanning_time = current_time
                    except Exception as e:
                        logging.error(f"Failed during replanning: {e}")
                        logging.error(traceback.format_exc())

            # Compute and send action to the environment using Ramsete Controller
            if trajectory and ramsete_controller:
                desired_pose, desired_velocity = trajectory.get_desired_state(current_time)
                if desired_pose is not None:
                    # Get the robot's current state
                    with lock:
                        robot_x, robot_y, robot_theta_rad = get_robot_state_from_shared_memory(shm, shm_size)
                        logging.info(f"Robot current state: ({robot_x}, {robot_y}, {robot_theta_rad})")

                    # Compute control commands using Ramsete Controller
                    try:
                        v, omega = ramsete_controller.compute_control(
                            current_pose=(robot_x, robot_y, robot_theta_rad),
                            desired_pose=desired_pose,
                            desired_velocity=desired_velocity
                        )
                        logging.info(f"Ramsete Controller Output - v: {v:.2f}, omega: {omega:.2f}")

                        # For swerve drive, compute vy based on desired_pose and current theta
                        # Alternatively, if vy is to be controlled, modify the controller to output vy
                        # Here, we set vy to 0.0 and rely on the robot's orientation control
                        vy = 0.0

                        # Construct the action tuple
                        action = (v, vy, omega)
                        action_queue.put(action)
                        logging.debug(f"Action sent to environment: {action}")
                    except Exception as e:
                        logging.error(f"Failed to compute Ramsete control: {e}")
                        logging.error(traceback.format_exc())
                else:
                    # Trajectory has ended; stop the robot
                    action = (0.0, 0.0, 0.0)
                    action_queue.put(action)
                    logging.info("Trajectory completed. Stopping the robot.")
            else:
                # No trajectory; send zero velocities
                action = (0.0, 0.0, 0.0)
                action_queue.put(action)
                logging.debug("No active trajectory. Robot is staying in place.")

            # Sleep briefly to control the loop rate
            time.sleep(0.05)  # Reduced sleep for higher responsiveness
    except KeyboardInterrupt:
        logging.info("Main process interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Signal termination to child processes
        termination_event.set()

        # Wait for child processes to terminate gracefully
        env_process.join()
        visualization_process.join()

        # Clean up shared memory
        try:
            shm.close()
            shm.unlink()
            logging.info("Main process: 'my_shared_memory' shared memory unlinked.")
        except FileNotFoundError:
            logging.warning("Main process: 'my_shared_memory' shared memory not found for unlinking.")
        except Exception as e:
            logging.error(f"Error during shared memory cleanup: {e}")

        # Plot the trajectory
        if ramsete_controller:
            ramsete_controller.plot_trajectory()

        print("All processes ended.")


if __name__ == "__main__":
    run_main()
