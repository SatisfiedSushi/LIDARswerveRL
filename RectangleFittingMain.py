# RectangleFittingMain.py

import argparse
import logging
from multiprocessing import Process, Manager, shared_memory, Event
import time
import struct
import math
import numpy as np
import torch
import os
import warnings

# Removed pygame imports as per user instruction
# import pygame

# Corrected import spelling
from OkayPlan import OkayPlan  # Ensure OkayPlan.py is in the same directory
from RectangleFittingReveiver import visualization_main  # Corrected spelling
from OptimizedDEVRLEnvIntake import env  # Ensure this is your environment class/module


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

    # Load OkayPlan parameters with weights_only=True to address FutureWarning
    try:
        params = torch.load('Relax0.4_S0_2023-09-23_21_38.pt', map_location=opt.dvc, weights_only=True)[-1]
    except TypeError:
        # Handle cases where 'weights_only' might not be supported
        params = torch.load('Relax0.4_S0_2023-09-23_21_38.pt', map_location=opt.dvc)[-1]
        logging.warning("weights_only=True not supported for torch.load. Loaded with default settings.")
    except FileNotFoundError:
        logging.error("Parameter file 'Relax0.4_S0_2023-09-23_21_38.pt' not found.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        exit(1)

    # Adjust parameters based on KP flag
    if not opt.KP:
        if len(params) > 50:
            params[50] = 0
            logging.info("Kinematics_Penalty disabled by setting parameter 50 to 0.")
        else:
            logging.warning("Parameter index 50 out of range. Skipping KP adjustment.")

    # Initialize OkayPlan
    okayplan = None  # Initialize OkayPlan later once we have a destination

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

    # Initialize path-related variables
    current_path = []
    path_index = 0
    replanning_interval = 0.5  # seconds
    last_replanning_time = 0

    # Initialize current_destination
    current_destination = None  # Added initialization to prevent UnboundLocalError

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
                    current_destination = new_destination
                    logging.info(f"New destination set via mouse click: {current_destination}")
                    okayplan = OkayPlan(opt, params)  # Reset OkayPlan with new destination
                    # Assuming start_point should be the current robot position
                    with lock:
                        robot_x, robot_y, _ = get_robot_state_from_shared_memory(shm, shm_size)
                    okayplan.Priori_Path_Init((robot_x, robot_y), current_destination)  # Initialize Priori_Path

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
                logging.info(f"Received {len(obstacles)} static obstacles and {len(dynamic_segments)} dynamic obstacles.")

            # Check for collision or arrival
            if collide:
                logging.warning("Robot has collided with an obstacle!")
                # Handle collision (e.g., stop the robot, reset, etc.)
                action = (0.0, 0.0, 0.0)
                action_queue.put(action)
                current_destination = None  # Reset destination
                current_path = []
                path_index = 0
                continue  # Skip further processing in this loop

            if arrive:
                logging.info("Robot has arrived at the destination!")
                # Handle arrival (e.g., stop the robot, await new destination, etc.)
                action = (0.0, 0.0, 0.0)
                action_queue.put(action)
                current_destination = None  # Reset destination
                current_path = []
                path_index = 0
                continue  # Skip further processing in this loop

            # Determine if it's time to replan
            if (current_time - last_replanning_time) >= replanning_interval:
                if current_destination and okayplan is not None:
                    # Get the current robot position again
                    with lock:
                        robot_x, robot_y, robot_theta_rad = get_robot_state_from_shared_memory(shm, shm_size)

                    # Inflate obstacles to account for robot's thickness
                    robot_radius = opt.robot_radius

                    # Convert obstacles to tensor and ensure they are on the correct device
                    if isinstance(obstacles, np.ndarray):
                        obstacles_tensor = torch.tensor(obstacles, dtype=torch.float32, device=opt.dvc)
                    elif isinstance(obstacles, torch.Tensor):
                        obstacles_tensor = obstacles.to(opt.dvc)
                    else:
                        obstacles_tensor = torch.tensor(obstacles, dtype=torch.float32, device=opt.dvc)

                    # Inflate static obstacles
                    inflated_obstacles = inflate_segments(obstacles_tensor, robot_radius)

                    # Similarly, convert and move dynamic segments to the correct device
                    if isinstance(dynamic_segments, np.ndarray):
                        dynamic_segments_tensor = torch.tensor(dynamic_segments, dtype=torch.float32, device=opt.dvc)
                    elif isinstance(dynamic_segments, torch.Tensor):
                        dynamic_segments_tensor = dynamic_segments.to(opt.dvc)
                    else:
                        dynamic_segments_tensor = torch.tensor(dynamic_segments, dtype=torch.float32, device=opt.dvc)

                    # Inflate dynamic obstacles
                    inflated_dynamic_segments = inflate_segments(dynamic_segments_tensor, robot_radius)

                    # Create environment info for OkayPlan
                    env_info_for_planner = {
                        'start_point': (robot_x, robot_y),
                        'target_point': current_destination,
                        'd2target': math.hypot(current_destination[0] - robot_x, current_destination[1] - robot_y),
                        'Obs_Segments': inflated_obstacles,  # Inflated static obstacles including L-Shape Fitting obstacles
                        'Flat_pdct_segments': inflated_dynamic_segments,  # Inflated dynamic obstacles
                        'Arrive': arrive,
                        'Collide': collide,
                    }

                    # Forward inflated obstacles to visualization
                    # Convert torch tensors to lists for serialization
                    static_obstacles_list = inflated_obstacles.cpu().numpy().tolist()
                    dynamic_obstacles_list = inflated_dynamic_segments.cpu().numpy().tolist()

                    # Send obstacles to visualization
                    obstacle_queue.put({
                        'static_obstacles': static_obstacles_list,
                        'dynamic_obstacles': dynamic_obstacles_list
                    })

                    # Generate path using OkayPlan
                    try:
                        path_tensor, is_collision_free = okayplan.plan(env_info_for_planner)
                        planned_path = path_tensor.cpu().numpy()

                        # Extract waypoints from the planned path
                        NP = okayplan.NP
                        if NP * 2 > len(planned_path):
                            logging.error("Planned path length is insufficient for the number of waypoints.")
                            current_path = []
                        else:
                            new_path = [(planned_path[i], planned_path[i + NP]) for i in range(NP)]
                            # Remove the first point (robot's current position) if present
                            if new_path and math.isclose(new_path[0][0], robot_x, abs_tol=1e-3) and math.isclose(new_path[0][1],
                                                                                                                robot_y,
                                                                                                                abs_tol=1e-3):
                                new_path = new_path[1:]

                            current_path = new_path
                            path_index = 0

                            logging.info(f"Generated new path: {current_path}")

                            # Send the new path to visualization
                            path_queue.put(current_path)

                    except AttributeError as e:
                        logging.error(f"Path planning failed: {e}")
                        current_path = []
                    except Exception as e:
                        logging.error(f"Unexpected error during path planning: {e}")
                        current_path = []

                    # Update the last replanning time
                    last_replanning_time = current_time

            # Compute and send action to the environment
            if current_path and path_index < len(current_path):
                next_point = current_path[path_index]

                # Check if reached the next point
                distance_to_point = math.hypot(next_point[0] - robot_x, next_point[1] - robot_y)
                # Removed debug logging
                # logging.info(f"Distance to next point: {distance_to_point:.2f}")

                if distance_to_point < 0.2:  # Threshold distance to consider point reached
                    # Removed logging of reaching waypoints
                    # logging.info(f"Reached waypoint {path_index + 1}/{len(current_path)}: {next_point}")

                    # Check if there is another point in the path
                    if path_index + 1 < len(current_path):
                        path_index += 1
                        next_point = current_path[path_index]
                        # Removed logging of moving to next waypoint
                        # logging.info(f"Moving to waypoint {path_index + 1}/{len(current_path)}: {next_point}")
                    else:
                        # Reached the final point
                        # logging.info("Reached the final waypoint. Stopping the robot.")
                        action = (0.0, 0.0, 0.0)  # Stop the robot
                        action_queue.put(action)
                        current_path = []  # Clear the path
                        continue  # Skip action computation for the final point

                # Compute action towards the current next_point
                action = compute_action_to_point(robot_x, robot_y, robot_theta_rad, next_point)
                action_queue.put(action)
            else:
                # No path or reached the end
                action = (0.0, 0.0, 0.0)  # Stay in place
                action_queue.put(action)

            # Sleep briefly to control the loop rate
            time.sleep(0.05)  # Reduced sleep for higher responsiveness
    except KeyboardInterrupt:
        logging.info("Main process interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
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

        print("All processes ended.")


if __name__ == "__main__":
    run_main()
