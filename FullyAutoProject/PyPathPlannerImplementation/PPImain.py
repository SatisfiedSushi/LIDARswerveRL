import argparse
import logging
from multiprocessing import Process, Manager, shared_memory
import time
import struct
import math
import numpy as np
import torch

from wpimath.geometry import Pose2d, Translation2d, Rotation2d
from wpimath.controller import PIDController

from FullyAutoProject.OptimizedDEVRLEnvIntake import env
from FullyAutoProject.PyPathPlannerImplementation.config import PIDConstants
from FullyAutoProject.PyPathPlannerImplementation.path import PathConstraints, GoalEndState
from FullyAutoProject.RectangleFittingReceiver import visualization_main, get_robot_state_from_shared_memory
from pathfinding import Pathfinding
from controller import PPHolonomicDriveController  # Advanced path following
from telemetry import PPLibTelemetry
from util import DriveFeedforwards

def inflate_segments(obstacle_segments, robot_radius):
    """
    Inflate line segments outward by the robot's radius.

    Parameters:
    - obstacle_segments (torch.Tensor): Shape (M, 2, 2)
    - robot_radius (float): Radius to inflate the segments.

    Returns:
    - inflated_segments (torch.Tensor): Shape (M', 2, 2)
    """
    segments = obstacle_segments.cpu().numpy()
    inflated = []

    for seg in segments:
        (x1, y1), (x2, y2) = seg
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue  # Skip zero-length segments

        nx = -dy / length
        ny = dx / length

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
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("../env_process.log"),
            logging.StreamHandler()
        ]
    )

    env_instance = env(render_mode="human", max_teleop_time=1000000, lock=lock)
    obs, info = env_instance.reset()

    try:
        while not termination_event.is_set():
            if not action_queue.empty():
                action = action_queue.get()
            else:
                action = (0.0, 0.0, 0.0)

            obs, reward, terminated, truncated, info = env_instance.step(action, testing_mode=False)
            env_instance.render()

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
    visualization_main(
        env_shm_name=shm_name,
        shm_size=shm_size,
        lock=lock,
        destination_queue=destination_queue,
        obstacle_queue=obstacle_queue,
        path_queue=path_queue,
        termination_event=termination_event
    )

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_radius', type=float, default=1, help='Radius of the robot for collision avoidance')
    opt = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("../robot_navigation.log"),
            logging.StreamHandler()
        ]
    )

    manager = Manager()
    lock = manager.Lock()
    destination_queue = manager.Queue()
    obstacle_queue = manager.Queue()
    action_queue = manager.Queue()
    path_queue = manager.Queue()
    env_info_queue = manager.Queue()
    termination_event = manager.Event()

    shm_size = 12288
    shm_name = 'my_shared_memory'

    try:
        shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        logging.info(f"Shared memory created with name: {shm.name}, size: {shm.size} bytes")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=shm_name)
        logging.info(f"Shared memory connected with name: {shm.name}, size: {shm.size} bytes")
    except Exception as e:
        logging.error(f"Failed to create shared memory: {e}")
        exit(1)

    Pathfinding.ensureInitialized()

    env_process = Process(target=run_environment, args=(lock, action_queue, env_info_queue, termination_event))
    visualization_process = Process(
        target=run_visualization,
        args=(shm_name, shm_size, lock, destination_queue, obstacle_queue, path_queue, termination_event)
    )

    env_process.start()
    visualization_process.start()

    translation_constants = PIDConstants(kP=1.0, kI=0.0, kD=0.0)
    rotation_constants = PIDConstants(kP=1.0, kI=0.0, kD=0.0)
    holonomic_controller = PPHolonomicDriveController(translation_constants, rotation_constants)

    current_path = []
    try:
        while not termination_event.is_set():
            if not destination_queue.empty():
                new_destination = destination_queue.get()
                if new_destination:
                    goal_translation = Translation2d(new_destination[0], new_destination[1])
                    Pathfinding.setGoalPosition(goal_translation)

            if not env_info_queue.empty():
                env_info = env_info_queue.get()
                dynamic_segments = env_info.get('Flat_pdct_segments', [])
                obstacles = [(Translation2d(x1, y1), Translation2d(x2, y2)) for (x1, y1), (x2, y2) in dynamic_segments]
                inflated_obstacles = inflate_segments(torch.tensor(obstacles), opt.robot_radius)
                Pathfinding.setDynamicObstacles(inflated_obstacles, Translation2d(0, 0))

            if Pathfinding.isNewPathAvailable():
                constraints = PathConstraints(maxVelocityMps=2.0, maxAccelerationMpsSq=2.0, maxAngularVelocityRps=1.0,
                                              maxAngularAccelerationRpsSq=1.0)
                goal_end_state = GoalEndState(velocity=0.0, rotation=Rotation2d(0))
                current_path = Pathfinding.getCurrentPath(constraints, goal_end_state).getPathPoses()
                path_as_tuples = [(pose.X(), pose.Y(), pose.rotation().radians()) for pose in current_path]
                path_queue.put(path_as_tuples)

            if current_path:
                for state in current_path:
                    robot_x, robot_y, robot_theta = get_robot_state_from_shared_memory(shm, shm_size)
                    current_pose = Pose2d(Translation2d(robot_x, robot_y), Rotation2d(robot_theta))
                    target_pose = Pose2d(Translation2d(state[0], state[1]), Rotation2d(state[2]))

                    commanded_speeds = holonomic_controller.calculateRobotRelativeSpeeds(current_pose, target_pose)
                    action_queue.put((commanded_speeds.vx, commanded_speeds.vy, commanded_speeds.omega))

                    PPLibTelemetry.setCurrentPose(current_pose)
                    PPLibTelemetry.setTargetPose(target_pose)
                    time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("Main process interrupted by user.")
    finally:
        termination_event.set()
        env_process.join()
        visualization_process.join()

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
