# test.py

import logging
import time
import struct
import math
from multiprocessing import Process, Manager, shared_memory
from slam_module import SLAMModule
from receiver import visualization_main
from OptimizedDEVRLEnvIntake import env

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_environment(lock):
    env_instance = env(render_mode="human", max_teleop_time=1000000, lock=lock)
    obs, info = env_instance.reset()
    try:
        obs, reward, terminated, truncated, info = env_instance.step((1.0, 0.0, 0.1), testing_mode=True)
        env_instance.render()
    except KeyboardInterrupt:
        logging.info("Environment interrupted by user.")
    finally:
        env_instance.close()
        print("Environment ended.")

def run_slam_module(lock, shm_input_name, shm_input_size, shm_output_name, shm_output_size):
    slam_module = SLAMModule(
        lock,
        shm_input_name=shm_input_name,
        shm_input_size=shm_input_size,
        shm_output_name=shm_output_name,
        shm_output_size=shm_output_size,
    )
    slam_module.run()

def run_visualization(shm_output_name, shm_output_size):
    visualization_main(shm_output_name=shm_output_name, shm_output_size=shm_output_size)

def verify_shared_memory(env_shm_name, slam_shm_name, shm_size):
    """
    Reads data from the environment and SLAM shared memory segments and verifies consistency.
    """
    try:
        # Connect to shared memory segments
        env_shm = shared_memory.SharedMemory(name=env_shm_name)
        slam_shm = shared_memory.SharedMemory(name=slam_shm_name)

        # Read environment data
        buffer = env_shm.buf[:shm_size]
        offset = 0

        # Robot position, orientation, timestamp
        robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
        offset += struct.calcsize('ff')
        robot_theta_deg, = struct.unpack_from('f', buffer, offset)
        offset += struct.calcsize('f')
        timestamp, = struct.unpack_from('f', buffer, offset)
        offset += struct.calcsize('f')

        # Number of cameras
        num_cameras, = struct.unpack_from('i', buffer, offset)
        offset += struct.calcsize('i')

        cameras = []
        for _ in range(num_cameras):
            cam_x, cam_y = struct.unpack_from('ff', buffer, offset)
            offset += struct.calcsize('ff')
            look_x, look_y = struct.unpack_from('ff', buffer, offset)
            offset += struct.calcsize('ff')
            num_rays, = struct.unpack_from('i', buffer, offset)
            offset += struct.calcsize('i')

            rays = []
            for _ in range(num_rays):
                inter_x, inter_y = struct.unpack_from('ff', buffer, offset)
                offset += struct.calcsize('ff')
                distance, = struct.unpack_from('f', buffer, offset)
                offset += struct.calcsize('f')
                object_id, = struct.unpack_from('i', buffer, offset)
                offset += struct.calcsize('i')
                ray_angle, = struct.unpack_from('f', buffer, offset)
                offset += struct.calcsize('f')

                rays.append({
                    'intersection': (inter_x, inter_y) if inter_x != float('inf') else None,
                    'distance': distance,
                    'object_id': object_id if object_id != -1 else None,
                    'ray_angle': ray_angle
                })

            cameras.append({
                'camera_pos': (cam_x, cam_y),
                'look_at_pos': (look_x, look_y),
                'rays': rays
            })

        env_data = {
            "robot_position": (robot_x, robot_y),
            "robot_orientation": robot_theta_deg,
            "timestamp": timestamp,
            "cameras": cameras
        }

        # Read SLAM data
        buffer = slam_shm.buf[:shm_size]
        slam_x, slam_y, slam_theta_deg = struct.unpack_from('fff', buffer, 0)
        slam_timestamp, = struct.unpack_from('f', buffer, struct.calcsize('fff'))
        num_landmarks, = struct.unpack_from('i', buffer, struct.calcsize('ffff'))

        landmarks = []
        offset = struct.calcsize('ffff') + struct.calcsize('i')
        for _ in range(num_landmarks):
            lx, ly = struct.unpack_from('ff', buffer, offset)
            offset += struct.calcsize('ff')
            landmarks.append((lx, ly))

        slam_data = {
            "slam_pose": (slam_x, slam_y, slam_theta_deg),
            "timestamp": slam_timestamp,
            "landmarks": landmarks
        }

        # Log Environment Data
        logging.info("Environment Data:")
        logging.info(f"  Robot Position: {env_data['robot_position']}")
        logging.info(f"  Robot Orientation: {env_data['robot_orientation']}")
        logging.info(f"  Timestamp: {env_data['timestamp']}")
        logging.info(f"  Number of Cameras: {len(env_data['cameras'])}")

        for i, camera in enumerate(env_data["cameras"]):
            logging.info(f"    Camera {i + 1} Position: {camera['camera_pos']}")
            logging.info(f"    Look At Position: {camera['look_at_pos']}")
            logging.info(f"    Number of Rays: {len(camera['rays'])}")
            for j, ray in enumerate(camera["rays"][:5]):  # Log first 5 rays only for brevity
                logging.info(f"      Ray {j + 1}: Intersection={ray['intersection']}, "
                             f"Distance={ray['distance']}, Object ID={ray['object_id']}, "
                             f"Angle={ray['ray_angle']}")

        # Log SLAM Data
        logging.info("SLAM Data:")
        logging.info(f"  SLAM Position: {slam_data['slam_pose'][:2]}")
        logging.info(f"  SLAM Orientation: {slam_data['slam_pose'][2]}")
        logging.info(f"  Timestamp: {slam_data['timestamp']}")
        logging.info(f"  Number of Landmarks: {len(slam_data['landmarks'])}")

        for i, landmark in enumerate(slam_data["landmarks"][:5]):  # Log first 5 landmarks for brevity
            logging.info(f"    Landmark {i + 1}: Position={landmark}")

        # Verify data consistency
        logging.info("Verifying shared memory data consistency...")
        if env_data["timestamp"] != slam_data["timestamp"]:
            logging.warning(f"Timestamps mismatch: Env={env_data['timestamp']}, SLAM={slam_data['timestamp']}")

        if math.isclose(env_data["robot_position"][0], slam_data["slam_pose"][0], abs_tol=0.01) and \
           math.isclose(env_data["robot_position"][1], slam_data["slam_pose"][1], abs_tol=0.01):
            logging.info("Position data matches.")
        else:
            logging.warning(f"Position mismatch: Env={env_data['robot_position']}, SLAM={slam_data['slam_pose'][:2]}")

        if math.isclose(env_data["robot_orientation"], slam_data["slam_pose"][2], abs_tol=0.5):
            logging.info("Orientation data matches.")
        else:
            logging.warning(f"Orientation mismatch: Env={env_data['robot_orientation']}, SLAM={slam_data['slam_pose'][2]}")

    except FileNotFoundError as e:
        logging.error(f"Shared memory error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during verification: {e}")
    finally:
        # Close shared memory segments
        env_shm.close()
        slam_shm.close()
        logging.info("Data verification completed.")

if __name__ == "__main__":
    manager = Manager()
    lock = manager.Lock()

    # Define shared memory sizes
    shm_input_size = 16060  # Adjust based on actual requirements
    shm_output_size = 16060  # Adjust based on expected SLAM output size

    # Start the Environment process first to ensure 'my_shared_memory' is created
    env_process = Process(target=run_environment, args=(lock,))
    env_process.start()

    # Give the Environment some time to initialize and create 'my_shared_memory'
    time.sleep(1)

    # Start the SLAMModule process
    slam_process = Process(target=run_slam_module, args=(
        lock,
        "my_shared_memory",
        shm_input_size,
        "slam_output",
        shm_output_size
    ))
    slam_process.start()

    # Start the Visualization process
    visualization_process = Process(target=run_visualization, args=("slam_output", shm_output_size))
    visualization_process.start()

    # Give processes some time to populate shared memory
    time.sleep(2)

    # Verify shared memory data once
    verify_shared_memory("my_shared_memory", "slam_output", shm_input_size)

    # Terminate all processes gracefully
    env_process.terminate()
    slam_process.terminate()
    visualization_process.terminate()

    env_process.join()
    slam_process.join()
    visualization_process.join()

    # Clean up shared memories
    try:
        shm_output = shared_memory.SharedMemory(name="slam_output")
        shm_output.close()
        shm_output.unlink()
        logging.info("Main process: 'slam_output' shared memory unlinked.")
    except FileNotFoundError:
        logging.warning("Main process: 'slam_output' shared memory not found for unlinking.")

    try:
        shm_input = shared_memory.SharedMemory(name='my_shared_memory')
        shm_input.close()
        shm_input.unlink()
        logging.info("Main process: 'my_shared_memory' shared memory unlinked.")
    except FileNotFoundError:
        logging.warning("Main process: 'my_shared_memory' shared memory not found for unlinking.")

    print("All processes ended.")
