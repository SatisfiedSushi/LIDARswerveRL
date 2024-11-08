# test.py

import logging
from multiprocessing import Process, Manager, shared_memory
from slam_module import SLAMModule
from receiver import visualization_main
from OptimizedDEVRLEnvIntake import env
import time

def run_environment(lock):
    env_instance = env(render_mode="human", max_teleop_time=1000000, lock=lock)
    obs, info = env_instance.reset()
    try:
        while True:
            obs, reward, terminated, truncated, info = env_instance.step((1.0, 0.0, 0.1), testing_mode=True)
            env_instance.render()
            if terminated or truncated:
                obs, info = env_instance.reset()
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

    time.sleep(1)

    # Start the Visualization process
    visualization_process = Process(target=run_visualization, args=("slam_output", shm_output_size))
    visualization_process.start()

    try:
        # Wait for the Environment process to complete
        env_process.join()
    except KeyboardInterrupt:
        logging.info("Main process interrupted by user.")
    finally:
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
