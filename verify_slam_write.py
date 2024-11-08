# verify_slam_output.py

from multiprocessing import shared_memory
import struct

def read_slam_output(shm_output_size=16060):
    try:
        shm_output = shared_memory.SharedMemory(name='slam_output')
        buffer = shm_output.buf[:shm_output_size]
        offset = 0
        # Unpack robot pose
        robot_x, robot_y, robot_theta_deg = struct.unpack_from('fff', buffer, offset)
        offset += struct.calcsize('fff')
        # Unpack timestamp
        timestamp, = struct.unpack_from('f', buffer, offset)
        offset += struct.calcsize('f')
        # Unpack number of landmarks
        num_landmarks, = struct.unpack_from('i', buffer, offset)
        offset += struct.calcsize('i')
        print(f"SLAM Pose: ({robot_x}, {robot_y}), Orientation: {robot_theta_deg} degrees")
        print(f"Timestamp: {timestamp}")
        print(f"Number of Landmarks: {num_landmarks}")
        # Read landmark positions
        landmarks = []
        for _ in range(num_landmarks):
            lx, ly = struct.unpack_from('ff', buffer, offset)
            offset += struct.calcsize('ff')
            landmarks.append((lx, ly))
        print("Landmarks:")
        for landmark in landmarks:
            print(f"  {landmark}")
    except FileNotFoundError:
        print("Shared memory 'slam_output' not found.")
    except struct.error as e:
        print(f"Struct unpacking error: {e}")
    finally:
        try:
            shm_output.close()
        except:
            pass

if __name__ == "__main__":
    read_slam_output()
