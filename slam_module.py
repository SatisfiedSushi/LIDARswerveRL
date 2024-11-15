import struct
import time
from multiprocessing import shared_memory
import numpy as np
import math
import sys
import logging
import random


class EKF_SLAM:
    def __init__(self):
        # Initialize state vector [x, y, theta]
        self.mu = np.zeros(3)
        # Initialize covariance matrix
        self.Sigma = np.eye(3) * 0.1
        # Initialize landmarks dictionary {id: (x, y)}
        self.landmarks = {}
        self.next_landmark_id = 0
        # Process noise covariance
        self.R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
        # Measurement noise covariance
        self.Q = np.diag([0.2, np.deg2rad(5)]) ** 2

        # Initialize moving objects dictionary {id: MovingObject}
        self.moving_objects = {}
        self.next_moving_object_id = 0

        # Initialize logger for EKF_SLAM
        self.logger = logging.getLogger('EKF_SLAM')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def predict(self, control, dt=0.1):
        """
        EKF SLAM Prediction Step
        """
        v, w = control
        theta = self.mu[2]

        if w != 0:
            delta_theta = w * dt
            delta_x = (v / w) * (math.sin(theta + delta_theta) - math.sin(theta))
            delta_y = (v / w) * (-math.cos(theta + delta_theta) + math.cos(theta))
        else:
            delta_theta = 0
            delta_x = v * math.cos(theta) * dt
            delta_y = v * math.sin(theta) * dt

        # Update state
        self.mu[0] += delta_x
        self.mu[1] += delta_y
        self.mu[2] += delta_theta
        self.mu[2] = self.normalize_angle(self.mu[2])

        # Jacobian of motion model for the robot pose (3x3)
        if w != 0:
            G_r = np.array([
                [1, 0, (v / w) * (math.cos(theta + delta_theta) - math.cos(theta))],
                [0, 1, (v / w) * (math.sin(theta + delta_theta) - math.sin(theta))],
                [0, 0, 1]
            ])
        else:
            G_r = np.array([
                [1, 0, -v * math.sin(theta) * dt],
                [0, 1, v * math.cos(theta) * dt],
                [0, 0, 1]
            ])

        # Construct the full Jacobian G (n x n)
        n = len(self.mu)
        G = np.eye(n)
        G[0:3, 0:3] = G_r

        # Expand process noise covariance R to match Sigma's size (n x n)
        R_expanded = np.zeros((n, n))
        R_expanded[0:3, 0:3] = self.R

        # Update covariance
        self.Sigma = G @ self.Sigma @ G.T + R_expanded

        # Predict moving objects
        for obj_id, moving_object in self.moving_objects.items():
            moving_object.predict(dt)

        self.logger.debug(f"After predict - mu: {self.mu}, Sigma: {self.Sigma}")

    def update(self, observations):
        """
        EKF SLAM Update Step
        observations: list of (range, bearing)
        """
        for obs in observations:
            r, b = obs
            # Convert to world coordinates
            x = self.mu[0] + r * math.cos(self.mu[2] + b)
            y = self.mu[1] + r * math.sin(self.mu[2] + b)
            landmark_pos = np.array([x, y])

            # Data association with static landmarks
            associated_id = self.associate_landmark(landmark_pos)

            if associated_id is not None:
                # Update existing landmark
                self.update_landmark(associated_id, r, b)
            else:
                # Attempt to associate with moving objects
                associated_moving_id = self.associate_moving_object(landmark_pos)

                if associated_moving_id is not None:
                    # Update moving object
                    moving_object = self.moving_objects[associated_moving_id]
                    moving_object.update((x, y))
                else:
                    # Decide whether to create a new landmark or moving object
                    if self.is_moving_object(r, b):
                        # Create new moving object
                        moving_object = MovingObject((x, y))
                        self.moving_objects[self.next_moving_object_id] = moving_object
                        self.next_moving_object_id += 1
                        self.logger.debug(f"Added new moving object ID {self.next_moving_object_id - 1} at position {(x, y)}")
                    else:
                        # Add new landmark
                        self.add_new_landmark(landmark_pos)

        # After updating all observations, resample moving objects
        for moving_object in self.moving_objects.values():
            moving_object.resample()

    def update_landmark(self, landmark_id, r, b):
        """
        Update existing landmark with new observation.
        """
        z = np.array([r, b])
        landmark_index = 3 + landmark_id * 2  # Position in state vector
        landmark = self.mu[landmark_index:landmark_index + 2]
        delta = landmark - self.mu[0:2]
        q = delta @ delta
        sqrt_q = math.sqrt(q)
        z_hat = np.array([sqrt_q, math.atan2(delta[1], delta[0]) - self.mu[2]])
        z_hat[1] = self.normalize_angle(z_hat[1])

        y_k = z - z_hat
        y_k[1] = self.normalize_angle(y_k[1])

        # Measurement Jacobian H (2 x n)
        H = np.zeros((2, len(self.mu)))
        H[0, 0] = -delta[0] / sqrt_q
        H[0, 1] = -delta[1] / sqrt_q
        H[0, 2] = 0
        H[0, landmark_index] = delta[0] / sqrt_q
        H[0, landmark_index + 1] = delta[1] / sqrt_q

        H[1, 0] = delta[1] / q
        H[1, 1] = -delta[0] / q
        H[1, 2] = -1
        H[1, landmark_index] = -delta[1] / q
        H[1, landmark_index + 1] = delta[0] / q

        # Measurement covariance
        S = H @ self.Sigma @ H.T + self.Q

        # Kalman Gain
        try:
            K = self.Sigma @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.logger.error("Singular matrix encountered during Kalman Gain computation.")
            return

        # Update state
        self.mu = self.mu + K @ y_k
        self.mu[2] = self.normalize_angle(self.mu[2])

        # Update covariance
        self.Sigma = (np.eye(len(self.mu)) - K @ H) @ self.Sigma

        # Update landmark position
        self.landmarks[landmark_id] = self.mu[landmark_index:landmark_index + 2]

        self.logger.debug(f"Updated landmark ID {landmark_id} at position {self.landmarks[landmark_id]}")
        self.logger.debug(f"After update - mu: {self.mu}, Sigma: {self.Sigma}")

    def add_new_landmark(self, landmark_pos):
        """
        Adds a new landmark to the state vector and covariance matrix.
        """
        associated_id = self.next_landmark_id
        self.landmarks[associated_id] = landmark_pos
        self.next_landmark_id += 1
        # Expand state vector and covariance
        self.mu = np.append(self.mu, landmark_pos)
        # Expand covariance matrix
        n = len(self.mu)
        Sigma_new = np.zeros((n, n))
        Sigma_new[:n - 2, :n - 2] = self.Sigma
        Sigma_new[-2:, -2:] = np.eye(2) * 1e6  # Large uncertainty for new landmark
        self.Sigma = Sigma_new
        self.logger.debug(f"Added new landmark ID {associated_id} at position {landmark_pos}")

    def associate_landmark(self, landmark_pos, threshold=0.5):
        """
        Associate a landmark with existing landmarks based on distance threshold.
        """
        min_dist = float('inf')
        associated_id = None
        for lid, pos in self.landmarks.items():
            dist = np.linalg.norm(landmark_pos - pos)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                associated_id = lid
        return associated_id

    def associate_moving_object(self, object_pos, threshold=0.5):
        """
        Associate an observation with existing moving objects based on distance threshold.
        """
        min_dist = float('inf')
        associated_id = None
        for oid, moving_object in self.moving_objects.items():
            estimated_pos = moving_object.estimate()
            dist = np.linalg.norm(object_pos - estimated_pos)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                associated_id = oid
        return associated_id

    def is_moving_object(self, r, b):
        """
        Determine if an observation corresponds to a moving object.
        Here, we can implement logic to decide if an object is moving.
        For simplicity, we'll assume that if it's not a landmark, it's a moving object.
        """
        return True  # Simplified assumption

    def get_pose(self):
        """
        Returns the current estimated robot pose.
        """
        return self.mu[:3]

    def get_map(self):
        """
        Returns the current estimated map landmarks.
        """
        return np.array(list(self.landmarks.values()))

    def get_moving_objects(self):
        """
        Returns the estimated positions of moving objects.
        """
        positions = []
        for moving_object in self.moving_objects.values():
            positions.append(moving_object.estimate())
        return positions

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to be between -pi and pi.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi


class MovingObject:
    def __init__(self, initial_pos):
        self.num_particles = 100
        self.particles = [np.array(initial_pos) + np.random.randn(2) * 0.5 for _ in range(self.num_particles)]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, dt):
        """
        Predict the state of the moving object particles.
        """
        for i in range(self.num_particles):
            # Simple motion model: random walk
            dx = np.random.randn() * 0.1
            dy = np.random.randn() * 0.1
            self.particles[i] += np.array([dx, dy])

    def update(self, observation):
        """
        Update the particles weights based on observation.
        """
        for i in range(self.num_particles):
            distance = np.linalg.norm(self.particles[i] - observation)
            self.weights[i] *= self.gaussian(distance, 0.5)
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles based on weights.
        """
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = [self.particles[i] for i in indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """
        Estimate the position of the moving object.
        """
        return np.average(self.particles, axis=0, weights=self.weights)

    @staticmethod
    def gaussian(x, sigma):
        return math.exp(- (x ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


class SLAMModule:
    def __init__(self, lock, shm_input_name='my_shared_memory', shm_input_size=16060,
                 shm_output_name='slam_output', shm_output_size=16060):
        self.lock = lock
        self.shm_input_name = shm_input_name
        self.shm_input_size = shm_input_size
        self.shm_output_name = shm_output_name
        self.shm_output_size = shm_output_size

        # Initialize Logger
        self.logger = logging.getLogger('SLAMModule')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.logger.info("Initializing SLAMModule...")

        # Implement a retry mechanism to wait for 'my_shared_memory' to be available
        while True:
            try:
                self.shm_input = shared_memory.SharedMemory(name=self.shm_input_name)
                self.logger.info(f"SLAM Module connected to input shared memory: {self.shm_input.name}")
                break
            except FileNotFoundError:
                self.logger.warning(f"SLAM Module: Input shared memory '{self.shm_input_name}' not found. Retrying in 0.5 seconds...")
                time.sleep(0.5)

        # Create or connect to 'slam_output' shared memory
        try:
            self.shm_output = shared_memory.SharedMemory(name=self.shm_output_name)
            self.logger.info(f"SLAM Module connected to existing output shared memory: {self.shm_output.name}")
        except FileNotFoundError:
            # Create 'slam_output' shared memory
            self.shm_output = shared_memory.SharedMemory(create=True, size=self.shm_output_size,
                                                         name=self.shm_output_name)
            self.logger.info(f"SLAM Module created output shared memory: {self.shm_output.name}")

        self.slam = EKF_SLAM()
        self.prev_pose = None

    def read_from_shared_memory(self):
        """
        Reads data from the environment's shared memory.
        Returns:
        - pose: Tuple of (x, y, theta_rad)
        - timestamp
        - cameras: List of camera data (unused here)
        """
        with self.lock:
            buffer = self.shm_input.buf[:self.shm_input_size]
            offset = 0
            try:
                # Unpack robot pose
                robot_x, robot_y, robot_theta_deg = struct.unpack_from('fff', buffer, offset)
                offset += struct.calcsize('fff')
                robot_theta_rad = math.radians(robot_theta_deg)

                # Unpack timestamp
                timestamp, = struct.unpack_from('f', buffer, offset)
                offset += struct.calcsize('f')

                # Unpack number of cameras
                num_cameras, = struct.unpack_from('i', buffer, offset)
                offset += struct.calcsize('i')

                cameras = []
                for _ in range(num_cameras):
                    # Camera Position
                    cam_x, cam_y = struct.unpack_from('ff', buffer, offset)
                    offset += struct.calcsize('ff')
                    # Look At Position
                    look_x, look_y = struct.unpack_from('ff', buffer, offset)
                    offset += struct.calcsize('ff')
                    # Number of Rays
                    num_rays, = struct.unpack_from('i', buffer, offset)
                    offset += struct.calcsize('i')
                    rays = []
                    for _ in range(num_rays):
                        # Intersection x, y
                        inter_x, inter_y = struct.unpack_from('ff', buffer, offset)
                        offset += struct.calcsize('ff')
                        # Distance
                        distance, = struct.unpack_from('f', buffer, offset)
                        offset += struct.calcsize('f')
                        # Object ID
                        object_id, = struct.unpack_from('i', buffer, offset)
                        offset += struct.calcsize('i')
                        # Ray Angle (already in radians)
                        ray_angle, = struct.unpack_from('f', buffer, offset)
                        offset += struct.calcsize('f')

                        # Handle infinite intersections
                        if inter_x == float('inf') and inter_y == float('inf'):
                            intersection = None
                        else:
                            intersection = (inter_x, inter_y)
                        rays.append({
                            'intersection': intersection,
                            'distance': distance,
                            'object_id': object_id if object_id != -1 else None,
                            'ray_angle': ray_angle
                        })
                    cameras.append({
                        'camera_pos': (cam_x, cam_y),
                        'look_at_pos': (look_x, look_y),
                        'rays': rays
                    })

                return (robot_x, robot_y, robot_theta_rad), timestamp, cameras
            except struct.error as e:
                self.logger.error(f"Struct unpacking error: {e}")
                return None, None, None
            except Exception as e:
                self.logger.error(f"Error reading shared memory: {e}")
                return None, None, None

    def write_to_shared_memory_output(self, pose, timestamp, map_points, moving_objects):
        """
        Writes the SLAM estimated pose, map landmarks, and moving objects to the output shared memory.
        """
        buffer = bytearray()
        try:
            # Pack robot pose (x, y, theta in degrees)
            buffer += struct.pack('fff', pose[0], pose[1], math.degrees(pose[2]))
            # Pack timestamp
            buffer += struct.pack('f', timestamp)
            # Pack number of landmarks
            num_landmarks = len(map_points)
            buffer += struct.pack('i', num_landmarks)
            # Pack landmark positions
            for point in map_points:
                buffer += struct.pack('ff', point[0], point[1])
            # Pack number of moving objects
            num_moving_objects = len(moving_objects)
            buffer += struct.pack('i', num_moving_objects)
            # Pack moving object positions
            for pos in moving_objects:
                buffer += struct.pack('ff', pos[0], pos[1])
            # Write to shared memory
            with self.lock:
                shm_view = memoryview(self.shm_output.buf)
                shm_view[:len(buffer)] = buffer
            self.logger.debug(f"Wrote SLAM Pose: {pose}, Landmarks Count: {num_landmarks}, Moving Objects Count: {num_moving_objects}")
        except struct.error as e:
            self.logger.error(f"Struct packing error: {e}")
        except Exception as e:
            self.logger.error(f"Error writing to shared memory: {e}")

    def run(self):
        """
        Main loop for the SLAM module.
        """
        try:
            while True:
                pose, timestamp, cameras = self.read_from_shared_memory()
                if pose is None:
                    time.sleep(0.1)
                    continue
                robot_x, robot_y, robot_theta_rad = pose
                current_time = timestamp  # Assuming timestamp is in seconds

                # Compute control inputs based on pose changes
                control = self.compute_control(pose, current_time)

                # Run SLAM predict and update
                self.slam.predict(control)
                observations = []
                for cam in cameras:
                    for ray in cam['rays']:
                        distance = ray['distance']
                        if distance == float('inf') or distance >= 500:
                            continue  # Ignore invalid distances
                        bearing = ray['ray_angle']  # Assuming bearing is already relative to robot orientation
                        observations.append((distance, bearing))
                if observations:
                    self.slam.update(observations)

                # Get SLAM results
                slam_pose = self.slam.get_pose()
                slam_map = self.slam.get_map()
                moving_objects = self.slam.get_moving_objects()

                # Write SLAM results to output shared memory
                self.write_to_shared_memory_output(slam_pose, timestamp, slam_map, moving_objects)

                # Sleep briefly to match data rate
                time.sleep(0.05)  # 20 Hz
        except KeyboardInterrupt:
            self.logger.info("SLAM Module interrupted.")
        finally:
            self.shm_input.close()
            self.shm_output.close()
            self.logger.info("SLAM Module ended.")

    def compute_control(self, current_pose, current_time):
        """
        Computes control inputs based on pose changes.
        """
        robot_x, robot_y, robot_theta_rad = current_pose
        if self.prev_pose is None:
            self.prev_pose = current_pose
            self.prev_time = current_time
            return (0.0, 0.0)
        else:
            prev_x, prev_y, prev_theta_rad = self.prev_pose
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 1e-3  # Prevent division by zero
            dx = robot_x - prev_x
            dy = robot_y - prev_y
            dtheta = self.normalize_angle(robot_theta_rad - prev_theta_rad)
            self.prev_pose = current_pose
            self.prev_time = current_time
            # Convert pose change to control inputs
            v = math.sqrt(dx ** 2 + dy ** 2) / dt
            w = dtheta / dt
            self.logger.debug(f"Computed Control - v: {v}, w: {w}, dt: {dt}")
            return (v, w)

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to be between -pi and pi.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi


def run_slam_module(lock, shm_input_name='my_shared_memory', shm_input_size=16060,
                   shm_output_name='slam_output', shm_output_size=16060):
    """
    Initializes and runs the SLAM module.
    """
    slam_module = SLAMModule(
        lock=lock,
        shm_input_name=shm_input_name,
        shm_input_size=shm_input_size,
        shm_output_name=shm_output_name,
        shm_output_size=shm_output_size
    )
    slam_module.run()


if __name__ == "__main__":
    import argparse
    from multiprocessing import Process, Manager

    parser = argparse.ArgumentParser(description='EKF SLAM Module with Moving Object Tracking.')
    parser.add_argument('--shm_input_name', type=str, default='my_shared_memory', help='Name of input shared memory.')
    parser.add_argument('--shm_input_size', type=int, default=16060, help='Size of input shared memory in bytes.')
    parser.add_argument('--shm_output_name', type=str, default='slam_output', help='Name of output shared memory.')
    parser.add_argument('--shm_output_size', type=int, default=16060, help='Size of output shared memory in bytes.')

    args = parser.parse_args()

    manager = Manager()
    lock = manager.Lock()

    slam_process = Process(target=run_slam_module, args=(
        lock,
        args.shm_input_name,
        args.shm_input_size,
        args.shm_output_name,
        args.shm_output_size
    ))

    slam_process.start()
    slam_process.join()
