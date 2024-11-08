# slam_module.py

import struct
import time
from multiprocessing import shared_memory
import numpy as np
import math
import sys
import logging

class EKF_SLAM:
    def __init__(self):
        # Initialize state vector, covariance, landmarks, etc.
        self.mu = np.zeros(3)
        self.Sigma = np.eye(3) * 0.1
        self.landmarks = {}
        self.next_landmark_id = 0
        self.R = np.diag([0.1, 0.1, np.deg2rad(5)])**2
        self.Q = np.diag([0.2, np.deg2rad(5)])**2

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

        n = len(self.mu)  # Current size of state vector

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
        G = np.eye(n)
        G[0:3, 0:3] = G_r

        # Expand process noise covariance R to match Sigma's size (n x n)
        R_expanded = np.zeros((n, n))
        R_expanded[0:3, 0:3] = self.R

        # Update covariance
        self.Sigma = G @ self.Sigma @ G.T + R_expanded

        # Logging
        # self.logger.debug(f"After predict - mu: {self.mu}, Sigma: {self.Sigma}")

    def update(self, observations):
        """
        observations: list of (range, bearing)
        """
        for obs in observations:
            r, b = obs
            # Convert to world coordinates
            x = self.mu[0] + r * math.cos(self.mu[2] + b)
            y = self.mu[1] + r * math.sin(self.mu[2] + b)
            landmark_pos = np.array([x, y])

            # Find if landmark is already observed (data association)
            associated_id = self.associate_landmark(landmark_pos)

            if associated_id is None:
                # New landmark
                associated_id = self.next_landmark_id
                self.landmarks[associated_id] = landmark_pos
                self.next_landmark_id += 1
                # Expand state vector and covariance
                self.mu = np.append(self.mu, [x, y])
                # Expand covariance matrix
                self.Sigma = np.pad(self.Sigma, ((0,2),(0,2)), 'constant')
                # Initialize new landmark covariance
                self.Sigma[-2,-2] = self.Q[0,0]
                self.Sigma[-1,-1] = self.Q[1,1]
                # self.logger.debug(f"Added new landmark ID {associated_id} at position {landmark_pos}")
            else:
                # Update existing landmark
                z = np.array([r, b])
                landmark_index = 3 + associated_id * 2  # Position in state vector
                landmark = self.mu[landmark_index:landmark_index+2]
                delta = landmark - self.mu[0:2]
                q = delta @ delta
                sqrt_q = math.sqrt(q)
                z_hat = np.array([sqrt_q, math.atan2(delta[1], delta[0]) - self.mu[2]])
                z_hat[1] = self.normalize_angle(z_hat[1])

                y_k = z - z_hat
                y_k[1] = self.normalize_angle(y_k[1])

                # Measurement Jacobian H (2 x n)
                n = len(self.mu)
                H = np.zeros((2, n))
                H[0, 0] = -delta[0]/sqrt_q
                H[0, 1] = -delta[1]/sqrt_q
                H[0, 2] = 0
                H[0, landmark_index] = delta[0]/sqrt_q
                H[0, landmark_index+1] = delta[1]/sqrt_q

                H[1, 0] = delta[1]/q
                H[1, 1] = -delta[0]/q
                H[1, 2] = -1
                H[1, landmark_index] = -delta[1]/q
                H[1, landmark_index+1] = delta[0]/q

                # Measurement covariance
                S = H @ self.Sigma @ H.T + self.Q

                # Kalman Gain
                try:
                    K = self.Sigma @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    self.logger.error("Singular matrix encountered during Kalman Gain computation.")
                    continue

                # Update state
                self.mu = self.mu + K @ y_k
                self.mu[2] = self.normalize_angle(self.mu[2])

                # Update covariance
                self.Sigma = (np.eye(n) - K @ H) @ self.Sigma

                # Update landmark position
                self.landmarks[associated_id] = self.mu[landmark_index:landmark_index+2]

                # Logging
                # self.logger.debug(f"Updated landmark ID {associated_id} at position {self.landmarks[associated_id]}")
                # self.logger.debug(f"After update - mu: {self.mu}, Sigma: {self.Sigma}")

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

    def get_pose(self):
        return self.mu[:3]

    def get_map(self):
        return np.array(list(self.landmarks.values()))

    @staticmethod
    def normalize_angle(angle):
        # Keep angle between -pi and pi
        return (angle + math.pi) % (2 * math.pi) - math.pi

class SLAMModule:
    def __init__(self, lock, shm_input_name='my_shared_memory', shm_input_size=16060, shm_output_name='slam_output',
                 shm_output_size=16060):
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
        with self.lock:
            buffer = self.shm_input.buf[:self.shm_input_size]
            offset = 0
            try:
                # Unpack robot position and orientation
                robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
                offset += struct.calcsize('ff')
                robot_theta_deg, = struct.unpack_from('f', buffer, offset)
                robot_theta_rad = math.radians(robot_theta_deg)
                offset += struct.calcsize('f')
                # Unpack timestamp
                timestamp, = struct.unpack_from('f', buffer, offset)
                # self.logger.debug(f"SLAM Module: Read data with timestamp {timestamp}")
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
                self.logger.error(f"SLAM Module: Struct unpacking error: {e}")
                return None, None, None
            except Exception as e:
                self.logger.error(f"SLAM Module: Error reading shared memory: {e}")
                return None, None, None

    def write_to_shared_memory_output(self, pose, timestamp, map_points):
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
            # Write to shared memory
            with self.lock:
                shm_view = memoryview(self.shm_output.buf)
                shm_view[:len(buffer)] = buffer
            # self.logger.debug(f"Writing SLAM Pose: {pose}, Landmarks Count: {len(map_points)}")
        except struct.error as e:
            self.logger.error(f"SLAM Module: Struct packing error: {e}")
        except Exception as e:
            self.logger.error(f"SLAM Module: Error writing to shared memory: {e}")

    def run(self):
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
                        ray_angle_rad = ray['ray_angle']  # Already in radians
                        # Compute bearing relative to robot orientation
                        bearing = self.normalize_angle(ray_angle_rad - robot_theta_rad)
                        observations.append((distance, bearing))
                if observations:
                    self.slam.update(observations)
                # Get SLAM results
                slam_pose = self.slam.get_pose()
                slam_map = self.slam.get_map()
                # Logging
                # self.logger.debug(f"SLAM Estimated Pose: {slam_pose}, Landmarks Count: {len(slam_map)}")
                # Write SLAM results to output shared memory
                self.write_to_shared_memory_output(slam_pose, timestamp, slam_map)

                # Sleep to match data rate (optional: implement if needed)
        except KeyboardInterrupt:
            self.logger.info("SLAM Module interrupted.")
        finally:
            self.shm_input.close()
            self.shm_output.close()
            self.logger.info("SLAM Module ended.")

    def compute_control(self, current_pose, current_time):
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
            v = math.sqrt(dx**2 + dy**2) / dt
            w = dtheta / dt
            # self.logger.debug(f"Computed Control - v: {v}, w: {w}, dt: {dt}")
            return (v, w)

    @staticmethod
    def normalize_angle(angle):
        # Keep angle between -pi and pi
        return (angle + math.pi) % (2 * math.pi) - math.pi
