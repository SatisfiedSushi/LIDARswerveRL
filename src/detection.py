# src/detection.py

import numpy as np

class MovingObjectDetector:
    def __init__(self, occupancy_map, robot_pose, max_laser_range=10.0, laser_tolerance=0.1):
        self.occupancy_map = occupancy_map
        self.robot_pose = robot_pose  # [x, y, theta]
        self.max_laser_range = max_laser_range
        self.laser_tolerance = laser_tolerance

    def detect_moving_objects(self, measurements):
        """
        Detect moving objects based on consistency with the occupancy map.
        measurements: list of [range, angle] from Lidar
        Returns: list of detected moving points [x, y]
        """
        moving_points = []
        x, y, theta = self.robot_pose
        for r, alpha in measurements:
            if r > self.max_laser_range:
                continue  # Ignore out-of-range measurements

            world_x = x + r * np.cos(theta + alpha)
            world_y = y + r * np.sin(theta + alpha)

            grid_x, grid_y = self.occupancy_map.world_to_grid(world_x, world_y)

            # Check occupancy probability
            if 0 <= grid_x < self.occupancy_map.width and 0 <= grid_y < self.occupancy_map.height:
                prob = self.occupancy_map.get_probability_map()[grid_x, grid_y]
                if prob < 0.5 + self.laser_tolerance:
                    moving_points.append([world_x, world_y])

        return moving_points
