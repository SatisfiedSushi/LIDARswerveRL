# src/mapping.py

import numpy as np

class OccupancyGridMap:
    def __init__(self, size_x, size_y, resolution):
        self.size_x = size_x  # meters
        self.size_y = size_y
        self.resolution = resolution  # meters per cell
        self.width = int(size_x / resolution)
        self.height = int(size_y / resolution)
        self.grid = np.zeros((self.width, self.height))  # log-odds representation

        # Log-odds parameters
        self.l0 = 0  # Prior
        self.l_occ = np.log(0.7 / (1 - 0.7))  # Occupied
        self.l_free = np.log(0.3 / (1 - 0.3))  # Free

    def world_to_grid(self, x, y):
        grid_x = int((x + self.size_x / 2) / self.resolution)
        grid_y = int((y + self.size_y / 2) / self.resolution)
        return grid_x, grid_y

    def update(self, robot_pose, measurements):
        """
        Update the occupancy grid based on robot pose and measurements.
        robot_pose: [x, y, theta]
        measurements: list of [range, angle] from Lidar
        """
        x, y, theta = robot_pose
        for r, alpha in measurements:
            # Convert polar to world coordinates
            world_x = x + r * np.cos(theta + alpha)
            world_y = y + r * np.sin(theta + alpha)

            # Convert to grid coordinates
            grid_x, grid_y = self.world_to_grid(world_x, world_y)

            # Update the occupied cell
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.grid[grid_x, grid_y] += self.l_occ

            # Optionally, update free cells along the ray
            num_free = int(r / self.resolution)
            for i in range(num_free):
                fx = x + i * self.resolution * np.cos(theta + alpha)
                fy = y + i * self.resolution * np.sin(theta + alpha)
                f_grid_x, f_grid_y = self.world_to_grid(fx, fy)
                if 0 <= f_grid_x < self.width and 0 <= f_grid_y < self.height:
                    self.grid[f_grid_x, f_grid_y] += self.l_free

    def get_probability_map(self):
        return 1 - 1 / (1 + np.exp(self.grid))
