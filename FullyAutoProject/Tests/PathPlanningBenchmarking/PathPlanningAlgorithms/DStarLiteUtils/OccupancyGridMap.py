# PathPlanningAlgorithms/DStarLiteUtils/OccupancyGridMap.py

import numpy as np

class OccupancyGridMap:
    def __init__(self, x_dim, y_dim, exploration_setting='8N'):
        """
        Initialize the Occupancy Grid Map.

        :param x_dim: Number of cells in the x-direction.
        :param y_dim: Number of cells in the y-direction.
        :param exploration_setting: '4N' or '8N' for neighbor connectivity.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.grid = np.zeros((x_dim, y_dim), dtype=int)  # 0 for free, 255 for obstacle
        self.exploration_setting = exploration_setting

    def is_unoccupied(self, pos):
        """
        Check if a position is unoccupied.

        :param pos: Tuple (x, y)
        :return: Boolean
        """
        x, y = pos
        return self.grid[x, y] != 255

    def succ(self, pos):
        """
        Get successors of a position based on exploration setting.

        :param pos: Tuple (x, y)
        :return: List of tuples
        """
        x, y = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.exploration_setting == '8N':
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim and self.is_unoccupied((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors

    def get_obstacle_segments(self):
        """
        Extract obstacle segments from the grid.

        :return: List of tuples representing line segments
        """
        segments = []
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.grid[x, y] == 255:
                    # Check right and down neighbors
                    if y < self.y_dim - 1 and self.grid[x, y + 1] == 255:
                        segments.append(((x, y), (x, y + 1)))
                    if x < self.x_dim - 1 and self.grid[x + 1, y] == 255:
                        segments.append(((x, y), (x + 1, y)))
        return segments

    def has_obstacles(self):
        """
        Check if the map has any obstacles.

        :return: Boolean
        """
        return np.any(self.grid == 255)
