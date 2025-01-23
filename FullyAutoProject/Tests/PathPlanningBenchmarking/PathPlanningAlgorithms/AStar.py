# PathPlanningAlgorithms/AStar.py

import heapq

from FullyAutoProject.Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class AStar:
    def __init__(self, grid, start, goal):
        """
        Initialize A* algorithm.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for start position.
        :param goal: Tuple (x, y) for goal position.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.x_dim, self.y_dim = grid.shape
        self.open_set = []
        heapq.heappush(self.open_set, (0 + heuristic(start, goal), 0, start))
        self.came_from = {}
        self.g_score = {start: 0}

    def run(self):
        while self.open_set:
            current_f, current_g, current = heapq.heappop(self.open_set)
            if current == self.goal:
                return self.reconstruct_path()
            for neighbor in self.get_neighbors(current):
                tentative_g = self.g_score[current] + heuristic(current, neighbor)
                if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, self.goal)
                    heapq.heappush(self.open_set, (f_score, tentative_g, neighbor))
        return None  # No path found

    def get_neighbors(self, vertex):
        x, y = vertex
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim and self.grid[nx, ny] != 255:
                neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self):
        path = []
        current = self.goal
        while current != self.start:
            path.append(current)
            current = self.came_from.get(current)
            if current is None:
                return None  # Path reconstruction failed
        path.append(self.start)
        path.reverse()
        return path
