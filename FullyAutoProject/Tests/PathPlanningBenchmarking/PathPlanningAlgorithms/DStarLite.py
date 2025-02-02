# PathPlanningAlgorithms/DStarLite.py

import logging
import numpy as np

from Tests.PathPlanningBenchmarking.PathPlanningAlgorithms.DStarLiteUtils.OccupancyGridMap import \
    OccupancyGridMap
from Tests.PathPlanningBenchmarking.PathPlanningAlgorithms.DStarLiteUtils.Priority import Priority
from Tests.PathPlanningBenchmarking.PathPlanningAlgorithms.DStarLiteUtils.PriorityQueue import \
    PriorityQueue
from Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class DStarLite:
    def __init__(self, grid, start, goal):
        """
        Initialize D* Lite with the given grid, start, and goal.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for start position.
        :param goal: Tuple (x, y) for goal position.
        """
        self.map = OccupancyGridMap(x_dim=grid.shape[0], y_dim=grid.shape[1], exploration_setting='8N')
        self.map.grid = grid.copy()

        self.s_start = start
        self.s_goal = goal
        self.s_last = start
        self.k_m = 0  # Accumulation
        self.U = PriorityQueue()
        self.rhs = np.ones((self.map.x_dim, self.map.y_dim)) * np.inf
        self.g = self.rhs.copy()

        self.rhs[self.s_goal] = 0
        self.U.insert(self.s_goal, self.calculate_key(self.s_goal))

    def calculate_key(self, s):
        """
        Calculate the priority key for a given node.

        :param s: Tuple (x, y)
        :return: Priority object
        """
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)

    def c(self, u, v):
        """
        Calculate the cost between nodes.

        :param u: Tuple (x, y)
        :param v: Tuple (x, y)
        :return: Cost (float)
        """
        if not self.map.is_unoccupied(u) or not self.map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)

    def contain(self, u):
        """
        Check if a node is in the priority queue.

        :param u: Tuple (x, y)
        :return: Boolean
        """
        return u in self.U.entry_finder

    def update_vertex(self, u):
        """
        Update the vertex in the priority queue based on its g and rhs values.

        :param u: Tuple (x, y)
        """
        if u != self.s_goal:
            min_rhs = float('inf')
            for s in self.map.succ(u):
                temp = self.c(u, s) + self.g[s]
                if temp < min_rhs:
                    min_rhs = temp
            self.rhs[u] = min_rhs
        if self.g[u] != self.rhs[u]:
            if self.contain(u):
                self.U.update(u, self.calculate_key(u))
            else:
                self.U.insert(u, self.calculate_key(u))
        else:
            if self.contain(u):
                self.U.remove(u)

    def compute_shortest_path(self):
        """
        Compute the shortest path using D* Lite algorithm.
        """
        while (self.U.top_key() < self.calculate_key(self.s_start)) or (self.rhs[self.s_start] > self.g[self.s_start]):
            u = self.U.pop()
            k_old = self.calculate_key(u)
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)
                for s in self.map.succ(u):
                    self.update_vertex(s)
            else:
                g_old = self.g[u]
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.map.succ(u):
                    if self.rhs[s] == self.c(s, u) + g_old:
                        self.update_vertex(s)

    def reconstruct_path(self):
        """
        Reconstruct the path from start to goal.

        :return: List of tuples representing the path.
        """
        path = []
        current = self.s_start
        if self.g[current] == float('inf'):
            logging.error("No path found!")
            return None
        while current != self.s_goal:
            path.append(current)
            neighbors = self.map.succ(current)
            min_cost = float('inf')
            next_node = None
            for s in neighbors:
                cost = self.c(current, s) + self.g[s]
                if cost < min_cost:
                    min_cost = cost
                    next_node = s
            if next_node is None:
                logging.error("Path reconstruction failed!")
                return None
            current = next_node
        path.append(self.s_goal)
        return path

    def run(self):
        """
        Execute the D* Lite algorithm and return the path from start to goal.

        :return: List of tuples representing the path, or None if no path found.
        """
        try:
            self.compute_shortest_path()
            path = self.reconstruct_path()
            if path:
                logging.info("D* Lite: Path found successfully.")
                return path
            else:
                logging.error("D* Lite: Path reconstruction failed.")
                return None
        except Exception as e:
            logging.error(f"D* Lite encountered an error: {e}")
            return None
