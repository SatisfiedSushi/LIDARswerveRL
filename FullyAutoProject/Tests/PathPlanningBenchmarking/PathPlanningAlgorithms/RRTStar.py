# PathPlanningAlgorithms/RRTStar.py

import math
import random

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None

class RRTStar:
    def __init__(self, grid, start, goal, max_iterations=500, step_size=10, search_radius=15):
        """
        Initialize the RRT* algorithm.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for the start position.
        :param goal: Tuple (x, y) for the goal position.
        :param max_iterations: Maximum number of iterations to run.
        :param step_size: Distance to extend the tree in each step.
        :param search_radius: Radius to search for nearby nodes for rewiring.
        """
        self.grid = grid
        self.start = Node(start)
        self.goal = Node(goal)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.search_radius = search_radius
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
        self.nodes = [self.start]

    def run(self):
        for _ in range(self.max_iterations):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            if self.is_collision_free(nearest_node, new_node):
                near_nodes = self.get_near_nodes(new_node)
                best_parent = self.choose_parent(near_nodes, new_node)
                if best_parent:
                    new_node.parent = best_parent
                    self.nodes.append(new_node)
                    self.rewire(near_nodes, new_node)
                    if self.distance(new_node, self.goal) <= self.step_size:
                        if self.is_collision_free(new_node, self.goal):
                            self.goal.parent = new_node
                            self.nodes.append(self.goal)
                            return self.reconstruct_path()
        # No path found
        return None

    def get_random_node(self):
        """Sample a random node within the grid."""
        i = random.randint(0, self.rows - 1)
        j = random.randint(0, self.cols - 1)
        return Node((i, j))

    def get_nearest_node(self, random_node):
        """Find the nearest node in the tree to the random_node."""
        nearest = self.nodes[0]
        min_dist = self.distance(nearest, random_node)
        for node in self.nodes:
            dist = self.distance(node, random_node)
            if dist < min_dist:
                nearest = node
                min_dist = dist
        return nearest

    def steer(self, from_node, to_node):
        """Steer from 'from_node' towards 'to_node' by step_size."""
        distance = self.distance(from_node, to_node)
        if distance < self.step_size:
            return to_node
        else:
            theta = math.atan2(to_node.position[1] - from_node.position[1],
                               to_node.position[0] - from_node.position[0])
            new_i = int(from_node.position[0] + self.step_size * math.cos(theta))
            new_j = int(from_node.position[1] + self.step_size * math.sin(theta))
            new_i = max(0, min(new_i, self.rows - 1))
            new_j = max(0, min(new_j, self.cols - 1))
            return Node((new_i, new_j))

    def is_collision_free(self, node1, node2):
        """Check if the path between two nodes is free from obstacles."""
        i1, j1 = node1.position
        i2, j2 = node2.position
        # Bresenham's line algorithm
        points = self.get_line(i1, j1, i2, j2)
        for point in points:
            i, j = point
            if i >= self.rows or j >= self.cols or i < 0 or j < 0:
                return False  # Out of bounds
            if self.grid[i][j] == 255:
                return False
        return True

    def get_line(self, i1, j1, i2, j2):
        """Generate points on the line from (i1,j1) to (i2,j2) using Bresenham's algorithm."""
        points = []
        di = abs(i2 - i1)
        dj = abs(j2 - j1)
        si = 1 if i2 > i1 else -1
        sj = 1 if j2 > j1 else -1
        if dj > di:
            di, dj = dj, di
            steep = True
        else:
            steep = False
        error = 2 * dj - di
        i, j = i1, j1
        for _ in range(di):
            points.append((i, j))
            if steep:
                i += si
            else:
                j += sj
            if error > 0:
                if steep:
                    j += sj
                else:
                    i += si
                error -= 2 * di
            error += 2 * dj
        points.append((i2, j2))
        return points

    def get_near_nodes(self, new_node):
        """Find all nodes within search_radius of new_node."""
        near_nodes = []
        for node in self.nodes:
            if self.distance(node, new_node) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def choose_parent(self, near_nodes, new_node):
        """Choose the best parent for new_node from near_nodes."""
        min_cost = float('inf')
        best_parent = None
        for node in near_nodes:
            if self.is_collision_free(node, new_node):
                cost = self.distance(self.start, node) + self.distance(node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
        return best_parent

    def rewire(self, near_nodes, new_node):
        """Rewire the tree if a better path is found through new_node."""
        for node in near_nodes:
            if self.is_collision_free(new_node, node):
                cost_through_new = self.distance(self.start, new_node) + self.distance(new_node, node)
                current_cost = self.distance(self.start, node)
                if cost_through_new < current_cost:
                    node.parent = new_node

    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        path = []
        current = self.goal
        while current != self.start:
            path.append(current.position)
            current = current.parent
            if current is None:
                return None  # Path reconstruction failed
        path.append(self.start.position)
        path.reverse()
        return path

    @staticmethod
    def distance(node1, node2):
        """Euclidean distance between two nodes."""
        return math.hypot(node1.position[0] - node2.position[0], node1.position[1] - node2.position[1])
