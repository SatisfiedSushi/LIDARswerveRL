# d_star_lite.py

"""
D* Lite grid planning
author: vss2sn (28676655+vss2sn@users.noreply.github.com)
Link to papers:
D* Lite (Link: http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)
Improved Fast Replanning for Robot Navigation in Unknown Terrain
(Link: http://www.cs.cmu.edu/~maxim/files/dlite_icra02.pdf)
Implemented maintaining similarity with the pseudocode for understanding.
Code can be significantly optimized by using a priority queue for U, etc.
Avoiding additional imports based on repository philosophy.
"""

import math
import numpy as np

class Node:
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost

def add_coordinates(node1: Node, node2: Node):
    new_node = Node()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node

def compare_coordinates(node1: Node, node2: Node):
    return node1.x == node2.x and node1.y == node2.y

class DStarLite:

    # Please adjust the heuristic function (h) if you change the list of
    # possible motions
    motions = [
        Node(1, 0, 1),
        Node(0, 1, 1),
        Node(-1, 0, 1),
        Node(0, -1, 1),
        Node(1, 1, math.sqrt(2)),
        Node(1, -1, math.sqrt(2)),
        Node(-1, 1, math.sqrt(2)),
        Node(-1, -1, math.sqrt(2))
    ]

    def __init__(self, ox: list, oy: list, x_max: int, y_max: int):
        # Ensure that within the algorithm implementation all node coordinates
        # are indices in the grid and extend
        # from 0 to x_max and y_max
        self.x_min_world = 0
        self.y_min_world = 0
        self.x_max = x_max
        self.y_max = y_max
        self.obstacles = [Node(x, y) for x, y in zip(ox, oy)]
        self.obstacles_xy = np.array(
            [[obstacle.x, obstacle.y] for obstacle in self.obstacles]
        )
        self.start = Node(0, 0)
        self.goal = Node(0, 0)
        self.U = list()
        self.km = 0.0
        self.kold = 0.0
        self.rhs = self.create_grid(float("inf"))
        self.g = self.create_grid(float("inf"))
        self.detected_obstacles_xy = np.empty((0, 2))
        self.xy = np.empty((0, 2))
        self.initialized = False

    def create_grid(self, val: float):
        return np.full((self.x_max, self.y_max), val)

    def is_obstacle(self, node: Node):
        x = node.x
        y = node.y
        is_in_obstacles = any((self.obstacles_xy[:, 0] == x) & (self.obstacles_xy[:, 1] == y))
        return is_in_obstacles

    def c(self, node1: Node, node2: Node):
        if self.is_obstacle(node2):
            # Attempting to move from or to an obstacle
            return math.inf
        new_node = Node(node1.x - node2.x, node1.y - node2.y)
        detected_motion = list(filter(lambda motion:
                                      compare_coordinates(motion, new_node),
                                      self.motions))
        return detected_motion[0].cost

    def h(self, s: Node):
        # Heuristic function (Euclidean distance)
        return math.hypot(self.start.x - s.x, self.start.y - s.y)

    def calculate_key(self, s: Node):
        return (min(self.g[s.x][s.y], self.rhs[s.x][s.y]) + self.h(s)
                + self.km, min(self.g[s.x][s.y], self.rhs[s.x][s.y]))

    def is_valid(self, node: Node):
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def get_neighbours(self, u: Node):
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def pred(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def succ(self, u: Node):
        # Grid, so each vertex is connected to the ones around it
        return self.get_neighbours(u)

    def initialize(self, start: Node, goal: Node):
        self.start = start
        self.goal = goal
        if not self.initialized:
            self.initialized = True
            self.U = list()  # Would normally be a priority queue
            self.km = 0.0
            self.rhs = self.create_grid(math.inf)
            self.g = self.create_grid(math.inf)
            self.rhs[self.goal.x][self.goal.y] = 0
            self.U.append((self.goal, self.calculate_key(self.goal)))
            self.detected_obstacles_xy = np.empty((0, 2))

    def update_vertex(self, u: Node):
        if not compare_coordinates(u, self.goal):
            self.rhs[u.x][u.y] = min([self.c(u, sprime) +
                                      self.g[sprime.x][sprime.y]
                                      for sprime in self.succ(u)])
        self.U = [(node, key) for node, key in self.U
                  if not compare_coordinates(node, u)]
        if self.g[u.x][u.y] != self.rhs[u.x][u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key_pair1: tuple, key_pair2: tuple):
        return key_pair1[0] < key_pair2[0] or \
               (key_pair1[0] == key_pair2[0] and key_pair1[1] < key_pair2[1])

    def compute_shortest_path(self):
        self.U.sort(key=lambda x: x[1])
        while (len(self.U) > 0 and
               (self.compare_keys(self.U[0][1], self.calculate_key(self.start)) or
                self.rhs[self.start.x][self.start.y] != self.g[self.start.x][self.start.y])):
            k_old = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)
            if self.compare_keys(k_old, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x][u.y] > self.rhs[u.x][u.y]:
                self.g[u.x][u.y] = self.rhs[u.x][u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x][u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)

    def compute_current_path(self):
        path = []
        current = self.start
        while not compare_coordinates(current, self.goal):
            path.append(current)
            if self.g[current.x][current.y] == math.inf:
                print("No path possible")
                return []
            next_nodes = self.succ(current)
            min_cost = math.inf
            next_node = None
            for node in next_nodes:
                cost = self.c(current, node) + self.g[node.x][node.y]
                if cost < min_cost:
                    min_cost = cost
                    next_node = node
            if next_node is None:
                print("No path possible")
                return []
            current = next_node
        path.append(self.goal)
        return path

    def main(self, start: Node, goal: Node,
             spoofed_ox: list, spoofed_oy: list):
        # This method is kept for compatibility but not used in this context
        pass
