# Simulations/visual_dynamic_simulation.py

import tkinter as tk
import time
import math
import logging
import numpy as np
from tkinter import messagebox

from Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class VisualDynamicSimulation(tk.Toplevel):
    def __init__(self, master, grid, start, goal, algorithms, dynamic_obstacles, obstacle_speed=2, cell_size=10, grid_size=500):
        """
        Initialize the visual dynamic simulation.

        :param master: Parent Tkinter window.
        :param grid: 2D numpy array representing the environment.
        :param start: Start position tuple (x, y) in pixels.
        :param goal: Goal position tuple (x, y) in pixels.
        :param algorithms: List of path planning algorithm classes.
        :param dynamic_obstacles: List of dynamic obstacle dictionaries with positions and directions.
        :param obstacle_speed: Speed at which dynamic obstacles move.
        :param cell_size: Size of each grid cell in pixels.
        :param grid_size: Size of the grid in pixels.
        """
        super().__init__(master)
        self.title("Dynamic Simulation Visual")
        self.geometry(f"{grid_size}x{grid_size + 50}")  # Extra space for algorithm label
        self.resizable(False, False)

        self.grid = grid
        self.start = start
        self.goal = goal
        self.algorithms = algorithms
        self.current_algo_index = 0
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_speed = obstacle_speed
        self.cell_size = cell_size
        self.grid_size = grid_size

        # Simulation State
        self.robot = {'x': start[0], 'y': start[1], 'path': [], 'path_index': 0}
        self.path = []
        self.running = True

        # Store Initial States for Resetting
        self.initial_dynamic_obstacles = [obs.copy() for obs in self.dynamic_obstacles]
        self.initial_robot_position = {'x': start[0], 'y': start[1]}

        # Create Canvas
        self.canvas = tk.Canvas(self, width=grid_size, height=grid_size, bg='white')
        self.canvas.pack()

        # Label to display current algorithm
        self.algo_label = tk.Label(self, text="", font=("Helvetica", 12, "bold"))
        self.algo_label.pack(pady=5)

        # Draw Start and Goal
        self.draw_start_goal()

        # Draw Obstacles
        self.draw_obstacles()

        # Initialize Robot
        self.robot_obj = self.canvas.create_oval(
            self.robot['x']-10, self.robot['y']-10,
            self.robot['x']+10, self.robot['y']+10,
            fill='blue'
        )

        # Initialize Path Line
        self.path_line = None

        # Start the simulation
        self.start_simulation()

        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def draw_start_goal(self):
        # Draw Start
        self.canvas.create_oval(
            self.start[0]-10, self.start[1]-10,
            self.start[0]+10, self.start[1]+10,
            fill='green', outline='black'
        )
        self.canvas.create_text(self.start[0], self.start[1], text="Start", fill="white")

        # Draw Goal
        self.canvas.create_oval(
            self.goal[0]-10, self.goal[1]-10,
            self.goal[0]+10, self.goal[1]+10,
            fill='red', outline='black'
        )
        self.canvas.create_text(self.goal[0], self.goal[1], text="Goal", fill="white")

        # Draw buffer around Start
        buffer_size = 30
        self.start_buffer_coords = (
            self.start[0]-buffer_size, self.start[1]-buffer_size,
            self.start[0]+buffer_size, self.start[1]+buffer_size
        )
        self.start_buffer = self.canvas.create_rectangle(
            *self.start_buffer_coords,
            fill='', outline='blue', dash=(2, 2)
        )

        # Draw buffer around Goal
        self.goal_buffer_coords = (
            self.goal[0]-buffer_size, self.goal[1]-buffer_size,
            self.goal[0]+buffer_size, self.goal[1]+buffer_size
        )
        self.goal_buffer = self.canvas.create_rectangle(
            *self.goal_buffer_coords,
            fill='', outline='blue', dash=(2, 2)
        )

    def draw_obstacles(self):
        self.obstacle_rects = []
        for obs in self.dynamic_obstacles:
            rect = self.canvas.create_rectangle(
                obs['x'], obs['y'],
                obs['x'] + obs['size'], obs['y'] + obs['size'],
                fill='gray', outline='black'
            )
            self.obstacle_rects.append(rect)

    def start_simulation(self):
        if self.current_algo_index >= len(self.algorithms):
            logging.error("No algorithms available to run the simulation.")
            messagebox.showerror("No Algorithms", "No algorithms available to run the simulation.")
            self.destroy()
            return

        self.current_algorithm = self.algorithms[self.current_algo_index]
        self.algo_label.config(text=f"Current Algorithm: {self.current_algorithm.__name__}")
        logging.info(f"Starting simulation with {self.current_algorithm.__name__}")
        try:
            self.plan_and_draw_path()
            self.after(100, self.run_simulation)
        except Exception as e:
            logging.error(f"Algorithm {self.current_algorithm.__name__} failed with error: {e}")
            self.switch_algorithm()

    def run_simulation(self):
        """
        Run the visual dynamic simulation loop.
        """
        if not self.running:
            return

        # Move dynamic obstacles
        self.move_dynamic_obstacles()

        # Check for path blockage and replanning
        if self.check_path_blocked():
            logging.info(f"Path blocked for {self.current_algorithm.__name__}. Replanning...")
            try:
                replanned = self.plan_and_draw_path()
                if not replanned:
                    logging.warning(f"No path found during replanning for {self.current_algorithm.__name__}.")
                    self.switch_algorithm()
                    return
            except Exception as e:
                logging.error(f"Replanning with {self.current_algorithm.__name__} failed: {e}")
                self.switch_algorithm()
                return

        # Move robot along the path
        self.move_robot()

        # Check if robot has reached the goal
        if math.hypot(self.robot['x'] - self.goal[0], self.robot['y'] - self.goal[1]) < 10:
            logging.info(f"Robot has reached the goal using {self.current_algorithm.__name__}!")
            self.running = False
            try:
                messagebox.showinfo("Success", f"Robot reached the goal using {self.current_algorithm.__name__}!")
            except:
                pass  # Window might be destroyed
            self.destroy()
            return

        # Schedule the next simulation step
        self.after(50, self.run_simulation)  # Adjust delay as needed for simulation speed

    def move_dynamic_obstacles(self):
        """
        Update positions of dynamic obstacles and handle bouncing.
        """
        for idx, obs in enumerate(self.dynamic_obstacles):
            # Update position
            obs['x'] += obs['dir'][0] * self.obstacle_speed
            obs['y'] += obs['dir'][1] * self.obstacle_speed

            # Bounce off walls
            if obs['x'] <= 0 or obs['x'] + obs['size'] >= self.grid_size:
                obs['dir'] = (-obs['dir'][0], obs['dir'][1])
                # Clamp position within bounds
                obs['x'] = max(min(obs['x'], self.grid_size - obs['size']), 0)
            if obs['y'] <= 0 or obs['y'] + obs['size'] >= self.grid_size:
                obs['dir'] = (obs['dir'][0], -obs['dir'][1])
                obs['y'] = max(min(obs['y'], self.grid_size - obs['size']), 0)

            # Bounce off other obstacles
            for jdx, other_obs in enumerate(self.dynamic_obstacles):
                if idx == jdx:
                    continue
                if self.check_overlap(obs, other_obs):
                    # Reverse directions
                    obs['dir'] = (-obs['dir'][0], -obs['dir'][1])
                    other_obs['dir'] = (-other_obs['dir'][0], -other_obs['dir'][1])

                    # Adjust positions to prevent sticking
                    obs['x'] += obs['dir'][0] * self.obstacle_speed
                    obs['y'] += obs['dir'][1] * self.obstacle_speed
                    other_obs['x'] += other_obs['dir'][0] * self.obstacle_speed
                    other_obs['y'] += other_obs['dir'][1] * self.obstacle_speed

            # Bounce off start buffer
            if self.check_collision_with_buffer(obs, self.start_buffer_coords):
                obs['dir'] = (-obs['dir'][0], -obs['dir'][1])
                # Adjust position
                obs['x'] += obs['dir'][0] * self.obstacle_speed
                obs['y'] += obs['dir'][1] * self.obstacle_speed

            # Bounce off goal buffer
            if self.check_collision_with_buffer(obs, self.goal_buffer_coords):
                obs['dir'] = (-obs['dir'][0], -obs['dir'][1])
                # Adjust position
                obs['x'] += obs['dir'][0] * self.obstacle_speed
                obs['y'] += obs['dir'][1] * self.obstacle_speed

            # Update the canvas rectangle position
            self.canvas.coords(
                self.obstacle_rects[idx],
                obs['x'], obs['y'],
                obs['x'] + obs['size'], obs['y'] + obs['size']
            )

    def check_overlap(self, obs1, obs2):
        """
        Check if two obstacles overlap.

        :param obs1: First obstacle dictionary.
        :param obs2: Second obstacle dictionary.
        :return: True if overlapping, False otherwise.
        """
        return not (obs1['x'] + obs1['size'] < obs2['x'] or
                    obs1['x'] > obs2['x'] + obs2['size'] or
                    obs1['y'] + obs1['size'] < obs2['y'] or
                    obs1['y'] > obs2['y'] + obs2['size'])

    def check_collision_with_buffer(self, obs, buffer_coords):
        """
        Check if an obstacle collides with a buffer area.

        :param obs: Obstacle dictionary.
        :param buffer_coords: Tuple (x1, y1, x2, y2) defining the buffer rectangle.
        :return: True if collides, False otherwise.
        """
        obs_rect = (obs['x'], obs['y'], obs['x'] + obs['size'], obs['y'] + obs['size'])
        buffer_rect = buffer_coords
        return self.rect_overlap(obs_rect, buffer_rect)

    @staticmethod
    def rect_overlap(rect1, rect2):
        """
        Check if two rectangles overlap.
        rect = (x1, y1, x2, y2)
        """
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or
                    rect1[3] < rect2[1] or rect1[1] > rect2[3])

    def check_path_blocked(self):
        """
        Check if the current path is blocked by any dynamic obstacle.

        :return: True if blocked, False otherwise.
        """
        if not self.path:
            return True

        for obs in self.dynamic_obstacles:
            obs_rect = (obs['x'], obs['y'], obs['x'] + obs['size'], obs['y'] + obs['size'])
            for point in self.path:
                x, y = point
                if obs['x'] <= x <= obs['x'] + obs['size'] and obs['y'] <= y <= obs['y'] + obs['size']:
                    return True
        return False

    def plan_and_draw_path(self):
        """
        Plan a new path using the current path planning algorithm and draw it on the canvas.

        :return: True if path found, False otherwise.
        """
        # Convert pixel coordinates to grid indices
        start_grid = (int(self.robot['y'] // self.cell_size), int(self.robot['x'] // self.cell_size))
        goal_grid = (int(self.goal[1] // self.cell_size), int(self.goal[0] // self.cell_size))

        # Initialize and run the algorithm
        if self.current_algorithm.__name__ == "OkayPlan":
            # Define parameters for OkayPlan
            params = [1.0] * 19  # Example: Replace with actual parameters
            planner = self.current_algorithm(grid=self.grid, start=start_grid, goal=goal_grid, params=params)
            path, collision = planner.plan(env_info={
                'start_point': start_grid,
                'target_point': goal_grid,
                'd2target': heuristic(start_grid, goal_grid),
                'Obs_Segments': [],  # Populate if necessary
                'Flat_pdct_segments': []  # Populate if necessary
            })
        else:
            planner = self.current_algorithm(grid=self.grid, start=start_grid, goal=goal_grid)
            path = planner.run()
            collision = False  # Adjust based on algorithm's output

        if path:
            # Convert grid path back to pixel coordinates
            path_pixels = [(j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2) for i, j in path]
            self.path = path_pixels
            self.robot['path'] = path_pixels
            self.robot['path_index'] = 0
            self.draw_path()
            return True
        else:
            self.path = []
            self.robot['path'] = []
            self.robot['path_index'] = 0
            if self.path_line:
                self.canvas.delete(self.path_line)
                self.path_line = None
            return False

    def draw_path(self):
        """
        Draw the planned path on the canvas.
        """
        if self.path_line:
            self.canvas.delete(self.path_line)

        if len(self.path) > 1:
            points = []
            for point in self.path:
                points.extend(point)
            self.path_line = self.canvas.create_line(
                *points,
                fill='green',
                width=2,
                dash=(4, 2)
            )

    def move_robot(self):
        """
        Move the robot along the planned path.
        """
        if self.robot['path_index'] >= len(self.path):
            return  # Path completed

        target = self.path[self.robot['path_index']]
        current_x, current_y = self.robot['x'], self.robot['y']
        target_x, target_y = target

        # Calculate direction
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.hypot(dx, dy)

        if distance < 2:
            # Move to the next waypoint
            self.robot['path_index'] += 1
            return

        # Normalize direction
        dx /= distance
        dy /= distance

        # Move robot
        self.robot['x'] += dx * 2  # Robot speed (pixels per frame)
        self.robot['y'] += dy * 2

        # Update robot position on canvas
        self.canvas.coords(
            self.robot_obj,
            self.robot['x'] - 10, self.robot['y'] - 10,
            self.robot['x'] + 10, self.robot['y'] + 10
        )

    def switch_algorithm(self):
        """
        Switch to the next algorithm in the list. If all algorithms fail, close the simulation.
        """
        self.current_algo_index += 1
        if self.current_algo_index < len(self.algorithms):
            logging.info(f"Switching to next algorithm: {self.algorithms[self.current_algo_index].__name__}")
            self.current_algorithm = self.algorithms[self.current_algo_index]
            self.algo_label.config(text=f"Current Algorithm: {self.current_algorithm.__name__}")

            # Reset Robot and Obstacles to Initial States
            self.reset_simulation_state()

            # Attempt to plan and run with the new algorithm
            try:
                path_found = self.plan_and_draw_path()
                if path_found:
                    self.after(100, self.run_simulation)
                else:
                    # If planning fails immediately, try the next algorithm
                    logging.warning(f"No path found with {self.current_algorithm.__name__}.")
                    self.switch_algorithm()
            except Exception as e:
                logging.error(f"Algorithm {self.current_algorithm.__name__} failed with error: {e}")
                self.switch_algorithm()
        else:
            logging.error("All algorithms failed to find a path. Closing simulation.")
            try:
                messagebox.showerror("Simulation Failed", "All algorithms failed to find a path. Closing simulation.")
            except:
                pass  # In case the window is already destroyed
            self.running = False
            self.destroy()

    def reset_simulation_state(self):
        """
        Reset the robot and dynamic obstacles to their initial positions and states.
        """
        # Reset Robot Position
        self.robot['x'] = self.initial_robot_position['x']
        self.robot['y'] = self.initial_robot_position['y']
        self.canvas.coords(
            self.robot_obj,
            self.robot['x'] - 10, self.robot['y'] - 10,
            self.robot['x'] + 10, self.robot['y'] + 10
        )

        # Reset Dynamic Obstacles
        for idx, obs in enumerate(self.initial_dynamic_obstacles):
            self.dynamic_obstacles[idx]['x'] = obs['x']
            self.dynamic_obstacles[idx]['y'] = obs['y']
            self.dynamic_obstacles[idx]['dir'] = obs['dir']

            # Update the canvas rectangle position
            self.canvas.coords(
                self.obstacle_rects[idx],
                obs['x'], obs['y'],
                obs['x'] + obs['size'], obs['y'] + obs['size']
            )

        # Reset Path and Path Index
        self.path = []
        self.robot['path'] = []
        self.robot['path_index'] = 0

        # Remove Existing Path Line
        if self.path_line:
            self.canvas.delete(self.path_line)
            self.path_line = None

    def on_closing(self):
        """
        Handle the window closing event.
        """
        self.running = False
        self.destroy()
