# run_benchmarking.py

import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from multiprocessing import Process, Manager
import random
import time
import os
import logging
import numpy as np
import math

from PathPlanningAlgorithms.AStar import AStar
from PathPlanningAlgorithms.RRTStar import RRTStar
from PathPlanningAlgorithms.DStarLite import DStarLite
from PathPlanningAlgorithms.OkayPlan import OkayPlan
from Simulations.static_simulation import StaticSimulation
from Simulations.dynamic_simulation import DynamicSimulation
from Simulations.visual_dynamic_simulation import VisualDynamicSimulation
from PathPlanningAlgorithms.DStarLiteUtils.OccupancyGridMap import OccupancyGridMap
from utils.heuristic import heuristic

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/benchmarking_{int(time.time())}.log"),
        logging.StreamHandler()
    ]
)

# Constants
GRID_SIZE = 500  # Pixels
CELL_SIZE = 10  # Pixels per grid cell
START_POS = (50, 50)  # Pixel coordinates (Top-Left)
GOAL_POS = (450, 450)  # Pixel coordinates (Bottom-Right)


class PathPlanningBenchmarkingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Path Planning Benchmarking")

        # Initialize obstacle list
        self.obstacles = []
        self.dynamic_obstacles = []  # For dynamic simulations

        # Create Canvas
        self.canvas = tk.Canvas(master, width=GRID_SIZE, height=GRID_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=15, padx=10, pady=10)

        # Draw Start and Goal
        self.draw_start_goal()

        # Controls
        # Obstacle Size
        tk.Label(master, text="Obstacle Size (30-60 px):").grid(row=0, column=1, sticky='w')
        self.size_entry = tk.Entry(master)
        self.size_entry.grid(row=0, column=2, padx=5)
        self.size_entry.insert(0, "30")

        # Obstacle X
        tk.Label(master, text="Obstacle X:").grid(row=1, column=1, sticky='w')
        self.x_entry = tk.Entry(master)
        self.x_entry.grid(row=1, column=2, padx=5)
        self.x_entry.insert(0, "100")

        # Obstacle Y
        tk.Label(master, text="Obstacle Y:").grid(row=2, column=1, sticky='w')
        self.y_entry = tk.Entry(master)
        self.y_entry.grid(row=2, column=2, padx=5)
        self.y_entry.insert(0, "100")

        # Add Obstacle Button
        self.add_button = tk.Button(master, text="Add Obstacle", command=self.add_obstacle)
        self.add_button.grid(row=3, column=1, columnspan=2, pady=5)

        # Clear Obstacles Button
        self.clear_button = tk.Button(master, text="Clear Obstacles", command=self.clear_obstacles)
        self.clear_button.grid(row=4, column=1, columnspan=2, pady=5)

        # Number of Obstacles Slider
        tk.Label(master, text="Number of Obstacles:").grid(row=8, column=1, sticky='w')
        self.num_obstacles_slider = tk.Scale(master, from_=1, to=20, orient='horizontal')
        self.num_obstacles_slider.set(5)  # Default value
        self.num_obstacles_slider.grid(row=8, column=2, padx=5)

        # Randomly Place Obstacles Button
        self.random_button = tk.Button(master, text="Randomly Place Obstacles", command=self.randomly_place_obstacles)
        self.random_button.grid(row=9, column=1, columnspan=2, pady=5)

        # Benchmark Settings Frame
        self.settings_frame = tk.LabelFrame(master, text="Benchmark Settings", padx=10, pady=10)
        self.settings_frame.grid(row=10, column=0, columnspan=3, padx=10, pady=10, sticky='we')

        # Number of Trials
        tk.Label(self.settings_frame, text="Number of Trials:").grid(row=0, column=0, sticky='w')
        self.num_trials_entry = tk.Entry(self.settings_frame)
        self.num_trials_entry.grid(row=0, column=1, padx=5)
        self.num_trials_entry.insert(0, "10")  # Default value

        # Algorithm Selection
        tk.Label(self.settings_frame, text="Select Algorithms:").grid(row=1, column=0, sticky='w')
        self.algorithms_var = tk.Variable(value=["AStar", "RRTStar", "DStarLite", "OkayPlan"])
        self.algorithms_listbox = tk.Listbox(self.settings_frame, listvariable=self.algorithms_var,
                                             selectmode='multiple', height=5, exportselection=0)
        self.algorithms_listbox.grid(row=1, column=1, padx=5, pady=5)

        # Select All and Deselect All Buttons
        self.select_all_button = tk.Button(self.settings_frame, text="Select All",
                                           command=lambda: self.select_all(True))
        self.select_all_button.grid(row=1, column=2, padx=5, pady=2)
        self.deselect_all_button = tk.Button(self.settings_frame, text="Deselect All",
                                             command=lambda: self.select_all(False))
        self.deselect_all_button.grid(row=2, column=2, padx=5, pady=2)

        # Benchmark Progress Frame
        self.progress_frame = tk.LabelFrame(master, text="Benchmark Progress", padx=10, pady=10)
        self.progress_frame.grid(row=11, column=0, columnspan=3, padx=10, pady=10, sticky='we')

        # Initialize progress bars as empty; they'll be created when benchmarking starts
        self.progress_bars = {}

        # Benchmark Buttons
        # Run Static Benchmark Button
        self.run_static_button = tk.Button(master, text="Run Static Benchmark", command=self.run_static_benchmark)
        self.run_static_button.grid(row=5, column=1, columnspan=2, pady=5)

        # Run Dynamic Benchmark Button
        self.run_dynamic_button = tk.Button(master, text="Run Dynamic Benchmark", command=self.run_dynamic_benchmark)
        self.run_dynamic_button.grid(row=6, column=1, columnspan=2, pady=5)

        # Show Visual Simulation Button
        self.visual_button = tk.Button(master, text="Show Dynamic Simulation Visual",
                                       command=self.show_visual_simulation)
        self.visual_button.grid(row=7, column=1, columnspan=2, pady=5)

        # Exit Button
        self.exit_button = tk.Button(master, text="Exit", command=master.quit)
        self.exit_button.grid(row=12, column=1, columnspan=2, pady=5)

    def draw_start_goal(self):
        # Draw Start
        self.canvas.create_oval(
            START_POS[0] - 10, START_POS[1] - 10,
            START_POS[0] + 10, START_POS[1] + 10,
            fill='green', outline='black'
        )
        self.canvas.create_text(START_POS[0], START_POS[1], text="Start", fill="white")

        # Draw Goal
        self.canvas.create_oval(
            GOAL_POS[0] - 10, GOAL_POS[1] - 10,
            GOAL_POS[0] + 10, GOAL_POS[1] + 10,
            fill='red', outline='black'
        )
        self.canvas.create_text(GOAL_POS[0], GOAL_POS[1], text="Goal", fill="white")

        # Draw buffer around Start
        buffer_size = 30
        self.start_buffer_coords = (
            START_POS[0] - buffer_size, START_POS[1] - buffer_size,
            START_POS[0] + buffer_size, START_POS[1] + buffer_size
        )
        self.start_buffer = self.canvas.create_rectangle(
            *self.start_buffer_coords,
            fill='', outline='blue', dash=(2, 2)
        )

        # Draw buffer around Goal
        self.goal_buffer_coords = (
            GOAL_POS[0] - buffer_size, GOAL_POS[1] - buffer_size,
            GOAL_POS[0] + buffer_size, GOAL_POS[1] + buffer_size
        )
        self.goal_buffer = self.canvas.create_rectangle(
            *self.goal_buffer_coords,
            fill='', outline='blue', dash=(2, 2)
        )

    def add_obstacle(self):
        try:
            size = int(self.size_entry.get())
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            angle = 0  # Default angle for manual addition

            if size < 30 or size > 60:
                messagebox.showerror("Invalid Size", "Obstacle size must be between 30 and 60 pixels.")
                return
            if x < 0 or x + size > GRID_SIZE or y < 0 or y + size > GRID_SIZE:
                messagebox.showerror("Invalid Position", "Obstacle position out of bounds.")
                return

            # Check overlap with start and goal
            obstacle_rect = (x, y, size)
            if self.check_overlap_with_start_goal(obstacle_rect):
                messagebox.showerror("Overlap Error", "Obstacle overlaps with Start or Goal.")
                return

            # Check overlap with existing obstacles
            if self.check_overlap_with_existing_obstacles(obstacle_rect):
                messagebox.showerror("Overlap Error", "Obstacle overlaps with existing obstacles.")
                return

            # Create axis-aligned rectangle
            rect_id = self.canvas.create_rectangle(x, y, x + size, y + size, fill='gray', outline='black')
            self.obstacles.append({'x': x, 'y': y, 'size': size, 'angle': angle, 'id': rect_id})
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer values for size and position.")

    def clear_obstacles(self):
        self.obstacles = []
        self.dynamic_obstacles = []
        self.canvas.delete("all")
        self.draw_start_goal()

    def randomly_place_obstacles(self):
        """
        Place a specified number of obstacles randomly on the canvas with random sizes and rotations.
        """
        num_obstacles = self.num_obstacles_slider.get()
        self.clear_obstacles()  # Optionally clear existing obstacles

        for _ in range(num_obstacles):
            size = random.randint(30, 60)
            x = random.randint(0, GRID_SIZE - size)
            y = random.randint(0, GRID_SIZE - size)
            angle = random.uniform(0, 360)

            # Ensure no overlap with start and goal
            obstacle_rect = (x, y, size)
            if self.check_overlap_with_start_goal(obstacle_rect):
                continue

            # Ensure no overlap with existing obstacles
            if self.check_overlap_with_existing_obstacles(obstacle_rect):
                continue

            # Create rotated rectangle (for simplicity, using axis-aligned rectangles)
            rect_id = self.canvas.create_rectangle(x, y, x + size, y + size, fill='gray', outline='black')
            self.obstacles.append({'x': x, 'y': y, 'size': size, 'angle': angle, 'id': rect_id})

    def check_overlap_with_start_goal(self, obstacle):
        x, y, size = obstacle
        # Define start and goal rectangles with buffer
        start_rect = (START_POS[0] - 30, START_POS[1] - 30, START_POS[0] + 30, START_POS[1] + 30)
        goal_rect = (GOAL_POS[0] - 30, GOAL_POS[1] - 30, GOAL_POS[0] + 30, GOAL_POS[1] + 30)

        obstacle_rect = (x, y, x + size, y + size)

        # Check overlap
        return self.rect_overlap(obstacle_rect, start_rect) or self.rect_overlap(obstacle_rect, goal_rect)

    def check_overlap_with_existing_obstacles(self, new_obstacle):
        x, y, size = new_obstacle
        new_rect = (x - 10, y - 10, x + size + 10, y + size + 10)  # Buffer of 10 pixels

        for obs in self.obstacles:
            obs_x, obs_y, obs_size = obs['x'], obs['y'], obs['size']
            obs_rect = (obs_x, obs_y, obs_x + obs_size, obs_y + obs_size)
            if self.rect_overlap(new_rect, obs_rect):
                return True
        return False

    @staticmethod
    def rect_overlap(rect1, rect2):
        """
        Check if two rectangles overlap.
        rect = (x1, y1, x2, y2)
        """
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or
                    rect1[3] < rect2[1] or rect1[1] > rect2[3])

    def run_static_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run static benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for static benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='static', grid=grid, algorithms=algorithms)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='static')

    def run_dynamic_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run dynamic benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Define dynamic obstacles (for simplicity, randomly select some obstacles to move)
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for dynamic benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='dynamic', grid=grid, algorithms=algorithms,
                                                        dynamic_obstacles=dynamic_obstacles)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='dynamic')

    def show_visual_simulation(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles",
                                 "Please add at least one obstacle before running the visual simulation.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define dynamic obstacles
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Define algorithms to visualize
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to visualize.")
            return

        # If multiple algorithms are selected, allow user to choose one for the simulation
        if len(selected_algorithms) > 1:
            algo_selection = simpledialog.askstring("Algorithm Selection",
                                                    f"Multiple algorithms selected: {', '.join([algo.__name__ for algo in selected_algorithms])}\nEnter the algorithm name to use for simulation:")
            if algo_selection and algo_selection in [algo.__name__ for algo in selected_algorithms]:
                selected_algorithms = [algo for algo in selected_algorithms if algo.__name__ == algo_selection]
            else:
                messagebox.showerror("Invalid Selection", "No valid algorithm selected. Aborting simulation.")
                return

        algorithms = selected_algorithms  # List containing one algorithm class

        # Launch the visual simulation
        VisualDynamicSimulation(
            master=self.master,
            grid=grid,
            start=START_POS,
            goal=GOAL_POS,
            algorithms=algorithms,  # Pass the list of algorithms
            dynamic_obstacles=dynamic_obstacles,
            obstacle_speed=2,
            cell_size=CELL_SIZE,
            grid_size=GRID_SIZE
        )

    def prepare_grid(self):
        """
        Convert obstacle list to a grid representation.
        """
        grid_size = GRID_SIZE // CELL_SIZE
        grid = np.zeros((grid_size, grid_size), dtype=int)

        for obs in self.obstacles:
            x, y, size = obs['x'], obs['y'], obs['size']
            start_i = y // CELL_SIZE
            start_j = x // CELL_SIZE
            end_i = min((y + size) // CELL_SIZE, grid_size - 1)
            end_j = min((x + size) // CELL_SIZE, grid_size - 1)
            grid[start_i:end_i + 1, start_j:end_j + 1] = 255  # Mark as obstacle
        return grid

    def create_dynamic_obstacles(self):
        """
        Select a subset of obstacles to be dynamic by assigning directions.
        """
        if not self.obstacles:
            return []

        num_dynamic = min(5, len(self.obstacles))  # Limit number of dynamic obstacles
        dynamic_obstacles = []
        selected = random.sample(self.obstacles, num_dynamic)

        for obs in selected:
            x, y, size = obs['x'], obs['y'], obs['size']
            direction = [random.choice([-1, 1]), random.choice([-1, 1])]
            dynamic_obstacles.append({'x': x, 'y': y, 'size': size, 'dir': direction})
        return dynamic_obstacles

    def display_results(self, results, sim_type):
        """
        Display benchmarking results in a new window.
        """
        result_window = tk.Toplevel(self.master)
        result_window.title(f"{sim_type.capitalize()} Simulation Results")

        tk.Label(result_window, text=f"{sim_type.capitalize()} Simulation Benchmark Results",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        for algo, metrics in results.items():
            frame = tk.Frame(result_window)
            frame.pack(pady=5, padx=10, fill='x')

            tk.Label(frame, text=f"Algorithm: {algo}", font=("Helvetica", 12, "underline")).pack(anchor='w')
            tk.Label(frame, text=f" - Average Computation Time: {metrics['computation_time']:.4f} seconds").pack(
                anchor='w')
            tk.Label(frame, text=f" - Average Path Length: {metrics['path_length']:.2f} pixels").pack(anchor='w')
            tk.Label(frame, text=f" - Average Accuracy: {metrics['accuracy']:.2f}%").pack(anchor='w')
            tk.Label(frame, text=f" - Collision Rate: {metrics['collision_rate']:.2f}%").pack(anchor='w')

        # Optionally, save results to a file
        save_button = tk.Button(result_window, text="Save Results",
                                command=lambda: self.save_results(results, sim_type))
        save_button.pack(pady=10)

    def save_results(self, results, sim_type):
        """
        Save benchmarking results to a text file.
        """
        timestamp = int(time.time())
        filename = f"{sim_type}_benchmark_{timestamp}.txt"
        if not os.path.exists('results'):
            os.makedirs('results')
        filepath = os.path.join('results', filename)

        with open(filepath, 'w') as f:
            f.write(f"{sim_type.capitalize()} Simulation Benchmark Results\n")
            f.write(f"Timestamp: {time.ctime(timestamp)}\n\n")
            for algo, metrics in results.items():
                f.write(f"Algorithm: {algo}\n")
                f.write(f" - Average Computation Time: {metrics['computation_time']:.4f} seconds\n")
                f.write(f" - Average Path Length: {metrics['path_length']:.2f} pixels\n")
                f.write(f" - Average Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f" - Collision Rate: {metrics['collision_rate']:.2f}%\n\n")

        messagebox.showinfo("Results Saved", f"Benchmark results saved to {filepath}")

    def update_progress(self, algo_names, current_trial, total_trials):
        """
        Update the progress bars for each algorithm.

        :param algo_names: List of algorithm names.
        :param current_trial: Current trial number.
        :param total_trials: Total number of trials.
        """
        for algo_name in algo_names:
            if algo_name not in self.progress_bars:
                # Create a new progress bar for the algorithm
                label = tk.Label(self.progress_frame, text=algo_name)
                label.pack(anchor='w')
                progress = ttk.Progressbar(self.progress_frame, orient='horizontal', length=300, mode='determinate')
                progress.pack(pady=2)
                progress['maximum'] = total_trials
                self.progress_bars[algo_name] = progress

            # Update progress
            self.progress_bars[algo_name]['value'] = current_trial

    def get_selected_algorithms(self):
        """
        Retrieve the list of selected algorithms from the listbox.

        :return: List of algorithm classes.
        """
        selected_indices = self.algorithms_listbox.curselection()
        selected_algos = []
        algo_map = {
            "AStar": AStar,
            "RRTStar": RRTStar,
            "DStarLite": DStarLite,
            "OkayPlan": OkayPlan
        }
        for idx in selected_indices:
            algo_name = self.algorithms_var.get()[idx]
            algo_class = algo_map.get(algo_name)
            if algo_class:
                selected_algos.append(algo_class)
        return selected_algos

    def select_all(self, select=True):
        """
        Select or deselect all items in the algorithms listbox.

        :param select: Boolean indicating whether to select or deselect.
        """
        if select:
            self.algorithms_listbox.select_set(0, tk.END)
        else:
            self.algorithms_listbox.select_clear(0, tk.END)

    def run_benchmark_parallel(self, sim_type, grid, algorithms, dynamic_obstacles=None):
        """
        Runs simulations for multiple algorithms in parallel.

        :param sim_type: 'static' or 'dynamic'
        :param grid: 2D numpy array representing the environment.
        :param algorithms: List of path planning algorithm classes.
        :param dynamic_obstacles: List of dynamic obstacle dictionaries (for dynamic simulations).
        :return: Dictionary containing results for each algorithm.
        """
        manager = Manager()
        results = manager.dict()
        processes = []

        for algo in algorithms:
            p = Process(target=simulate_and_record, args=(sim_type, grid, algo, dynamic_obstacles, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return dict(results)

    def run_static_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run static benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for static benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='static', grid=grid, algorithms=algorithms)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='static')

    def run_dynamic_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run dynamic benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Define dynamic obstacles (for simplicity, randomly select some obstacles to move)
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for dynamic benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='dynamic', grid=grid, algorithms=algorithms,
                                                        dynamic_obstacles=dynamic_obstacles)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='dynamic')

    def show_visual_simulation(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles",
                                 "Please add at least one obstacle before running the visual simulation.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define dynamic obstacles
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Define algorithms to visualize
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to visualize.")
            return

        # If multiple algorithms are selected, allow user to choose one for the simulation
        if len(selected_algorithms) > 1:
            algo_selection = simpledialog.askstring("Algorithm Selection",
                                                    f"Multiple algorithms selected: {', '.join([algo.__name__ for algo in selected_algorithms])}\nEnter the algorithm name to use for simulation:")
            if algo_selection and algo_selection in [algo.__name__ for algo in selected_algorithms]:
                selected_algorithms = [algo for algo in selected_algorithms if algo.__name__ == algo_selection]
            else:
                messagebox.showerror("Invalid Selection", "No valid algorithm selected. Aborting simulation.")
                return

        algorithms = selected_algorithms  # List containing one algorithm class

        # Launch the visual simulation
        VisualDynamicSimulation(
            master=self.master,
            grid=grid,
            start=START_POS,
            goal=GOAL_POS,
            algorithms=algorithms,  # Pass the list of algorithms
            dynamic_obstacles=dynamic_obstacles,
            obstacle_speed=2,
            cell_size=CELL_SIZE,
            grid_size=GRID_SIZE
        )

    def prepare_grid(self):
        """
        Convert obstacle list to a grid representation.
        """
        grid_size = GRID_SIZE // CELL_SIZE
        grid = np.zeros((grid_size, grid_size), dtype=int)

        for obs in self.obstacles:
            x, y, size = obs['x'], obs['y'], obs['size']
            start_i = y // CELL_SIZE
            start_j = x // CELL_SIZE
            end_i = min((y + size) // CELL_SIZE, grid_size - 1)
            end_j = min((x + size) // CELL_SIZE, grid_size - 1)
            grid[start_i:end_i + 1, start_j:end_j + 1] = 255  # Mark as obstacle
        return grid

    def create_dynamic_obstacles(self):
        """
        Select a subset of obstacles to be dynamic by assigning directions.
        """
        if not self.obstacles:
            return []

        num_dynamic = min(5, len(self.obstacles))  # Limit number of dynamic obstacles
        dynamic_obstacles = []
        selected = random.sample(self.obstacles, num_dynamic)

        for obs in selected:
            x, y, size = obs['x'], obs['y'], obs['size']
            direction = [random.choice([-1, 1]), random.choice([-1, 1])]
            dynamic_obstacles.append({'x': x, 'y': y, 'size': size, 'dir': direction})
        return dynamic_obstacles

    def display_results(self, results, sim_type):
        """
        Display benchmarking results in a new window.
        """
        result_window = tk.Toplevel(self.master)
        result_window.title(f"{sim_type.capitalize()} Simulation Results")

        tk.Label(result_window, text=f"{sim_type.capitalize()} Simulation Benchmark Results",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        for algo, metrics in results.items():
            frame = tk.Frame(result_window)
            frame.pack(pady=5, padx=10, fill='x')

            tk.Label(frame, text=f"Algorithm: {algo}", font=("Helvetica", 12, "underline")).pack(anchor='w')
            tk.Label(frame, text=f" - Average Computation Time: {metrics['computation_time']:.4f} seconds").pack(
                anchor='w')
            tk.Label(frame, text=f" - Average Path Length: {metrics['path_length']:.2f} pixels").pack(anchor='w')
            tk.Label(frame, text=f" - Average Accuracy: {metrics['accuracy']:.2f}%").pack(anchor='w')
            tk.Label(frame, text=f" - Collision Rate: {metrics['collision_rate']:.2f}%").pack(anchor='w')

        # Optionally, save results to a file
        save_button = tk.Button(result_window, text="Save Results",
                                command=lambda: self.save_results(results, sim_type))
        save_button.pack(pady=10)

    def save_results(self, results, sim_type):
        """
        Save benchmarking results to a text file.
        """
        timestamp = int(time.time())
        filename = f"{sim_type}_benchmark_{timestamp}.txt"
        if not os.path.exists('results'):
            os.makedirs('results')
        filepath = os.path.join('results', filename)

        with open(filepath, 'w') as f:
            f.write(f"{sim_type.capitalize()} Simulation Benchmark Results\n")
            f.write(f"Timestamp: {time.ctime(timestamp)}\n\n")
            for algo, metrics in results.items():
                f.write(f"Algorithm: {algo}\n")
                f.write(f" - Average Computation Time: {metrics['computation_time']:.4f} seconds\n")
                f.write(f" - Average Path Length: {metrics['path_length']:.2f} pixels\n")
                f.write(f" - Average Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f" - Collision Rate: {metrics['collision_rate']:.2f}%\n\n")

        messagebox.showinfo("Results Saved", f"Benchmark results saved to {filepath}")

    def update_progress(self, algo_names, current_trial, total_trials):
        """
        Update the progress bars for each algorithm.

        :param algo_names: List of algorithm names.
        :param current_trial: Current trial number.
        :param total_trials: Total number of trials.
        """
        for algo_name in algo_names:
            if algo_name not in self.progress_bars:
                # Create a new progress bar for the algorithm
                label = tk.Label(self.progress_frame, text=algo_name)
                label.pack(anchor='w')
                progress = ttk.Progressbar(self.progress_frame, orient='horizontal', length=300, mode='determinate')
                progress.pack(pady=2)
                progress['maximum'] = total_trials
                self.progress_bars[algo_name] = progress

            # Update progress
            self.progress_bars[algo_name]['value'] = current_trial

    def get_selected_algorithms(self):
        """
        Retrieve the list of selected algorithms from the listbox.

        :return: List of algorithm classes.
        """
        selected_indices = self.algorithms_listbox.curselection()
        selected_algos = []
        algo_map = {
            "AStar": AStar,
            "RRTStar": RRTStar,
            "DStarLite": DStarLite,
            "OkayPlan": OkayPlan
        }
        for idx in selected_indices:
            algo_name = self.algorithms_var.get()[idx]
            algo_class = algo_map.get(algo_name)
            if algo_class:
                selected_algos.append(algo_class)
        return selected_algos

    def select_all(self, select=True):
        """
        Select or deselect all items in the algorithms listbox.

        :param select: Boolean indicating whether to select or deselect.
        """
        if select:
            self.algorithms_listbox.select_set(0, tk.END)
        else:
            self.algorithms_listbox.select_clear(0, tk.END)

    def run_benchmark_parallel(self, sim_type, grid, algorithms, dynamic_obstacles=None):
        """
        Runs simulations for multiple algorithms in parallel.

        :param sim_type: 'static' or 'dynamic'
        :param grid: 2D numpy array representing the environment.
        :param algorithms: List of path planning algorithm classes.
        :param dynamic_obstacles: List of dynamic obstacle dictionaries (for dynamic simulations).
        :return: Dictionary containing results for each algorithm.
        """
        manager = Manager()
        results = manager.dict()
        processes = []

        for algo in algorithms:
            p = Process(target=simulate_and_record, args=(sim_type, grid, algo, dynamic_obstacles, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return dict(results)

    def run_static_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run static benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for static benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='static', grid=grid, algorithms=algorithms)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='static')

    def run_dynamic_benchmark(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles", "Please add at least one obstacle before benchmarking.")
            return

        # Confirm action
        if not messagebox.askyesno("Confirm", "Run dynamic benchmark with current obstacles?"):
            return

        # Get number of trials
        try:
            num_trials = int(self.num_trials_entry.get())
            if num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of trials must be a positive integer.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define algorithms to benchmark
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to benchmark.")
            return

        algorithms = selected_algorithms  # List of algorithm classes

        # Define dynamic obstacles (for simplicity, randomly select some obstacles to move)
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Run simulations for the specified number of trials
        results = {algo.__name__: {'computation_time': [], 'path_length': [], 'accuracy': [], 'collision': []} for algo
                   in algorithms}

        for trial in range(1, num_trials + 1):
            logging.info(f"Starting Trial {trial}/{num_trials} for dynamic benchmark.")
            trial_results = self.run_benchmark_parallel(sim_type='dynamic', grid=grid, algorithms=algorithms,
                                                        dynamic_obstacles=dynamic_obstacles)
            for algo in algorithms:
                algo_name = algo.__name__
                metrics = trial_results.get(algo_name, {})
                results[algo_name]['computation_time'].append(metrics.get('computation_time', 0))
                results[algo_name]['path_length'].append(metrics.get('path_length', 0))
                results[algo_name]['accuracy'].append(metrics.get('accuracy', 0))
                results[algo_name]['collision'].append(metrics.get('collision', False))
            self.update_progress(algo_names=[algo.__name__ for algo in algorithms], current_trial=trial,
                                 total_trials=num_trials)

        # Aggregate results
        aggregated_results = {}
        for algo in algorithms:
            algo_name = algo.__name__
            aggregated_results[algo_name] = {
                'computation_time': sum(results[algo_name]['computation_time']) / num_trials,
                'path_length': sum(results[algo_name]['path_length']) / num_trials,
                'accuracy': sum(results[algo_name]['accuracy']) / num_trials,
                'collision_rate': sum(results[algo_name]['collision']) / num_trials * 100  # Percentage
            }

        # Display results
        self.display_results(aggregated_results, sim_type='dynamic')

    def show_visual_simulation(self):
        if not self.obstacles:
            messagebox.showerror("No Obstacles",
                                 "Please add at least one obstacle before running the visual simulation.")
            return

        # Prepare grid based on obstacles
        grid = self.prepare_grid()

        # Define dynamic obstacles
        dynamic_obstacles = self.create_dynamic_obstacles()

        # Define algorithms to visualize
        selected_algorithms = self.get_selected_algorithms()
        if not selected_algorithms:
            messagebox.showerror("No Algorithms Selected", "Please select at least one algorithm to visualize.")
            return

        # If multiple algorithms are selected, allow user to choose one for the simulation
        if len(selected_algorithms) > 1:
            algo_selection = simpledialog.askstring("Algorithm Selection",
                                                    f"Multiple algorithms selected: {', '.join([algo.__name__ for algo in selected_algorithms])}\nEnter the algorithm name to use for simulation:")
            if algo_selection and algo_selection in [algo.__name__ for algo in selected_algorithms]:
                selected_algorithms = [algo for algo in selected_algorithms if algo.__name__ == algo_selection]
            else:
                messagebox.showerror("Invalid Selection", "No valid algorithm selected. Aborting simulation.")
                return

        algorithms = selected_algorithms  # List containing one algorithm class

        # Launch the visual simulation
        VisualDynamicSimulation(
            master=self.master,
            grid=grid,
            start=START_POS,
            goal=GOAL_POS,
            algorithms=algorithms,  # Pass the list of algorithms
            dynamic_obstacles=dynamic_obstacles,
            obstacle_speed=2,
            cell_size=CELL_SIZE,
            grid_size=GRID_SIZE
        )

    def prepare_grid(self):
        """
        Convert obstacle list to a grid representation.
        """
        grid_size = GRID_SIZE // CELL_SIZE
        grid = np.zeros((grid_size, grid_size), dtype=int)

        for obs in self.obstacles:
            x, y, size = obs['x'], obs['y'], obs['size']
            start_i = y // CELL_SIZE
            start_j = x // CELL_SIZE
            end_i = min((y + size) // CELL_SIZE, grid_size - 1)
            end_j = min((x + size) // CELL_SIZE, grid_size - 1)
            grid[start_i:end_i + 1, start_j:end_j + 1] = 255  # Mark as obstacle
        return grid

    def create_dynamic_obstacles(self):
        """
        Select a subset of obstacles to be dynamic by assigning directions.
        """
        if not self.obstacles:
            return []

        num_dynamic = min(5, len(self.obstacles))  # Limit number of dynamic obstacles
        dynamic_obstacles = []
        selected = random.sample(self.obstacles, num_dynamic)

        for obs in selected:
            x, y, size = obs['x'], obs['y'], obs['size']
            direction = [random.choice([-1, 1]), random.choice([-1, 1])]
            dynamic_obstacles.append({'x': x, 'y': y, 'size': size, 'dir': direction})
        return dynamic_obstacles

    def display_results(self, results, sim_type):
        """
        Display benchmarking results in a new window.
        """
        result_window = tk.Toplevel(self.master)
        result_window.title(f"{sim_type.capitalize()} Simulation Results")

        tk.Label(result_window, text=f"{sim_type.capitalize()} Simulation Benchmark Results",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        for algo, metrics in results.items():
            frame = tk.Frame(result_window)
            frame.pack(pady=5, padx=10, fill='x')

            tk.Label(frame, text=f"Algorithm: {algo}", font=("Helvetica", 12, "underline")).pack(anchor='w')
            tk.Label(frame, text=f" - Average Computation Time: {metrics['computation_time']:.4f} seconds").pack(
                anchor='w')
            tk.Label(frame, text=f" - Average Path Length: {metrics['path_length']:.2f} pixels").pack(anchor='w')
            tk.Label(frame, text=f" - Average Accuracy: {metrics['accuracy']:.2f}%").pack(anchor='w')
            tk.Label(frame, text=f" - Collision Rate: {metrics['collision_rate']:.2f}%").pack(anchor='w')

        # Optionally, save results to a file
        save_button = tk.Button(result_window, text="Save Results",
                                command=lambda: self.save_results(results, sim_type))
        save_button.pack(pady=10)

    def save_results(self, results, sim_type):
        """
        Save benchmarking results to a text file.
        """
        timestamp = int(time.time())
        filename = f"{sim_type}_benchmark_{timestamp}.txt"
        if not os.path.exists('results'):
            os.makedirs('results')
        filepath = os.path.join('results', filename)

        with open(filepath, 'w') as f:
            f.write(f"{sim_type.capitalize()} Simulation Benchmark Results\n")
            f.write(f"Timestamp: {time.ctime(timestamp)}\n\n")
            for algo, metrics in results.items():
                f.write(f"Algorithm: {algo}\n")
                f.write(f" - Average Computation Time: {metrics['computation_time']:.4f} seconds\n")
                f.write(f" - Average Path Length: {metrics['path_length']:.2f} pixels\n")
                f.write(f" - Average Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f" - Collision Rate: {metrics['collision_rate']:.2f}%\n\n")

        messagebox.showinfo("Results Saved", f"Benchmark results saved to {filepath}")

    def update_progress(self, algo_names, current_trial, total_trials):
        """
        Update the progress bars for each algorithm.

        :param algo_names: List of algorithm names.
        :param current_trial: Current trial number.
        :param total_trials: Total number of trials.
        """
        for algo_name in algo_names:
            if algo_name not in self.progress_bars:
                # Create a new progress bar for the algorithm
                label = tk.Label(self.progress_frame, text=algo_name)
                label.pack(anchor='w')
                progress = ttk.Progressbar(self.progress_frame, orient='horizontal', length=300, mode='determinate')
                progress.pack(pady=2)
                progress['maximum'] = total_trials
                self.progress_bars[algo_name] = progress

            # Update progress
            self.progress_bars[algo_name]['value'] = current_trial

    def get_selected_algorithms(self):
        """
        Retrieve the list of selected algorithms from the listbox.

        :return: List of algorithm classes.
        """
        selected_indices = self.algorithms_listbox.curselection()
        selected_algos = []
        algo_map = {
            "AStar": AStar,
            "RRTStar": RRTStar,
            "DStarLite": DStarLite,
            "OkayPlan": OkayPlan
        }
        for idx in selected_indices:
            algo_name = self.algorithms_var.get()[idx]
            algo_class = algo_map.get(algo_name)
            if algo_class:
                selected_algos.append(algo_class)
        return selected_algos

    def select_all(self, select=True):
        """
        Select or deselect all items in the algorithms listbox.

        :param select: Boolean indicating whether to select or deselect.
        """
        if select:
            self.algorithms_listbox.select_set(0, tk.END)
        else:
            self.algorithms_listbox.select_clear(0, tk.END)


def simulate_and_record(sim_type, grid, algorithm, dynamic_obstacles, results):
    """
    Simulates and records results for a single algorithm.

    :param sim_type: 'static' or 'dynamic'
    :param grid: 2D numpy array representing the environment.
    :param algorithm: Path planning algorithm class.
    :param dynamic_obstacles: List of dynamic obstacle dictionaries.
    :param results: Shared dictionary to store results.
    """
    algo_name = algorithm.__name__
    logging.info(f"Starting {algo_name} for {sim_type} simulation.")

    try:
        if sim_type == 'static':
            if algo_name == "OkayPlan":
                # Define parameters for OkayPlan
                params = [1.0] * 19  # Example: Replace with actual parameters as needed
                planner = algorithm(grid=grid, start=START_POS, goal=GOAL_POS, params=params)
                path, collision = planner.plan(env_info={
                    'start_point': START_POS,
                    'target_point': GOAL_POS,
                    'd2target': heuristic(START_POS, GOAL_POS),
                    'Obs_Segments': [],  # Populate if necessary
                    'Flat_pdct_segments': []  # Populate if necessary
                })
            else:
                sim = StaticSimulation(
                    grid=grid,
                    start=START_POS,
                    goal=GOAL_POS,
                    algorithm=algorithm
                )
                path, collision = sim.run()
        elif sim_type == 'dynamic':
            if algo_name == "OkayPlan":
                # Define parameters for OkayPlan
                params = [1.0] * 19  # Example: Replace with actual parameters as needed
                planner = algorithm(grid=grid, start=START_POS, goal=GOAL_POS, params=params)
                # Update 'Flat_pdct_segments' based on dynamic obstacles if needed
                path, collision = planner.plan(env_info={
                    'start_point': START_POS,
                    'target_point': GOAL_POS,
                    'd2target': heuristic(START_POS, GOAL_POS),
                    'Obs_Segments': [],  # Populate if necessary
                    'Flat_pdct_segments': []  # Populate based on dynamic obstacles
                })
            else:
                sim = DynamicSimulation(
                    grid=grid,
                    start=START_POS,
                    goal=GOAL_POS,
                    algorithm=algorithm,
                    dynamic_obstacles=dynamic_obstacles,
                    obstacle_speed=2
                )
                path, collision = sim.run()
        else:
            raise ValueError("sim_type must be 'static' or 'dynamic'")

        # Compute metrics
        computation_time = 0  # Placeholder, as we didn't track time within this function
        path_length = calculate_path_length(path) if path else float('inf')
        accuracy = calculate_accuracy(path, grid) if path else 0

        results[algo_name] = {
            'computation_time': computation_time,
            'path_length': path_length,
            'accuracy': accuracy,
            'collision': collision
        }
        logging.info(f"Completed {algo_name} for {sim_type} simulation.")
    except Exception as e:
        logging.error(f"Algorithm {algo_name} encountered an error: {e}")
        results[algo_name] = {
            'computation_time': None,
            'path_length': None,
            'accuracy': None,
            'collision': None
        }


def run_benchmark_parallel(sim_type, grid, algorithms, dynamic_obstacles=None):
    """
    Runs simulations for multiple algorithms in parallel.

    :param sim_type: 'static' or 'dynamic'
    :param grid: 2D numpy array representing the environment.
    :param algorithms: List of path planning algorithm classes.
    :param dynamic_obstacles: List of dynamic obstacle dictionaries (for dynamic simulations).
    :return: Dictionary containing results for each algorithm.
    """
    manager = Manager()
    results = manager.dict()
    processes = []

    for algo in algorithms:
        p = Process(target=simulate_and_record, args=(sim_type, grid, algo, dynamic_obstacles, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return dict(results)


def calculate_path_length(path):
    """
    Calculate the total length of the path.

    :param path: List of (x, y) tuples representing the path.
    :return: Total path length in pixels.
    """
    if not path or len(path) < 2:
        return 0
    length = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        length += (dx ** 2 + dy ** 2) ** 0.5
    return length


def calculate_accuracy(path, grid):
    """
    Calculate accuracy based on path adherence to the grid.
    Defines accuracy as the percentage of path points that are free.

    :param path: List of (x, y) tuples representing the path.
    :param grid: 2D numpy array representing the environment.
    :return: Accuracy percentage.
    """
    if not path:
        return 0
    free_points = 0
    for point in path:
        i = int(point[1] // CELL_SIZE)
        j = int(point[0] // CELL_SIZE)
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if grid[i][j] == 0:
                free_points += 1
    return (free_points / len(path)) * 100


def main():
    root = tk.Tk()
    app = PathPlanningBenchmarkingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
