import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import math
import os
import logging
import csv
from sklearn.neighbors import NearestNeighbors
import psutil
import threading
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk, ImageDraw  # Import Pillow modules

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MapVisualization:
    def __init__(self, root, data_file='depth_data.json', update_interval=200):
        """
        Initializes the MapVisualization class.

        Parameters:
        - root: Tkinter root window.
        - data_file: Path to the JSON file containing depth data.
        - update_interval: Time between updates in milliseconds.
        """
        self.root = root
        self.root.title("SLAM Map Visualization with Statistical Analysis")

        self.data_file = data_file
        self.update_interval = update_interval  # in milliseconds

        # Canvas dimensions
        self.canvas_width = 1200
        self.canvas_height = 900

        # Create Canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Initialize map parameters
        self.scale = 20  # Initial pixels per unit; will be updated dynamically
        self.origin_x = self.canvas_width / 2
        self.origin_y = self.canvas_height / 2

        # Initialize robot representation
        self.robot_radius = 10  # pixels
        self.robot = self.canvas.create_oval(
            self.origin_x - self.robot_radius,
            self.origin_y - self.robot_radius,
            self.origin_x + self.robot_radius,
            self.origin_y + self.robot_radius,
            fill='blue', outline='black'
        )

        # Orientation indicator
        self.orientation_line_length = 40  # pixels
        self.orientation_line = self.canvas.create_line(
            self.origin_x, self.origin_y,
            self.origin_x, self.origin_y - self.orientation_line_length,
            fill='red', width=3
        )

        # Timestamp label
        self.timestamp_label = tk.Label(root, text="Timestamp: N/A", font=("Arial", 12))
        self.timestamp_label.pack(pady=5)

        # Initialize data storage
        self.ground_truth_poses = []
        self.estimated_poses = []
        self.ground_truth_map = []
        self.slam_maps = []
        self.cpu_usage = []
        self.memory_usage = []

        # Initialize robot path
        self.robot_path = []  # Initialize robot_path

        # Initialize obstacle points with timestamp
        self.obstacle_points = set()  # Set of tuples: (ix, iy)

        # Initialize PIL Image for obstacles
        self.obstacle_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.obstacle_draw = ImageDraw.Draw(self.obstacle_image)
        self.photo_image = ImageTk.PhotoImage(self.obstacle_image)
        self.obstacle_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Define obstacle color (single color since there's only one algorithm)
        self.obstacle_color = (0, 0, 255)  # Blue

        # Define laser parameters
        self.MAX_LASER_RANGE = 10.0  # Maximum laser range units
        self.LASER_TOLERANCE = 0.1  # Tolerance for point removal

        # Start CPU and Memory monitoring in separate thread
        self.monitoring = True
        self.pid = os.getpid()  # Replace with SLAM algorithm's PID if different
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.start()

        # Create Buttons for Statistical Analysis
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.save_button = tk.Button(self.button_frame, text="Save Metrics", command=self.save_metrics)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = tk.Button(self.button_frame, text="Analyze Metrics", command=self.analyze_metrics)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        # Define maximum number of obstacle points to store
        self.MAX_OBSTACLE_POINTS = 5000  # Reduced to improve performance

        # Initialize bounds for dynamic scaling
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        # Define scale limits
        self.MAX_SCALE = 50  # Maximum pixels per unit
        self.MIN_SCALE = 5  # Minimum pixels per unit

        # Flag to indicate if scale has changed
        self.scale_changed = False

        # Start the update loop
        self.update_map()

    def world_to_canvas(self, x, y):
        """
        Transforms world coordinates to canvas coordinates.

        Parameters:
        - x: X-coordinate in world units.
        - y: Y-coordinate in world units.

        Returns:
        - (canvas_x, canvas_y): Tuple of canvas coordinates.
        """
        canvas_x = self.origin_x + x * self.scale
        canvas_y = self.origin_y - y * self.scale
        return (canvas_x, canvas_y)

    def calculate_pose_error(self):
        """
        Calculates the Absolute Pose Error (APE) between ground truth and estimated poses.

        Returns:
        - ape: Mean Absolute Pose Error.
        """
        if len(self.ground_truth_poses) != len(self.estimated_poses):
            logging.warning("Ground truth and estimated poses count mismatch.")
            return None
        gt = np.array(self.ground_truth_poses)
        est = np.array(self.estimated_poses)
        errors = np.linalg.norm(gt - est, axis=1)
        ape = np.mean(errors)
        return ape

    def calculate_map_accuracy(self):
        """
        Calculates the Map Accuracy using the k-nearest neighbor method.

        Returns:
        - mean_accuracy: Mean map accuracy in centimeters.
        """
        if not self.ground_truth_map or not self.slam_maps:
            logging.warning("Map data is incomplete for accuracy calculation.")
            return None
        gt_map = np.array(self.ground_truth_map)
        slam_map = np.array(self.slam_maps)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(slam_map)
        distances, _ = nbrs.kneighbors(gt_map)
        mean_accuracy = np.mean(distances) * 100  # Convert to centimeters if units are meters
        return mean_accuracy

    def monitor_resources(self):
        """
        Monitors CPU and memory usage of a process.
        """
        process = psutil.Process(self.pid)
        while self.monitoring:
            try:
                cpu = process.cpu_percent(interval=None)  # Non-blocking
                mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                self.cpu_usage.append(cpu)
                self.memory_usage.append(mem)
                time.sleep(self.update_interval / 1000.0)
            except psutil.NoSuchProcess:
                logging.error(f"Process with PID {self.pid} does not exist.")
                break

    def point_on_ray(self, robot_x, robot_y, ray_angle, point_x, point_y):
        """
        Determines if a point lies on a given ray within a tolerance.

        Parameters:
        - robot_x, robot_y: Robot's position.
        - ray_angle: Angle of the ray in degrees.
        - point_x, point_y: Coordinates of the point.

        Returns:
        - True if the point lies on the ray within tolerance, False otherwise.
        """
        # Convert angle to radians
        theta = math.radians(ray_angle)
        # Ray direction vector
        dx = math.cos(theta)
        dy = math.sin(theta)
        # Vector from robot to point
        px = point_x - robot_x
        py = point_y - robot_y
        # Project point onto ray
        dot = px * dx + py * dy
        if dot < 0 or dot > self.MAX_LASER_RANGE:
            return False
        # Compute perpendicular distance from point to ray
        perp_dist = abs(-dy * px + dx * py)
        return perp_dist <= self.LASER_TOLERANCE

    def update_map(self):
        """
        Reads the data file and updates the map accordingly.
        """
        if not os.path.exists(self.data_file):
            logging.error(f"Data file {self.data_file} does not exist.")
            self.root.after(self.update_interval, self.update_map)
            return

        try:
            with open(self.data_file, 'r') as f:
                depth_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            self.root.after(self.update_interval, self.update_map)
            return
        except Exception as e:
            logging.error(f"Error reading data file: {e}")
            self.root.after(self.update_interval, self.update_map)
            return

        # Parse robot information
        robot_position = depth_data.get("robot_position", {})
        robot_orientation = depth_data.get("robot_orientation", 0)
        timestamp = depth_data.get("timestamp", "N/A")
        depth_estimates = depth_data.get("depth_estimates", [])

        # Update robot position
        if isinstance(robot_position, dict):
            x = robot_position.get('x', 0)
            y = robot_position.get('y', 0)
        elif isinstance(robot_position, list) and len(robot_position) >= 2:
            x, y = robot_position[0], robot_position[1]
        else:
            x, y = 0, 0
            logging.warning("robot_position has an unexpected format.")

        # Store ground truth and estimated poses
        self.ground_truth_poses.append([x, y])
        self.estimated_poses.append([x, y])  # Replace with actual estimated pose from SLAM

        current_time = time.time()

        # Collect new obstacle points
        new_obstacles = []
        for estimate in depth_estimates:
            rays = estimate.get("rays", [])
            for ray in rays:
                intersection = ray.get("intersection", [])
                distance = ray.get("distance", self.MAX_LASER_RANGE)
                ray_angle = ray.get("ray_angle", 0)

                if len(intersection) < 2 or distance >= self.MAX_LASER_RANGE:
                    # Ray did not detect an obstacle within range
                    # Check and remove any existing obstacle points along this ray
                    for point in list(self.obstacle_points):
                        ix, iy = point
                        if self.point_on_ray(x, y, ray_angle, ix, iy):
                            self.obstacle_points.remove(point)
                            logging.info(
                                f"Removed obstacle at ({ix}, {iy}) due to no detection on ray angle {ray_angle}")
                else:
                    # Ray detected an obstacle
                    ix, iy = intersection[0], intersection[1]
                    self.obstacle_points.add((ix, iy))
                    logging.info(f"Added/Updated obstacle at ({ix}, {iy}) detected by ray angle {ray_angle}")

                    # Update bounds based on new obstacle
                    self.update_bounds(ix, iy)

        # Trim the obstacle points set if it exceeds the maximum
        if len(self.obstacle_points) > self.MAX_OBSTACLE_POINTS:
            # Remove the oldest points (not tracked individually, so remove randomly)
            excess = len(self.obstacle_points) - self.MAX_OBSTACLE_POINTS
            for _ in range(excess):
                removed_point = self.obstacle_points.pop()
                logging.info(f"Trimmed obstacle at {removed_point} to maintain MAX_OBSTACLE_POINTS")

        # Compute dynamic scaling once after processing all new obstacles
        self.compute_dynamic_scale()

        # Redraw all obstacles
        self.redraw_obstacles()

        # Move robot on the canvas
        canvas_x, canvas_y = self.world_to_canvas(x, y)
        self.canvas.coords(
            self.robot,
            canvas_x - self.robot_radius,
            canvas_y - self.robot_radius,
            canvas_x + self.robot_radius,
            canvas_y + self.robot_radius
        )

        # Update orientation line based on actual orientation
        orientation_rad = math.radians(robot_orientation)
        end_x = canvas_x + self.orientation_line_length * math.cos(orientation_rad)
        end_y = canvas_y - self.orientation_line_length * math.sin(orientation_rad)
        self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)

        # Store robot path - keep path lines larger for visibility
        self.robot_path.append((canvas_x, canvas_y))
        if len(self.robot_path) > 1:
            self.canvas.create_line(
                self.robot_path[-2][0], self.robot_path[-2][1],
                self.robot_path[-1][0], self.robot_path[-1][1],
                fill='blue', width=2  # increased width
            )

        # Update timestamp label
        if isinstance(timestamp, (int, float)):
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        else:
            formatted_time = str(timestamp)
            logging.warning("Timestamp is not a Unix timestamp.")
        self.timestamp_label.config(text=f"Timestamp: {formatted_time}")

        # Schedule next update
        self.root.after(self.update_interval, self.update_map)

    def update_bounds(self, x, y):
        """
        Updates the minimum and maximum bounds based on new x and y values.

        Parameters:
        - x: X-coordinate.
        - y: Y-coordinate.
        """
        if self.min_x is None or x < self.min_x:
            self.min_x = x
        if self.max_x is None or x > self.max_x:
            self.max_x = x
        if self.min_y is None or y < self.min_y:
            self.min_y = y
        if self.max_y is None or y > self.max_y:
            self.max_y = y

    def compute_dynamic_scale(self):
        """
        Computes the scale factor dynamically based on the current bounds to fit the map within the canvas.
        Limits the scale to predefined maximum and minimum values.
        """
        if self.min_x is None or self.max_x is None or self.min_y is None or self.max_y is None:
            return  # Cannot compute scale without bounds

        # Calculate the range of the map
        x_range = self.max_x - self.min_x
        y_range = self.max_y - self.min_y

        if x_range == 0 or y_range == 0:
            return  # Avoid division by zero

        # Define buffer (margin) in pixels
        buffer = 100  # pixels

        # Calculate scale to fit the map within the canvas with buffer
        scale_x = (self.canvas_width - 2 * buffer) / x_range
        scale_y = (self.canvas_height - 2 * buffer) / y_range

        # Choose the smaller scale to maintain aspect ratio
        new_scale = min(scale_x, scale_y)

        # Apply scale limits
        new_scale = min(new_scale, self.MAX_SCALE)
        new_scale = max(new_scale, self.MIN_SCALE)

        # Check if scale has changed
        if new_scale != self.scale:
            self.scale = new_scale
            self.scale_changed = True

            # Update origin to center the map
            self.origin_x = (self.canvas_width) / 2 - (self.min_x + self.max_x) / 2 * self.scale
            self.origin_y = (self.canvas_height) / 2 + (self.max_y + self.min_y) / 2 * self.scale

            logging.info(f"Dynamic Scale Updated: {self.scale:.2f} pixels/unit")
            logging.info(f"Origin Updated to: ({self.origin_x:.2f}, {self.origin_y:.2f})")

    def redraw_obstacles(self):
        """
        Redraws all obstacle points on the PIL image based on the updated scale and origin.
        """
        # Clear the obstacle image
        self.obstacle_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.obstacle_draw = ImageDraw.Draw(self.obstacle_image)

        # Redraw all obstacle points
        for ix, iy in self.obstacle_points:
            canvas_x, canvas_y = self.world_to_canvas(ix, iy)
            self.obstacle_draw.ellipse(
                [
                    (canvas_x - 2, canvas_y - 2),
                    (canvas_x + 2, canvas_y + 2)
                ],
                fill=self.obstacle_color,
                outline=self.obstacle_color
            )

        # Update the PhotoImage with the redrawn obstacle data
        self.photo_image = ImageTk.PhotoImage(self.obstacle_image)
        self.canvas.itemconfig(self.obstacle_canvas, image=self.photo_image)

    def save_metrics(self, filename='slam_metrics.csv'):
        """
        Saves the collected metrics to a CSV file for statistical analysis.
        """
        data_length = min(len(self.ground_truth_poses), len(self.estimated_poses),
                          len(self.cpu_usage), len(self.memory_usage))
        if data_length == 0:
            logging.warning("No data to save.")
            messagebox.showwarning("Save Metrics", "No data available to save.")
            return

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Pose_Error', 'Map_Accuracy', 'CPU_Usage', 'Memory_Usage'])
            for i in range(data_length):
                pose_error = self.calculate_pose_error()
                map_accuracy = self.calculate_map_accuracy()
                cpu = self.cpu_usage[i]
                mem = self.memory_usage[i]
                writer.writerow([i, pose_error, map_accuracy, cpu, mem])
        logging.info(f"Metrics saved to {filename}.")
        messagebox.showinfo("Save Metrics", f"Metrics successfully saved to {filename}.")

    def analyze_metrics(self):
        """
        Performs statistical analysis on the collected metrics.
        """
        filename = 'slam_metrics.csv'
        if not os.path.exists(filename):
            logging.error(f"Metrics file {filename} does not exist.")
            messagebox.showerror("Error", f"Metrics file {filename} does not exist.")
            return

        # Load metrics
        df = pd.read_csv(filename)
        print(df.head())

        # Drop rows with missing data
        df.dropna(inplace=True)

        # Since there's only one algorithm, we don't need to categorize by algorithm.
        # We'll analyze the metrics directly.

        # Plot histograms for each metric
        metrics = ['Pose_Error', 'Map_Accuracy', 'CPU_Usage', 'Memory_Usage']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[metric], kde=True, bins=30)
            plt.title(f'{metric} Distribution')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.show()

        # Example: Correlation Analysis between CPU Usage and Pose Error
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='CPU_Usage', y='Pose_Error', data=df)
        plt.title('CPU Usage vs. Pose Error')
        plt.xlabel('CPU Usage (%)')
        plt.ylabel('Pose Error')
        plt.show()

        correlation = df['CPU_Usage'].corr(df['Pose_Error'])
        logging.info(f"Correlation between CPU Usage and Pose Error: {correlation:.2f}")

        # Example: Additional statistical analyses can be added here

    def two_sample_t_test(self, sample1, sample2, alpha=0.05):
        """
        Performs a Two-Sample T-Test.

        Parameters:
        - sample1: First sample data (list or array).
        - sample2: Second sample data (list or array).
        - alpha: Significance level (default 0.05).

        Returns:
        - t_stat: T-statistic.
        - p_val: P-value.
        - conclusion: String stating whether to reject H0.
        """
        t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
        if p_val < alpha:
            conclusion = "Reject the null hypothesis. Significant difference exists."
        else:
            conclusion = "Fail to reject the null hypothesis. No significant difference."
        return t_stat, p_val, conclusion

    def on_closing(self):
        """
        Handles the window closing event.
        """
        self.monitoring = False
        self.monitor_thread.join()
        self.save_metrics()
        self.root.destroy()


def plackett_burman_experiment(num_factors):
    """
    Generates a Plackett–Burman design matrix.

    Parameters:
    - num_factors: Number of factors (SLAM parameters) to test.

    Returns:
    - design: DataFrame containing the design matrix with factors coded as -1 and 1.
    """
    from pyDOE import pbdesign
    design_matrix = pbdesign(num_factors)
    factors = [f'Param_{i + 1}' for i in range(num_factors)]
    design = pd.DataFrame(design_matrix, columns=factors)
    return design


def analyze_plackett_burman(design, results):
    """
    Analyzes the Plackett–Burman experiment results.

    Parameters:
    - design: DataFrame containing the design matrix.
    - results: Series containing the metric results corresponding to each experiment.

    Returns:
    - significant_params: List of parameters that have a significant effect.
    """
    import statsmodels.api as sm
    X = sm.add_constant(design)
    model = sm.OLS(results, X).fit()
    print(model.summary())

    # Extract p-values
    p_values = model.pvalues[1:]  # Exclude intercept
    significant_params = p_values[p_values < 0.05].index.tolist()
    print(f"Significant parameters: {significant_params}")
    return significant_params


def plot_effects(design, results, significant_params):
    """
    Plots the main effects of significant parameters.

    Parameters:
    - design: DataFrame containing the design matrix.
    - results: Series containing the metric results.
    - significant_params: List of significant parameters.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    for param in significant_params:
        plt.figure(figsize=(8, 5))
        sns.pointplot(x=param, y='Metric', data=pd.concat([design, results.rename('Metric')], axis=1))
        plt.title(f'Main Effect of {param}')
        plt.xlabel(param)
        plt.ylabel('Metric Result')
        plt.show()


def full_factorial_experiment(num_factors, levels=2):
    """
    Generates a Full Factorial design matrix.

    Parameters:
    - num_factors: Number of factors (SLAM parameters) to test.
    - levels: Number of levels per factor (default is 2).

    Returns:
    - design: DataFrame containing the design matrix with factors coded from 0 to levels-1.
    """
    from pyDOE import fullfact
    design_matrix = fullfact([levels] * num_factors)
    factors = [f'Param_{i + 1}' for i in range(num_factors)]
    design = pd.DataFrame(design_matrix, columns=factors)
    return design


def analyze_full_factorial(design, results):
    """
    Analyzes the Full Factorial experiment results.

    Parameters:
    - design: DataFrame containing the design matrix.
    - results: Series containing the metric results corresponding to each experiment.

    Returns:
    - optimal_params: Dictionary of parameters with their optimal levels.
    """
    import statsmodels.api as sm
    X = sm.add_constant(design)
    model = sm.OLS(results, X).fit()
    print(model.summary())

    # Determine optimal levels based on coefficients
    coefficients = model.params[1:]
    optimal_params = {}
    for param in design.columns:
        optimal_params[param] = 'High' if coefficients[param] > 0 else 'Low'
    print(f"Optimal Parameters: {optimal_params}")
    return optimal_params


def plot_full_factorial(design, results, optimal_params):
    """
    Plots the interaction effects of parameters.

    Parameters:
    - design: DataFrame containing the design matrix.
    - results: Series containing the metric results.
    - optimal_params: Dictionary of optimal parameters.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.pairplot(design)
    plt.suptitle('Full Factorial Design Interactions', y=1.02)
    plt.show()


def main():
    root = tk.Tk()
    app = MapVisualization(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
