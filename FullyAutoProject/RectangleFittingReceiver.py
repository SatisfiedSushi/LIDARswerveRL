import logging  # Added import
import tkinter as tk
import struct
import math
from multiprocessing import shared_memory
import sys
import numpy as np
from sklearn.cluster import DBSCAN
import warnings

from LShapeFitting import LShapeFitting

warnings.filterwarnings("ignore", category=UserWarning)


class TrajectoryPoint:
    def __init__(self, x: float, y: float, theta: float, v: float, omega: float, time: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega
        self.time = time


class RobotPositionVisualization:
    def __init__(
        self,
        parent,
        shm_name,
        shm_size,
        title,
        destination_queue,
        obstacle_queue,
        path_queue,
        termination_event,
        detected_obstacle_queue,
        lock=None,
        canvas_width=800,
        canvas_height=600,
        world_width=16.46,
        world_height=8.23
    ):
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.world_width = world_width
        self.world_height = world_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.scale_x = canvas_width / world_width
        self.scale_y = canvas_height / world_height
        self.lock = lock
        self.destination_queue = destination_queue
        self.obstacle_queue = obstacle_queue
        self.path_queue = path_queue
        self.detected_obstacle_queue = detected_obstacle_queue
        self.termination_event = termination_event

        # Define color constants
        self.COLOR_STATIC_OBSTACLE = 'green'
        self.COLOR_DYNAMIC_OBSTACLE = 'red'
        self.COLOR_STATIC_BBOX = 'orange'  # or use a hex code like '#FFD580'
        self.COLOR_DYNAMIC_BBOX = 'red'
        self.COLOR_ROBOT = 'blue'
        self.COLOR_ORIENTATION = 'red'
        self.COLOR_PATH_DSTAR = 'green'
        self.COLOR_PATH_RAMSETE = 'magenta'

        # Calculate scaling factor to maintain aspect ratio
        self.scale = min(self.scale_x, self.scale_y)
        self.padding_x = (self.canvas_width - (self.world_width * self.scale)) / 2
        self.padding_y = (self.canvas_height - (self.world_height * self.scale)) / 2

        # Logger setup
        self.logger = logging.getLogger(title)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Tkinter canvas setup
        self.canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height, bg='white')
        self.canvas.pack()

        # Bind the click event to the canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Robot representation
        self.robot_radius = 10  # Adjusted for better visibility
        self.robot = self.canvas.create_oval(0, 0, 0, 0, fill=self.COLOR_ROBOT)
        self.orientation_line = self.canvas.create_line(0, 0, 0, 0, fill=self.COLOR_ORIENTATION, width=2)

        # Obstacles (walls and additional obstacles)
        self.obstacles = {
            'left_wall': (0, world_height / 2, 0.1, world_height),
            'right_wall': (world_width, world_height / 2, 0.1, world_height),
            'bottom_wall': (world_width / 2, 0, world_width, 0.1),
            'top_wall': (world_width / 2, world_height, world_width, 0.1),
            'obstacle1': (4, 3, 0.5, 0.5),
            'obstacle2': (6, 2, 1, 1),
        }

        # Lidar points
        self.lidar_point_radius = 2
        self.lidar_points = []

        # Detected robots
        self.detected_robot_rectangles = []

        # Inflated Obstacle Bounding Boxes
        self.static_bboxes = []
        self.dynamic_bboxes = []

        # Path visualization
        self.dstar_path_items = []
        self.ramsete_path_items = []

        # Position and orientation display
        self.position_label = tk.Label(parent, text="Position: (x, y), Orientation: θ°")
        self.position_label.pack()

        # Initialize LShapeFitting instance
        self.l_shape_fitter = LShapeFitting()

        # Connect to shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.logger.info(f"Connected to shared memory: {self.shm_name}, size: {self.shm.size} bytes")

            if self.shm.size != self.shm_size:
                self.logger.warning(
                    f"Shared memory size mismatch: Expected {self.shm_size} bytes, got {self.shm.size} bytes")
        except FileNotFoundError:
            self.logger.error(f"Shared memory '{self.shm_name}' not found.")
            self.shm = None
        except Exception as e:
            self.logger.error(f"Failed to connect to shared memory: {e}")
            self.shm = None

        # Draw obstacles (walls and additional obstacles)
        self.draw_obstacles()

        # Start update loop
        self.update_visualization()

    def on_canvas_click(self, event):
        """
        Event handler for mouse click on the canvas.
        """
        # Get the canvas coordinates
        canvas_x = event.x
        canvas_y = event.y

        # Convert to world coordinates
        world_x, world_y = self.canvas_to_world(canvas_x, canvas_y)

        # Log the click event
        self.logger.info(f"Clicked at canvas ({canvas_x}, {canvas_y}), world ({world_x:.2f}, {world_y:.2f})")

        # Send the destination point to the main process
        self.destination_queue.put((world_x, world_y))

        # Optionally, you can draw a marker at the clicked location
        self.canvas.create_oval(
            canvas_x - 5, canvas_y - 5,
            canvas_x + 5, canvas_y + 5,
            outline='blue', fill='', width=2, dash=(3, 5)
        )

    def canvas_to_world(self, canvas_x, canvas_y):
        """
        Convert canvas coordinates to world coordinates.
        """
        world_x = (canvas_x - self.padding_x) / self.scale
        world_y = (self.canvas_height - canvas_y - self.padding_y) / self.scale
        return world_x, world_y

    def world_to_canvas(self, x, y):
        """
        Convert world coordinates to canvas coordinates.
        """
        canvas_x = (x * self.scale) + self.padding_x
        canvas_y = self.canvas_height - ((y * self.scale) + self.padding_y)
        return canvas_x, canvas_y

    def draw_obstacles(self):
        """
        Draw walls and additional obstacles on the canvas.
        """
        for obstacle_name, (obs_x, obs_y, size_x, size_y) in self.obstacles.items():
            left = obs_x - size_x / 2
            right = obs_x + size_x / 2
            bottom = obs_y - size_y / 2
            top = obs_y + size_y / 2
            x1, y1 = self.world_to_canvas(left, bottom)
            x2, y2 = self.world_to_canvas(right, top)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='green', fill='green', width=2)

    def is_point_in_obstacle(self, x, y):
        threshold = 0.1
        if (
            abs(x - 0) <= threshold or
            abs(x - self.world_width) <= threshold or
            abs(y - 0) <= threshold or
            abs(y - self.world_height) <= threshold
        ):
            return True

        for obstacle_name, (obs_x, obs_y, size_x, size_y) in self.obstacles.items():
            left = obs_x - size_x / 2
            right = obs_x + size_x / 2
            top = obs_y + size_y / 2
            bottom = obs_y - size_y / 2
            if left <= x <= right and bottom <= y <= top:
                return True

        return False

    def fit_and_draw_rectangle(self, cluster_points):
        if len(cluster_points) < 5:
            return

        # Extract x and y coordinates from the cluster points
        ox = [point[0] for point in cluster_points]
        oy = [point[1] for point in cluster_points]

        # Use the LShapeFitting class to fit rectangles
        rects, _ = self.l_shape_fitter.fitting(ox, oy)

        for rect in rects:
            rect.calc_rect_contour()
            canvas_coords = []
            for wx, wy in zip(rect.rect_c_x, rect.rect_c_y):
                cx, cy = self.world_to_canvas(wx, wy)
                canvas_coords.append((cx, cy))

            canvas_coords_flat = [coord for point in canvas_coords for coord in point]

            rect_item = self.canvas.create_polygon(
                canvas_coords_flat,
                outline='orange',
                fill='',
                width=2
            )
            self.detected_robot_rectangles.append(rect_item)

    def perform_lshape_fitting(self, lidar_points):
        """
        Perform L-shape fitting on the given lidar points.
        """
        try:
            # Convert to numpy array
            lidar_points_array = np.array(lidar_points)

            # Segment the lidar points into clusters using DBSCAN
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            labels = dbscan.fit_predict(lidar_points_array)

            unique_labels = set(labels)
            detected_obstacle_positions = []
            for label in unique_labels:
                if label == -1:
                    continue  # Noise
                cluster_points = lidar_points_array[labels == label]

                if len(cluster_points) < 5:
                    continue  # Skip small clusters

                # Perform L-shape fitting on the cluster
                rects, _ = self.l_shape_fitter.fitting(cluster_points[:, 0], cluster_points[:, 1])

                if rects is not None:
                    # Draw the rectangle representing the detected robot
                    self.fit_and_draw_rectangle(cluster_points)

                    # Add rectangle center to detected obstacles
                    rect_center = np.mean(cluster_points, axis=0)
                    detected_obstacle_positions.append((rect_center[0], rect_center[1]))

            # Send detected obstacles to the main process via the separate queue
            if detected_obstacle_positions:
                self.logger.debug(f"Sending detected obstacles: {detected_obstacle_positions}")
                self.detected_obstacle_queue.put(detected_obstacle_positions)
        except Exception as e:
            self.logger.error(f"Exception in perform_lshape_fitting: {e}")

    def draw_circle(self, x, y):
        cx, cy = self.world_to_canvas(x, y)
        radius = 0.28 * self.scale  # Convert 0.56 diameter to canvas units
        circle = self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline='blue', fill='blue'
        )
        self.detected_robot_rectangles.append(circle)

    def draw_path(self, path_type, path):
        # Remove old path based on type
        if path_type == 'dstar':
            for item in self.dstar_path_items:
                self.canvas.delete(item)
            self.dstar_path_items = []
        elif path_type == 'ramsete':
            for item in self.ramsete_path_items:
                self.canvas.delete(item)
            self.ramsete_path_items = []

        # Draw new path
        for i in range(len(path) - 1):
            x1, y1, _ = path[i]
            x2, y2, _ = path[i + 1]
            canvas_x1, canvas_y1 = self.world_to_canvas(x1, y1)
            canvas_x2, canvas_y2 = self.world_to_canvas(x2, y2)
            if path_type == 'dstar':
                line = self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, fill=self.COLOR_PATH_DSTAR, width=2, dash=(4, 2))
                self.dstar_path_items.append(line)
            elif path_type == 'ramsete':
                line = self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, fill=self.COLOR_PATH_RAMSETE, width=2)
                self.ramsete_path_items.append(line)

    def update_visualization(self):
        try:
            if self.shm:
                buffer = self.shm.buf[:self.shm_size]
                offset = 0

                # Read robot's position
                robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
                offset += struct.calcsize('ff')

                # Read robot's orientation
                robot_theta_rad, = struct.unpack_from('f', buffer, offset)
                offset += struct.calcsize('f')

                # Read timestamp
                timestamp, = struct.unpack_from('f', buffer, offset)
                offset += struct.calcsize('f')

                # Read number of cameras
                num_cameras, = struct.unpack_from('i', buffer, offset)
                offset += struct.calcsize('i')

                # Clear previous sensor data
                for item in self.lidar_points:
                    self.canvas.delete(item)
                self.lidar_points = []

                for rect in self.detected_robot_rectangles:
                    self.canvas.delete(rect)
                self.detected_robot_rectangles = []

                # Clear previous bounding boxes
                self.canvas.delete("bbox_static")
                self.canvas.delete("bbox_dynamic")

                lidar_points_for_clustering = []

                for _ in range(num_cameras):
                    # Read camera position
                    cam_x, cam_y = struct.unpack_from('ff', buffer, offset)
                    offset += struct.calcsize('ff')

                    # Read look_at position
                    look_x, look_y = struct.unpack_from('ff', buffer, offset)
                    offset += struct.calcsize('ff')

                    # Read number of rays
                    num_rays, = struct.unpack_from('i', buffer, offset)
                    offset += struct.calcsize('i')

                    for _ in range(num_rays):
                        # Read intersection point
                        inter_x, inter_y = struct.unpack_from('ff', buffer, offset)
                        offset += struct.calcsize('ff')

                        # Read distance
                        distance, = struct.unpack_from('f', buffer, offset)
                        offset += struct.calcsize('f')

                        # Read object ID
                        object_id, = struct.unpack_from('i', buffer, offset)
                        offset += struct.calcsize('i')

                        # Read ray angle
                        ray_angle_rad, = struct.unpack_from('f', buffer, offset)
                        offset += struct.calcsize('f')

                        # Process intersection point
                        if math.isfinite(inter_x) and math.isfinite(inter_y):
                            is_obstacle = self.is_point_in_obstacle(inter_x, inter_y)

                            canvas_x, canvas_y = self.world_to_canvas(inter_x, inter_y)

                            color = self.COLOR_STATIC_OBSTACLE if is_obstacle else self.COLOR_DYNAMIC_OBSTACLE
                            lidar_point = self.canvas.create_oval(
                                canvas_x - self.lidar_point_radius,
                                canvas_y - self.lidar_point_radius,
                                canvas_x + self.lidar_point_radius,
                                canvas_y + self.lidar_point_radius,
                                fill=color,
                                outline=color
                            )
                            self.lidar_points.append(lidar_point)

                            if not is_obstacle:
                                lidar_points_for_clustering.append([inter_x, inter_y])

                if lidar_points_for_clustering:
                    self.perform_lshape_fitting(lidar_points_for_clustering)

                # Update robot's position on the canvas
                canvas_x, canvas_y = self.world_to_canvas(robot_x, robot_y)

                self.canvas.coords(
                    self.robot,
                    canvas_x - self.robot_radius,
                    canvas_y - self.robot_radius,
                    canvas_x + self.robot_radius,
                    canvas_y + self.robot_radius
                )

                # Update robot's orientation line
                line_length = 20
                end_x = canvas_x + line_length * math.cos(robot_theta_rad)
                end_y = canvas_y - line_length * math.sin(robot_theta_rad)
                self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)

                # Update position and orientation label
                robot_theta_deg = math.degrees(robot_theta_rad) % 360
                self.position_label.config(
                    text=f"Position: ({robot_x:.2f}, {robot_y:.2f}), Orientation: {robot_theta_deg:.2f}°"
                )

                # Receive the path and bounding boxes from the queue
                while not self.path_queue.empty():
                    path_data = self.path_queue.get()
                    if isinstance(path_data, tuple) and len(path_data) == 2:
                        path_type, path = path_data
                        self.draw_path(path_type, path)
                    else:
                        # Assume it's a D*Lite path if no type is specified
                        self.draw_path('dstar', path_data)

                # Receive obstacle data from the main process
                while not self.obstacle_queue.empty():
                    obstacle_data = self.obstacle_queue.get()
                    if isinstance(obstacle_data, dict):
                        static_bboxes = obstacle_data.get('static_bboxes', [])
                        dynamic_bboxes = obstacle_data.get('dynamic_bboxes', [])
                        self.draw_bounding_boxes(static_bboxes, dynamic_bboxes)
                    else:
                        self.logger.error(f"Unexpected obstacle data format: {type(obstacle_data)}")
        except Exception as e:
            self.logger.error(f"Exception in update_visualization: {e}")
        finally:
            # Schedule the next update if not terminating
            if not self.termination_event.is_set():
                self.canvas.after(50, self.update_visualization)  # Adjust the delay as needed

    def draw_obstacles_persistently(self):
        if self.static_bboxes and self.dynamic_bboxes:
            self.draw_bounding_boxes(self.static_bboxes, self.dynamic_bboxes)

    def draw_bounding_boxes(self, static_bboxes, dynamic_bboxes):
        """
        Draw bounding boxes persistently without flashing.
        """
        # Remove only outdated bounding boxes
        self.canvas.delete("bbox_static")
        self.canvas.delete("bbox_dynamic")

        # Draw static bounding boxes
        for bbox in static_bboxes:
            x_min, y_min, x_max, y_max = bbox
            canvas_x1, canvas_y1 = self.world_to_canvas(x_min, y_min)
            canvas_x2, canvas_y2 = self.world_to_canvas(x_max, y_max)
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline=self.COLOR_STATIC_BBOX, width=2, tags="bbox_static"
            )

        # Draw dynamic bounding boxes
        for bbox in dynamic_bboxes:
            x_min, y_min, x_max, y_max = bbox
            canvas_x1, canvas_y1 = self.world_to_canvas(x_min, y_min)
            canvas_x2, canvas_y2 = self.world_to_canvas(x_max, y_max)
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline=self.COLOR_DYNAMIC_BBOX, width=2, tags="bbox_dynamic"
            )

    def close(self):
        if self.shm:
            self.shm.close()
            self.logger.info(f"Closed shared memory for {self.shm_name}")


def get_robot_state_from_shared_memory(shm, shm_size):
    """
    Get the robot's position and orientation from shared memory.

    Returns:
    - robot_x, robot_y, robot_theta_rad
    """
    buffer = shm.buf[:shm_size]
    offset = 0
    try:
        robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
        offset += struct.calcsize('ff')
        robot_theta_rad, = struct.unpack_from('f', buffer, offset)
        return robot_x, robot_y, robot_theta_rad
    except struct.error as e:
        logging.error(f"Struct unpacking error: {e}")
        return 0.0, 0.0, 0.0  # Default values


def visualization_main(
    env_shm_name,
    shm_size,
    lock,
    destination_queue,
    obstacle_queue_main_to_visual,
    path_queue,
    termination_event,
    detected_obstacle_queue
):
    """
    Visualization main function to handle mouse clicks and send destinations via destination_queue.

    This example uses tkinter to create a simple GUI window where mouse clicks are captured.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO to exclude DEBUG logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("visualization_process.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create the main window
    root = tk.Tk()
    root.title("Robot Destination Selector")
    root.geometry("800x600")  # Set window size as per your requirements

    # Create an instance of RobotPositionVisualization
    visualization = RobotPositionVisualization(
        parent=root,
        shm_name=env_shm_name,
        shm_size=shm_size,
        title="Visualization",
        destination_queue=destination_queue,
        obstacle_queue=obstacle_queue_main_to_visual,  # Main to Visual
        path_queue=path_queue,
        termination_event=termination_event,
        detected_obstacle_queue=detected_obstacle_queue,  # Visual to Main
        lock=lock,
        canvas_width=800,
        canvas_height=600,
        world_width=16.46,
        world_height=8.23
    )

    # Start the Tkinter main loop
    try:
        root.mainloop()
    except Exception as e:
        logging.error(f"An unexpected error occurred in visualization: {e}")
    finally:
        visualization.close()
        logging.info("Visualization process terminated gracefully.")
