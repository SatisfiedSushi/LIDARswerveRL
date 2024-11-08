# receiver.py

import tkinter as tk
from tkinter import ttk
import math
import struct
import time
import os
import logging
from multiprocessing import shared_memory
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SharedMemoryMapVisualization:
    def __init__(self, parent, shm_name, shm_size, scale=40, update_interval=50):
        self.parent = parent
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.scale = scale  # pixels per unit
        self.origin_x = 0      # Bottom-left corner
        self.origin_y = 600    # Canvas height to place origin at bottom-left
        self.update_interval = update_interval  # in milliseconds

        # Create Canvas
        self.canvas = tk.Canvas(parent, width=800, height=600, bg="white")
        self.canvas.pack()

        # Storage for obstacles and robot path
        self.obstacles = {}  # Maps object_id to (ix, iy)
        self.obstacle_graphics = {}  # Maps object_id to canvas item IDs
        self.robot_path = []

        # Create robot representation
        self.robot_radius = 10  # pixels
        self.robot = self.canvas.create_oval(
            self.origin_x - self.robot_radius,
            self.origin_y - self.robot_radius,
            self.origin_x + self.robot_radius,
            self.origin_y + self.robot_radius,
            fill='blue', outline='black'
        )

        # Create orientation indicator
        self.orientation_line = self.canvas.create_line(
            self.origin_x, self.origin_y,
            self.origin_x, self.origin_y - 20,
            fill='red', width=2
        )

        # Create timestamp label
        self.timestamp_label = tk.Label(parent, text="Timestamp: N/A")
        self.timestamp_label.pack(pady=5)

        # Connect to existing shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            logging.info(f"Connected to shared memory: {self.shm.name}")
        except FileNotFoundError:
            logging.error(f"Shared memory {self.shm_name} not found.")
            self.shm = None

        # Start the update loop
        self.update_map()

    def world_to_canvas(self, x, y):
        """
        Transforms world coordinates to canvas coordinates.
        """
        canvas_x = self.origin_x + x * self.scale
        canvas_y = self.origin_y - y * self.scale
        return (canvas_x, canvas_y)

    def update_map(self):
        """
        Reads data from shared memory and updates the map accordingly.
        """
        logging.debug("SharedMemoryMapVisualization: update_map called.")
        if not self.shm:
            logging.debug("SharedMemoryMapVisualization: Shared memory not connected. Skipping update.")
            self.parent.after(self.update_interval, self.update_map)
            return

        try:
            buffer = self.shm.buf[:self.shm_size]
            offset = 0

            # Robot Position
            if offset + struct.calcsize('ff') > len(buffer):
                raise ValueError("Insufficient data for robot position.")
            robot_x, robot_y = struct.unpack_from('ff', buffer, offset)
            logging.debug(f"SharedMemoryMapVisualization: Robot position unpacked: ({robot_x}, {robot_y})")
            offset += struct.calcsize('ff')

            # Robot Orientation
            if offset + struct.calcsize('f') > len(buffer):
                raise ValueError("Insufficient data for robot orientation.")
            robot_orientation, = struct.unpack_from('f', buffer, offset)
            logging.debug(f"SharedMemoryMapVisualization: Robot orientation unpacked: {robot_orientation} degrees")
            offset += struct.calcsize('f')

            # Timestamp
            if offset + struct.calcsize('f') > len(buffer):
                raise ValueError("Insufficient data for timestamp.")
            timestamp, = struct.unpack_from('f', buffer, offset)
            logging.debug(f"SharedMemoryMapVisualization: Timestamp unpacked: {timestamp}")
            offset += struct.calcsize('f')

            # Number of Cameras
            if offset + struct.calcsize('i') > len(buffer):
                raise ValueError("Insufficient data for number of cameras.")
            num_cameras, = struct.unpack_from('i', buffer, offset)
            logging.debug(f"SharedMemoryMapVisualization: Number of cameras unpacked: {num_cameras}")
            offset += struct.calcsize('i')

            depth_estimates = []
            for cam_index in range(num_cameras):
                # Camera Position
                if offset + struct.calcsize('ff') > len(buffer):
                    raise ValueError(f"Insufficient data for camera {cam_index} position.")
                cam_x, cam_y = struct.unpack_from('ff', buffer, offset)
                logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} position unpacked: ({cam_x}, {cam_y})")
                offset += struct.calcsize('ff')

                # Look At Position
                if offset + struct.calcsize('ff') > len(buffer):
                    raise ValueError(f"Insufficient data for camera {cam_index} look-at position.")
                look_x, look_y = struct.unpack_from('ff', buffer, offset)
                logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} look-at position unpacked: ({look_x}, {look_y})")
                offset += struct.calcsize('ff')

                # Number of Rays
                if offset + struct.calcsize('i') > len(buffer):
                    raise ValueError(f"Insufficient data for camera {cam_index} number of rays.")
                num_rays, = struct.unpack_from('i', buffer, offset)
                logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} number of rays unpacked: {num_rays}")
                offset += struct.calcsize('i')

                rays = []
                for ray_index in range(num_rays):
                    # Intersection
                    if offset + struct.calcsize('ff') > len(buffer):
                        raise ValueError(f"Insufficient data for camera {cam_index} ray {ray_index} intersection.")
                    ix, iy = struct.unpack_from('ff', buffer, offset)
                    logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} Ray {ray_index} intersection: ({ix}, {iy})")
                    offset += struct.calcsize('ff')

                    # Distance
                    if offset + struct.calcsize('f') > len(buffer):
                        raise ValueError(f"Insufficient data for camera {cam_index} ray {ray_index} distance.")
                    distance, = struct.unpack_from('f', buffer, offset)
                    logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} Ray {ray_index} distance unpacked: {distance}")
                    offset += struct.calcsize('f')

                    # Object ID
                    if offset + struct.calcsize('i') > len(buffer):
                        raise ValueError(f"Insufficient data for camera {cam_index} ray {ray_index} object ID.")
                    object_id, = struct.unpack_from('i', buffer, offset)
                    logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} Ray {ray_index} object ID unpacked: {object_id}")
                    offset += struct.calcsize('i')

                    # Ray Angle
                    if offset + struct.calcsize('f') > len(buffer):
                        raise ValueError(f"Insufficient data for camera {cam_index} ray {ray_index} angle.")
                    ray_angle, = struct.unpack_from('f', buffer, offset)
                    logging.debug(f"SharedMemoryMapVisualization: Camera {cam_index} Ray {ray_index} angle unpacked: {ray_angle}")
                    offset += struct.calcsize('f')

                    rays.append({
                        'intersection': (ix, iy) if ix != -1.0 and iy != -1.0 else None,
                        'distance': distance,
                        'object_id': object_id if object_id != -1 else None,
                        'ray_angle': ray_angle
                    })

                depth_estimates.append({
                    'camera_pos': (cam_x, cam_y),
                    'look_at_pos': (look_x, look_y),
                    'rays': rays
                })

            # Update robot position
            canvas_x, canvas_y = self.world_to_canvas(robot_x, robot_y)

            self.canvas.coords(
                self.robot,
                canvas_x - self.robot_radius,
                canvas_y - self.robot_radius,
                canvas_x + self.robot_radius,
                canvas_y + self.robot_radius
            )
            logging.debug(f"SharedMemoryMapVisualization: Robot moved to ({canvas_x}, {canvas_y}) on canvas.")

            # Update orientation line
            orientation_rad = math.radians(robot_orientation)
            line_length = 20  # pixels
            end_x = canvas_x + line_length * math.cos(orientation_rad)
            end_y = canvas_y - line_length * math.sin(orientation_rad)
            self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)
            logging.debug(f"SharedMemoryMapVisualization: Orientation line updated to ({end_x}, {end_y}).")

            # Store robot path
            self.robot_path.append((canvas_x, canvas_y))
            if len(self.robot_path) > 1:
                self.canvas.create_line(
                    self.robot_path[-2][0], self.robot_path[-2][1],
                    self.robot_path[-1][0], self.robot_path[-1][1],
                    fill='blue'
                )
                logging.debug("SharedMemoryMapVisualization: Robot path updated.")

            # Update timestamp label
            try:
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                logging.debug(f"SharedMemoryMapVisualization: Formatted timestamp: {formatted_time}")
            except (OSError, OverflowError, ValueError):
                formatted_time = "Invalid Timestamp"
                logging.warning("SharedMemoryMapVisualization: Timestamp is invalid.")
            self.timestamp_label.config(text=f"Timestamp: {formatted_time}")

            # Temporary storage for current detected object_ids
            current_object_ids = set()

            # Parse and plot obstacles
            for estimate in depth_estimates:
                rays = estimate["rays"]
                for ray in rays:
                    object_id = ray["object_id"]
                    intersection = ray["intersection"]
                    if object_id is None or intersection is None:
                        continue  # Skip if object_id is missing or intersection data is incomplete

                    ix, iy = intersection
                    obstacle_key = object_id  # Use object_id as the key
                    current_object_ids.add(object_id)

                    if obstacle_key in self.obstacles:
                        # Update existing obstacle
                        prev_ix, prev_iy = self.obstacles[obstacle_key]
                        if (ix, iy) != (prev_ix, prev_iy):
                            # Update position
                            self.obstacles[obstacle_key] = (ix, iy)
                            if obstacle_key in self.obstacle_graphics:
                                obstacle_canvas_x, obstacle_canvas_y = self.world_to_canvas(ix, iy)
                                self.canvas.coords(
                                    self.obstacle_graphics[obstacle_key],
                                    obstacle_canvas_x - 3,
                                    obstacle_canvas_y - 3,
                                    obstacle_canvas_x + 3,
                                    obstacle_canvas_y + 3
                                )
                                logging.debug(f"SharedMemoryMapVisualization: Obstacle {object_id} moved to ({ix}, {iy}).")
                    else:
                        # New obstacle
                        self.obstacles[obstacle_key] = (ix, iy)
                        obstacle_canvas_x, obstacle_canvas_y = self.world_to_canvas(ix, iy)
                        obstacle = self.canvas.create_oval(
                            obstacle_canvas_x - 3,
                            obstacle_canvas_y - 3,
                            obstacle_canvas_x + 3,
                            obstacle_canvas_y + 3,
                            fill='black'
                        )
                        self.obstacle_graphics[obstacle_key] = obstacle
                        logging.info(f"SharedMemoryMapVisualization: New obstacle added: ID={object_id}, Position=({ix}, {iy})")

            # Remove obstacles that are no longer detected
            existing_object_ids = set(self.obstacles.keys())
            object_ids_to_remove = existing_object_ids - current_object_ids
            for object_id in object_ids_to_remove:
                if object_id in self.obstacle_graphics:
                    self.canvas.delete(self.obstacle_graphics[object_id])
                    del self.obstacle_graphics[object_id]
                    logging.info(f"SharedMemoryMapVisualization: Obstacle removed: ID={object_id}")
                del self.obstacles[object_id]

        except Exception as e:
            logging.error(f"SharedMemoryMapVisualization: Error processing shared memory data: {e}")

        finally:
            # Schedule next update
            self.parent.after(self.update_interval, self.update_map)
            logging.debug("SharedMemoryMapVisualization: Scheduled next update.")


class FileBasedMapVisualization:
    def __init__(self, parent, data_file='depth_data.json', scale=10, update_interval=100):
        self.parent = parent
        self.data_file = data_file
        self.scale = scale  # pixels per unit
        self.origin_x = 400  # Center of the canvas
        self.origin_y = 300
        self.update_interval = update_interval  # in milliseconds

        # Create Canvas
        self.canvas = tk.Canvas(parent, width=800, height=600, bg="white")
        self.canvas.pack()

        # Storage for obstacles and robot path
        self.obstacles = {}  # Maps object_id to (ix, iy)
        self.obstacle_graphics = {}  # Maps object_id to canvas item IDs
        self.robot_path = []

        # Create robot representation
        self.robot_radius = 5  # pixels
        self.robot = self.canvas.create_oval(
            self.origin_x - self.robot_radius,
            self.origin_y - self.robot_radius,
            self.origin_x + self.robot_radius,
            self.origin_y + self.robot_radius,
            fill='blue', outline='black'
        )

        # Create orientation indicator
        self.orientation_line = self.canvas.create_line(
            self.origin_x, self.origin_y,
            self.origin_x, self.origin_y - 20,
            fill='red', width=2
        )

        # Create timestamp label
        self.timestamp_label = tk.Label(parent, text="Timestamp: N/A")
        self.timestamp_label.pack(pady=5)

        # Start the update loop
        self.update_map()

    def world_to_canvas(self, x, y):
        """
        Transforms world coordinates to canvas coordinates.
        """
        canvas_x = self.origin_x + x * self.scale
        canvas_y = self.origin_y - y * self.scale
        return (canvas_x, canvas_y)

    def update_map(self):
        """
        Reads the data file and updates the map accordingly.
        """
        logging.debug("FileBasedMapVisualization: update_map called.")
        if not os.path.exists(self.data_file):
            logging.error(f"FileBasedMapVisualization: Data file {self.data_file} does not exist.")
            self.parent.after(self.update_interval, self.update_map)
            return

        try:
            with open(self.data_file, 'r') as f:
                depth_data = json.load(f)
            logging.debug("FileBasedMapVisualization: Data file loaded successfully.")
        except json.JSONDecodeError as e:
            logging.error(f"FileBasedMapVisualization: JSON decode error: {e}")
            self.parent.after(self.update_interval, self.update_map)
            return
        except Exception as e:
            logging.error(f"FileBasedMapVisualization: Error reading data file: {e}")
            self.parent.after(self.update_interval, self.update_map)
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
            logging.warning("FileBasedMapVisualization: robot_position has an unexpected format.")

        # Transform to canvas coordinates
        canvas_x, canvas_y = self.world_to_canvas(x, y)

        # Move robot
        self.canvas.coords(
            self.robot,
            canvas_x - self.robot_radius,
            canvas_y - self.robot_radius,
            canvas_x + self.robot_radius,
            canvas_y + self.robot_radius
        )
        logging.debug(f"FileBasedMapVisualization: Robot moved to ({canvas_x}, {canvas_y}) on canvas.")

        # Update orientation line
        orientation_rad = math.radians(robot_orientation)
        line_length = 20  # pixels
        end_x = canvas_x + line_length * math.cos(orientation_rad)
        end_y = canvas_y - line_length * math.sin(orientation_rad)
        self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)
        logging.debug(f"FileBasedMapVisualization: Orientation line updated to ({end_x}, {end_y}).")

        # Store robot path
        self.robot_path.append((canvas_x, canvas_y))
        if len(self.robot_path) > 1:
            self.canvas.create_line(
                self.robot_path[-2][0], self.robot_path[-2][1],
                self.robot_path[-1][0], self.robot_path[-1][1],
                fill='blue'
            )
            logging.debug("FileBasedMapVisualization: Robot path updated.")

        # Update timestamp label
        if isinstance(timestamp, (int, float)):
            try:
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                logging.debug(f"FileBasedMapVisualization: Formatted timestamp: {formatted_time}")
            except (OSError, OverflowError, ValueError):
                formatted_time = "Invalid Timestamp"
                logging.warning("FileBasedMapVisualization: Timestamp is invalid.")
        else:
            formatted_time = str(timestamp)
            logging.warning("FileBasedMapVisualization: Timestamp is not a Unix timestamp.")
        self.timestamp_label.config(text=f"Timestamp: {formatted_time}")

        # Temporary storage for current detected object_ids
        current_object_ids = set()

        # Parse and plot obstacles
        for estimate in depth_estimates:
            rays = estimate.get("rays", [])
            for ray in rays:
                object_id = ray.get("object_id", None)
                intersection = ray.get("intersection", [])
                if object_id is None or len(intersection) < 2:
                    continue  # Skip if object_id is missing or intersection data is incomplete

                ix, iy = intersection[0], intersection[1]
                obstacle_key = object_id  # Use object_id as the key
                current_object_ids.add(object_id)

                if object_id in self.obstacles:
                    # Update existing obstacle
                    prev_ix, prev_iy = self.obstacles[object_id]
                    if (ix, iy) != (prev_ix, prev_iy):
                        # Update position
                        self.obstacles[object_id] = (ix, iy)
                        if object_id in self.obstacle_graphics:
                            obstacle_canvas_x, obstacle_canvas_y = self.world_to_canvas(ix, iy)
                            self.canvas.coords(
                                self.obstacle_graphics[object_id],
                                obstacle_canvas_x - 3,
                                obstacle_canvas_y - 3,
                                obstacle_canvas_x + 3,
                                obstacle_canvas_y + 3
                            )
                            logging.debug(f"FileBasedMapVisualization: Obstacle {object_id} moved to ({ix}, {iy}).")
                else:
                    # New obstacle
                    self.obstacles[object_id] = (ix, iy)
                    obstacle_canvas_x, obstacle_canvas_y = self.world_to_canvas(ix, iy)
                    obstacle = self.canvas.create_oval(
                        obstacle_canvas_x - 3,
                        obstacle_canvas_y - 3,
                        obstacle_canvas_x + 3,
                        obstacle_canvas_y + 3,
                        fill='black'
                    )
                    self.obstacle_graphics[object_id] = obstacle
                    logging.info(f"FileBasedMapVisualization: New obstacle added: ID={object_id}, Position=({ix}, {iy})")

        # Remove obstacles that are no longer detected
        existing_object_ids = set(self.obstacles.keys())
        object_ids_to_remove = existing_object_ids - current_object_ids
        for object_id in object_ids_to_remove:
            if object_id in self.obstacle_graphics:
                self.canvas.delete(self.obstacle_graphics[object_id])
                del self.obstacle_graphics[object_id]
                logging.info(f"FileBasedMapVisualization: Obstacle removed: ID={object_id}")
            del self.obstacles[object_id]

        # Schedule next update
        self.parent.after(self.update_interval, self.update_map)
        logging.debug("FileBasedMapVisualization: Scheduled next update.")


class ReceiverApp:
    def __init__(self, root, shm_name, shm_size, data_file='depth_data.json'):
        self.root = root
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.data_file = data_file

        # Create Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each tab
        self.shared_memory_frame = ttk.Frame(self.notebook)
        self.file_based_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.shared_memory_frame, text='Shared Memory Visualization')
        self.notebook.add(self.file_based_frame, text='File-Based Mapping')

        # Initialize visualization classes
        self.shared_memory_viz = SharedMemoryMapVisualization(
            parent=self.shared_memory_frame,
            shm_name=self.shm_name,
            shm_size=self.shm_size
        )

        self.file_based_viz = FileBasedMapVisualization(
            parent=self.file_based_frame,
            data_file=self.data_file
        )


def main():
    # Configuration
    shm_name = "my_shared_memory"  # Must match the sender's shared memory name
    shm_size = 16060  # Must match the sender's shared memory size (calculated in env.py)
    data_file = 'depth_data.json'  # Path to the JSON file for file-based mapping

    root = tk.Tk()
    root.title("SLAM Visualization")
    app = ReceiverApp(root, shm_name, shm_size, data_file)
    root.mainloop()


if __name__ == "__main__":
    main()
