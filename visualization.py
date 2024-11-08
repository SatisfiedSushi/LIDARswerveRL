# visualization.py

import tkinter as tk
import math
import struct
import time
from multiprocessing import shared_memory
import numpy as np

class SharedMemoryVisualization:
    def __init__(self, parent, shm_output_name='slam_output', shm_output_size=16060, scale=40, canvas_width=800, canvas_height=600):
        self.parent = parent
        self.shm_output_name = shm_output_name
        self.shm_output_size = shm_output_size
        self.scale = scale  # pixels per meter
        self.origin_x = canvas_width // 2  # Center of canvas
        self.origin_y = canvas_height // 2  # Center of canvas
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Initialize Canvas
        self.canvas = tk.Canvas(parent, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Initialize robot representation
        self.robot_radius = 10  # pixels
        self.robot = self.canvas.create_oval(
            self.origin_x - self.robot_radius,
            self.origin_y - self.robot_radius,
            self.origin_x + self.robot_radius,
            self.origin_y + self.robot_radius,
            fill='blue'
        )
        # Orientation line
        self.orientation_line = self.canvas.create_line(
            self.origin_x, self.origin_y,
            self.origin_x, self.origin_y - 20,
            fill='red', width=2
        )

        # Initialize map graphics
        self.map_graphics = []

        # Ground truth landmarks (if any)
        self.ground_truth_graphics = []

        # Timestamp label
        self.timestamp_label = tk.Label(parent, text="Timestamp: N/A")
        self.timestamp_label.pack(pady=5)

        # Connect to output shared memory
        try:
            self.shm_output = shared_memory.SharedMemory(name=self.shm_output_name)
            print(f"Visualization connected to shared memory: {self.shm_output.name}")
        except FileNotFoundError:
            print(f"Visualization: Shared memory '{self.shm_output_name}' not found.")
            self.shm_output = None  # Handle absence gracefully

        # Start update loop
        self.update_visualization()

    def world_to_canvas(self, x, y):
        """
        Convert world coordinates to canvas coordinates.
        Adjust origin as needed.
        """
        canvas_x = self.origin_x + x * self.scale
        canvas_y = self.origin_y - y * self.scale
        return (canvas_x, canvas_y)

    def update_visualization(self):
        """
        Read data from shared memory and update visualization.
        """
        if self.shm_output:
            # Read SLAM output
            slam_pose, slam_map, slam_timestamp = self.read_slam_output()

            if slam_pose and slam_map is not None:
                # Update robot position
                canvas_x, canvas_y = self.world_to_canvas(slam_pose[0], slam_pose[1])

                # Update robot representation
                self.canvas.coords(
                    self.robot,
                    canvas_x - self.robot_radius,
                    canvas_y - self.robot_radius,
                    canvas_x + self.robot_radius,
                    canvas_y + self.robot_radius
                )
                # Update orientation line
                orientation_rad = slam_pose[2]  # Already in radians
                line_length = 20  # pixels
                end_x = canvas_x + line_length * math.cos(orientation_rad)
                end_y = canvas_y - line_length * math.sin(orientation_rad)
                self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)
                # Update timestamp
                try:
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(slam_timestamp))
                except (OSError, OverflowError, ValueError):
                    formatted_time = "Invalid Timestamp"
                self.timestamp_label.config(text=f"Timestamp: {formatted_time}")
                # Update SLAM map
                self.update_map(slam_map)
                # Optionally, display ground truth landmarks
                # self.display_ground_truth()
        # Schedule next update
        self.parent.after(100, self.update_visualization)  # Update every 100 ms

    def read_slam_output(self):
        """
        Read SLAM output from shared memory.
        Returns pose, map, timestamp.
        """
        try:
            buffer = self.shm_output.buf[:self.shm_output_size]
            offset = 0
            # Unpack robot pose
            robot_x, robot_y, robot_theta_rad = struct.unpack_from('fff', buffer, offset)
            offset += struct.calcsize('fff')
            # Unpack timestamp
            timestamp, = struct.unpack_from('d', buffer, offset)
            offset += struct.calcsize('d')
            # Unpack number of landmarks
            num_landmarks, = struct.unpack_from('i', buffer, offset)
            offset += struct.calcsize('i')
            # Unpack landmark positions
            landmarks = []
            for _ in range(num_landmarks):
                lx, ly = struct.unpack_from('ff', buffer, offset)
                offset += struct.calcsize('ff')
                landmarks.append((lx, ly))
            return (robot_x, robot_y, robot_theta_rad), landmarks, timestamp
        except struct.error as e:
            print(f"Visualization: Struct unpacking error in SLAM output: {e}")
            return None, None, None
        except Exception as e:
            print(f"Visualization: Error reading shared memory: {e}")
            return None, None, None

    def update_map(self, landmarks):
        """
        Draw SLAM map landmarks as circles.
        """
        # Clear existing map graphics
        for item in self.map_graphics:
            self.canvas.delete(item)
        self.map_graphics = []
        # Draw landmarks
        for lx, ly in landmarks:
            canvas_x, canvas_y = self.world_to_canvas(lx, ly)
            point = self.canvas.create_oval(
                canvas_x - 3, canvas_y - 3,
                canvas_x + 3, canvas_y + 3,
                fill='green'
            )
            self.map_graphics.append(point)

    def close(self):
        if self.shm_output:
            self.shm_output.close()
            print("Visualization: Shared memory closed.")

def visualization_main(shm_output_name, shm_output_size):
    root = tk.Tk()
    root.title("SLAM Visualization")
    app = SharedMemoryVisualization(
        parent=root,
        shm_output_name=shm_output_name,
        shm_output_size=shm_output_size
    )
    root.mainloop()
