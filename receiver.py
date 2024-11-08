import logging
import tkinter as tk
import math
import struct
import time
from multiprocessing import shared_memory
import sys

class SharedMemoryVisualization:
    def __init__(self, parent, shm_output_name='slam_output', shm_output_size=16060, scale=200, canvas_width=1000, canvas_height=500):
        """
        Initializes the visualization with modified scale and initial robot centering.
        """
        self.parent = parent
        self.shm_output_name = shm_output_name
        self.shm_output_size = shm_output_size
        self.scale = scale  # Increased to 200 for closer zoom
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.origin_x = self.canvas_width // 2
        self.origin_y = self.canvas_height // 2

        # Set up world center for environment
        self.world_center_x = 8.23
        self.world_center_y = 4.115

        # Logger setup
        self.logger = logging.getLogger('SharedMemoryVisualization')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # Initialize Canvas
        self.canvas = tk.Canvas(parent, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Draw Grid
        self.draw_grid()

        # Initialize robot representation
        self.robot_radius = 6  # Smaller radius for centered look
        self.robot = self.canvas.create_oval(
            self.origin_x - self.robot_radius,
            self.origin_y - self.robot_radius,
            self.origin_x + self.robot_radius,
            self.origin_y + self.robot_radius,
            fill='blue'
        )
        self.orientation_line = self.canvas.create_line(
            self.origin_x, self.origin_y,
            self.origin_x, self.origin_y - 20,
            fill='red', width=2
        )

        # Map graphics storage and timestamp
        self.map_graphics = []
        self.timestamp_label = tk.Label(parent, text="Timestamp: N/A")
        self.timestamp_label.pack(pady=5)

        # Connect to shared memory
        try:
            self.shm_output = shared_memory.SharedMemory(name=self.shm_output_name)
            self.logger.info(f"Connected to shared memory: {self.shm_output.name}")
        except FileNotFoundError:
            self.logger.error(f"Shared memory '{self.shm_output_name}' not found.")
            self.shm_output = None

        # Start update loop
        self.update_visualization()

    def draw_grid(self):
        grid_spacing = 1 * self.scale  # 1 meter grid with the new scale
        for x in range(0, self.canvas_width, grid_spacing):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgray')
        for y in range(0, self.canvas_height, grid_spacing):
            self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgray')

    def world_to_canvas(self, x, y):
        """
        Convert world coordinates to canvas coordinates.
        """
        canvas_x = self.origin_x + (x - self.world_center_x) * self.scale
        canvas_y = self.origin_y - (y - self.world_center_y) * self.scale
        return (canvas_x, canvas_y)

    def update_visualization(self):
        try:
            if self.shm_output:
                slam_pose, slam_map, slam_timestamp = self.read_slam_output()

                if slam_pose and slam_map is not None:
                    canvas_x, canvas_y = self.world_to_canvas(slam_pose[0], slam_pose[1])

                    # Update robot position
                    self.canvas.coords(
                        self.robot,
                        canvas_x - self.robot_radius,
                        canvas_y - self.robot_radius,
                        canvas_x + self.robot_radius,
                        canvas_y + self.robot_radius
                    )
                    orientation_rad = math.radians(slam_pose[2])
                    line_length = 20
                    end_x = canvas_x + line_length * math.cos(orientation_rad)
                    end_y = canvas_y - line_length * math.sin(orientation_rad)
                    self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)

                    # Update timestamp
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(slam_timestamp))
                    self.timestamp_label.config(text=f"Timestamp: {formatted_time}")

                    # Update SLAM map
                    self.update_map(slam_map)
                else:
                    self.logger.warning("Invalid SLAM data received.")
        except Exception as e:
            self.logger.error(f"Exception in update_visualization: {e}")
        finally:
            self.parent.after(100, self.update_visualization)

    def read_slam_output(self):
        try:
            buffer = self.shm_output.buf[:self.shm_output_size]
            offset = 0
            robot_x, robot_y, robot_theta_deg = struct.unpack_from('fff', buffer, offset)
            offset += struct.calcsize('fff')
            timestamp, = struct.unpack_from('f', buffer, offset)
            offset += struct.calcsize('f')
            num_landmarks, = struct.unpack_from('i', buffer, offset)
            offset += struct.calcsize('i')

            landmarks = []
            for _ in range(num_landmarks):
                lx, ly = struct.unpack_from('ff', buffer, offset)
                offset += struct.calcsize('ff')
                landmarks.append((lx, ly))

            return (robot_x, robot_y, robot_theta_deg), landmarks, timestamp
        except struct.error as e:
            self.logger.error(f"Struct unpacking error: {e}")
            return None, None, None

    def update_map(self, landmarks):
        for item in self.map_graphics:
            self.canvas.delete(item)
        self.map_graphics = []
        for lx, ly in landmarks:
            canvas_x, canvas_y = self.world_to_canvas(lx, ly)
            if 0 <= canvas_x <= self.canvas_width and 0 <= canvas_y <= self.canvas_height:
                point = self.canvas.create_oval(
                    canvas_x - 1, canvas_y - 1,  # Smaller landmarks
                    canvas_x + 1, canvas_y + 1,
                    fill='green'
                )
                self.map_graphics.append(point)

    def close(self):
        if self.shm_output:
            self.shm_output.close()
            self.logger.info("Shared memory closed.")

def visualization_main(shm_output_name='slam_output', shm_output_size=16060):
    root = tk.Tk()
    root.title("SLAM Visualization")

    app = SharedMemoryVisualization(
        parent=root,
        shm_output_name=shm_output_name,
        shm_output_size=shm_output_size
    )
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("Visualization interrupted.")
    finally:
        app.close()
        root.destroy()
        logging.info("Visualization terminated.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SLAM Visualization Receiver.')
    parser.add_argument('--shm_output_name', type=str, default='slam_output', help='Name of output shared memory.')
    parser.add_argument('--shm_output_size', type=int, default=16060, help='Size of output shared memory in bytes.')
    args = parser.parse_args()

    visualization_main(shm_output_name=args.shm_output_name, shm_output_size=args.shm_output_size)
