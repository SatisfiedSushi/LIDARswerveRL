# receiver.py

import logging
import tkinter as tk
import struct
import math
from multiprocessing import shared_memory
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class RobotPositionVisualization:
    def __init__(self, parent, shm_name, shm_size, title, canvas_width=824, canvas_height=412, world_width=16.46,
                 world_height=8.23):
        """
        Initializes a visualization window for either the actual or SLAM position.

        Parameters:
        - shm_name: Name of the shared memory segment to read data from.
        - title: Title of the visualization (e.g., "Actual Position" or "SLAM Position").
        """
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.world_width = world_width
        self.world_height = world_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.scale_x = canvas_width / world_width
        self.scale_y = canvas_height / world_height

        # Logger setup
        self.logger = logging.getLogger(title)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Tkinter canvas setup
        self.canvas = tk.Canvas(parent, width=canvas_width, height=canvas_height, bg='white')
        self.canvas.pack()

        # Robot representation
        self.robot_radius = 10
        self.robot = self.canvas.create_oval(0, 0, 0, 0, fill='blue')
        self.orientation_line = self.canvas.create_line(0, 0, 0, 0, fill='red', width=2)

        # Position and orientation display
        self.position_label = tk.Label(parent, text="Position: (x, y), Orientation: θ°")
        self.position_label.pack()

        # Connect to shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.logger.info(f"Connected to shared memory: {self.shm_name}")
        except FileNotFoundError:
            self.logger.error(f"Shared memory '{self.shm_name}' not found.")
            self.shm = None

        # Start update loop
        self.update_visualization()

    def world_to_canvas(self, x, y):
        """
        Convert world coordinates to canvas coordinates.
        """
        canvas_x = (x / self.world_width) * self.canvas_width
        canvas_y = self.canvas_height - ((y / self.world_height) * self.canvas_height)
        return canvas_x, canvas_y

    def update_visualization(self):
        """
        Reads data from shared memory and updates the robot's position on the canvas.
        """
        try:
            if self.shm:
                buffer = self.shm.buf[:self.shm_size]
                robot_x, robot_y, robot_theta_deg = struct.unpack_from('fff', buffer, 0)

                # Convert to canvas coordinates
                canvas_x, canvas_y = self.world_to_canvas(robot_x, robot_y)

                # Update robot position
                self.canvas.coords(
                    self.robot,
                    canvas_x - self.robot_radius,
                    canvas_y - self.robot_radius,
                    canvas_x + self.robot_radius,
                    canvas_y + self.robot_radius
                )

                # Update orientation line
                orientation_rad = math.radians(robot_theta_deg)
                end_x = canvas_x + 20 * math.cos(orientation_rad)
                end_y = canvas_y - 20 * math.sin(orientation_rad)
                self.canvas.coords(self.orientation_line, canvas_x, canvas_y, end_x, end_y)

                # Update position and orientation label
                self.position_label.config(
                    text=f"Position: ({robot_x:.2f}, {robot_y:.2f}), Orientation: {robot_theta_deg:.2f}°")

                # self.logger.debug(f"Position: ({robot_x}, {robot_y}), Orientation: {robot_theta_deg} degrees")

        except Exception as e:
            self.logger.error(f"Exception in update_visualization: {e}")
        finally:
            # Schedule next update
            self.canvas.after(100, self.update_visualization)

    def close(self):
        """
        Close the shared memory connection.
        """
        if self.shm:
            self.shm.close()
            self.logger.info(f"Closed shared memory for {self.shm_name}")


def visualization_main(env_shm_name='my_shared_memory', slam_shm_name='slam_output', shm_size=16060):
    root = tk.Tk()
    root.title("Robot Position Visualization")

    # Create two frames for the two visuals
    env_frame = tk.Frame(root)
    env_frame.pack(side=tk.LEFT)
    slam_frame = tk.Frame(root)
    slam_frame.pack(side=tk.RIGHT)

    # Instantiate two visualizations: one for actual and one for SLAM-estimated positions
    env_visualization = RobotPositionVisualization(env_frame, shm_name=env_shm_name, shm_size=shm_size,
                                                   title="Actual Position")
    slam_visualization = RobotPositionVisualization(slam_frame, shm_name=slam_shm_name, shm_size=shm_size,
                                                    title="SLAM Position")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("Visualization interrupted.")
    finally:
        env_visualization.close()
        slam_visualization.close()
        root.destroy()
        logging.info("Visualization ended.")


if __name__ == "__main__":
    visualization_main()
