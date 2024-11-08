# src/visualization.py

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as mlines


class Visualization:
    def __init__(self, map_size, resolution):
        self.map_size = map_size
        self.resolution = resolution
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-map_size / 2, map_size / 2)
        self.ax.set_ylim(-map_size / 2, map_size / 2)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('SLAMMOT Visualization')
        self.robot_patch = Circle((0, 0), 1.0, fill=True, color='red', label='Robot')
        self.ax.add_patch(self.robot_patch)
        self.tracks_patches = {}
        self.track_colors = {}
        self.colors = plt.cm.get_cmap('hsv', 256)

    def update(self, robot_pose, occupancy_map, tracks):
        self.ax.clear()
        self.ax.set_xlim(-self.map_size / 2, self.map_size / 2)
        self.ax.set_ylim(-self.map_size / 2, self.map_size / 2)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('SLAMMOT Visualization')

        # Plot occupancy grid
        prob_map = occupancy_map
        self.ax.imshow(prob_map.T, cmap='gray', origin='lower',
                       extent=[-self.map_size / 2, self.map_size / 2,
                               -self.map_size / 2, self.map_size / 2],
                       alpha=0.5)

        # Plot robot
        x, y, theta = robot_pose
        self.robot_patch = Circle((x, y), 1.0, fill=True, color='red', label='Robot')
        self.ax.add_patch(self.robot_patch)
        # Indicate robot orientation
        arrow_length = 1.5
        arrow_x = x + arrow_length * np.cos(theta)
        arrow_y = y + arrow_length * np.sin(theta)
        self.ax.arrow(x, y, arrow_length * np.cos(theta), arrow_length * np.sin(theta),
                      head_width=0.5, head_length=0.5, fc='red', ec='red')

        # Plot tracked objects
        for track in tracks:
            obj_id = track.id
            obj_x, obj_y = track.position
            if obj_id not in self.tracks_patches:
                color = self.colors(obj_id % 256)
                self.track_colors[obj_id] = color
                self.tracks_patches[obj_id] = Circle((obj_x, obj_y), 0.5, fill=True, color=color,
                                                     label=f'Object {obj_id}')
                self.ax.add_patch(self.tracks_patches[obj_id])
            else:
                self.tracks_patches[obj_id].center = (obj_x, obj_y)

            # Optionally, plot trajectory
            history = np.array(track.history)
            self.ax.plot(history[:, 0], history[:, 1], linestyle='--', color=self.track_colors[obj_id])
            # Annotate with object_id
            self.ax.text(obj_x, obj_y, str(obj_id), color='white', fontsize=8,
                         ha='center', va='center')

        # Create a legend
        robot_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                     markersize=10, label='Robot')
        self.ax.legend(handles=[robot_legend], loc='upper right')

        plt.pause(0.001)

    def show(self):
        plt.show()
