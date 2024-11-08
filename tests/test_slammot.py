# main.py

import time
import numpy as np
import os
from src.localization import RobotLocalizationEKF
from src.mapping import OccupancyGridMap
from src.detection import MovingObjectDetector
from src.tracking import MHTTracker
from src.visualization import Visualization
from src.utils import load_sensor_data


def main():
    # Initialize components
    initial_pose = [0, 0, 0]  # [x, y, theta] initial pose
    initial_covariance = np.diag([0.1, 0.1, np.deg2rad(5) ** 2])
    ekf = RobotLocalizationEKF(initial_pose, initial_covariance)

    map_size = 200  # meters
    map_resolution = 0.2  # meters per cell
    occupancy_map = OccupancyGridMap(map_size, map_size, map_resolution)

    # Initialize detector with the initial pose
    moving_object_detector = MovingObjectDetector(occupancy_map, ekf.get_pose())

    # Initialize tracker
    mht_tracker = MHTTracker(max_missed=5)

    # Initialize visualization
    visualization = Visualization(map_size, map_resolution)

    # Path to JSON data directory
    json_data_dir = "data/"
    json_files = sorted([f for f in os.listdir(json_data_dir) if f.endswith('.json')])

    for json_file in json_files:
        json_path = os.path.join(json_data_dir, json_file)
        sensor_data = load_sensor_data(json_path)

        timestamp = sensor_data["timestamp"]
        robot_pose = sensor_data["robot_pose"]  # [x, y, theta]
        rays = sensor_data["rays"]  # List of rays with world_x, world_y, object_id

        # Update EKF with the provided robot pose
        ekf.ekf.x = np.array(robot_pose)
        ekf.ekf.P = initial_covariance  # Adjust covariance as needed

        # Update occupancy map
        measurements = [[ray["world_x"], ray["world_y"]] for ray in rays]
        occupancy_map.update(robot_pose, measurements)

        # Detect moving objects based on object_id
        moving_points = [ray for ray in rays if ray["object_id"] is not None]
        detections = [[ray["world_x"], ray["world_y"]] for ray in moving_points]
        object_ids = [ray["object_id"] for ray in moving_points]

        # Update tracker with detections and their object_ids
        mht_tracker.update(detections, object_ids, dt=0.1)  # Adjust dt based on data rate

        # Update visualization
        visualization.update(ekf.get_pose(), occupancy_map.get_probability_map(),
                             mht_tracker.get_active_tracks())

        # Optional: Handle camera data (e.g., display images)
        # image_path = sensor_data.get("camera", {}).get("image_path", "")
        # if image_path and os.path.exists(image_path):
        #     # Display or process the image
        #     pass

        # Simulate real-time processing
        time.sleep(0.1)  # Adjust based on your actual data rate

    # Final visualization
    visualization.show()


if __name__ == "__main__":
    main()
