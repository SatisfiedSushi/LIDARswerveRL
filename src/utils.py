# src/utils.py

import json
import os
import numpy as np


def load_sensor_data(json_file_path):
    """
    Load sensor data from a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract robot position and orientation
    robot_pos = data.get("robot_position", {})
    x = robot_pos.get("x", 0.0)
    y = robot_pos.get("y", 0.0)
    theta_deg = data.get("robot_orientation", 0.0)
    theta = np.deg2rad(theta_deg % 360)  # Convert to radians and normalize

    # Extract timestamp
    timestamp = data.get("timestamp", 0.0)

    # Extract depth estimates
    depth_estimates = data.get("depth_estimates", [])
    all_rays = []
    for depth in depth_estimates:
        camera_pos = depth.get("camera_pos", [0.0, 0.0])
        look_at_pos = depth.get("look_at_pos", [0.0, 0.0])
        cam_angle_deg = depth.get("cam_angle", 0.0)
        cam_angle = np.deg2rad(cam_angle_deg % 360)

        # Process rays
        rays = depth.get("rays", [])
        for ray in rays:
            intersection = ray.get("intersection", [0.0, 0.0])
            distance = ray.get("distance", 0.0)
            object_id = ray.get("object_id", None)  # None if not available
            ray_angle_deg = ray.get("ray_angle", 0.0)
            ray_angle = np.deg2rad(ray_angle_deg)  # Convert to radians

            # Calculate the absolute angle of the ray
            absolute_angle = theta + ray_angle

            # Calculate relative position to robot
            relative_x = distance * np.cos(absolute_angle)
            relative_y = distance * np.sin(absolute_angle)

            # World coordinates
            world_x = x + relative_x
            world_y = y + relative_y

            all_rays.append({
                "world_x": world_x,
                "world_y": world_y,
                "object_id": object_id
            })

    return {
        "timestamp": timestamp,
        "robot_pose": [x, y, theta],
        "rays": all_rays
    }
