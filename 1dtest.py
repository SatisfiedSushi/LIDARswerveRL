import tkinter as tk
import numpy as np
import math


# Function to rotate a point (x, y) around another point (cx, cy) by a given angle
def rotate_point(x, y, cx, cy, angle):
    radians = math.radians(angle)
    cos = math.cos(radians)
    sin = math.sin(radians)
    nx = cos * (x - cx) - sin * (y - cy) + cx
    ny = sin * (x - cx) + cos * (y - cy) + cy
    return nx, ny


# Function to create a box with size and rotation
def create_box(center, size, angle):
    cx, cy = center
    half_size = size / 2
    # Define the four corners of the box
    corners = [
        (cx - half_size, cy - half_size),
        (cx + half_size, cy - half_size),
        (cx + half_size, cy + half_size),
        (cx - half_size, cy + half_size)
    ]
    # Rotate the corners around the center
    rotated_corners = [rotate_point(x, y, cx, cy, angle) for x, y in corners]
    return rotated_corners


# Raycasting function to check for intersections between a ray and line segments (edges of boxes)
def ray_intersects_segment(ray_origin, ray_direction, seg_start, seg_end):
    ray_dx, ray_dy = ray_direction
    seg_dx, seg_dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]

    denominator = (-seg_dx * ray_dy + ray_dx * seg_dy)
    if denominator == 0:
        return None  # Parallel lines

    t = (-ray_dy * (ray_origin[0] - seg_start[0]) + ray_dx * (ray_origin[1] - seg_start[1])) / denominator
    u = (seg_dx * (ray_origin[1] - seg_start[1]) - seg_dy * (ray_origin[0] - seg_start[0])) / denominator

    if 0 <= t <= 1 and u >= 0:
        intersection_x = seg_start[0] + t * seg_dx
        intersection_y = seg_start[1] + t * seg_dy
        return (intersection_x, intersection_y), u  # Return intersection and distance along the ray
    return None


# Raycasting function that casts rays from the camera in different directions
def cast_rays(camera_pos, look_at_pos, fov_angle, ray_count, boxes):
    cam_angle = calculate_angle(camera_pos, look_at_pos)
    half_fov = fov_angle / 2

    # Define the rays by their angles
    rays = []
    for i in range(ray_count):
        ray_angle = cam_angle - half_fov + i * (fov_angle / (ray_count - 1))
        ray_direction = (math.cos(math.radians(ray_angle)), math.sin(math.radians(ray_angle)))
        rays.append((ray_angle, ray_direction))

    # List to store intersections of each ray with boxes
    ray_intersections = []
    for ray_angle, ray_direction in rays:
        closest_intersection = None
        closest_distance = float('inf')
        hit_object_id = None

        for object_id, box in enumerate(boxes):  # Track which box is hit
            for i in range(len(box)):
                seg_start = box[i]
                seg_end = box[(i + 1) % len(box)]

                intersection = ray_intersects_segment(camera_pos, ray_direction, seg_start, seg_end)
                if intersection:
                    point, distance = intersection
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = point
                        hit_object_id = object_id  # Mark the object hit by the ray

        ray_intersections.append((closest_intersection, closest_distance, hit_object_id))

    return ray_intersections



# Function to calculate the angle between two points
def calculate_angle(camera_pos, look_at_pos):
    dx = look_at_pos[0] - camera_pos[0]
    dy = look_at_pos[1] - camera_pos[1]
    return math.degrees(math.atan2(dy, dx))


# Function to draw the 2D perspective with raycasting and FOV visualization
# Function to draw the 2D perspective with raycasting and FOV visualization
def draw_2d_perspective(canvas, boxes, camera_pos, look_at_pos, fov_angle, ray_count):
    canvas.delete("all")

    # Draw boxes
    for box in boxes:
        canvas.create_polygon(box, outline='black', fill='', width=2)

    # Draw camera and look-at point
    canvas.create_oval(camera_pos[0] - 5, camera_pos[1] - 5, camera_pos[0] + 5, camera_pos[1] + 5, fill="red")
    canvas.create_text(camera_pos[0], camera_pos[1], text="Cam", anchor="nw", fill="red")
    canvas.create_oval(look_at_pos[0] - 5, look_at_pos[1] - 5, look_at_pos[0] + 5, look_at_pos[1] + 5, fill="green")
    canvas.create_text(look_at_pos[0], look_at_pos[1], text="Look", anchor="nw", fill="green")

    # Draw FOV lines
    cam_angle = calculate_angle(camera_pos, look_at_pos)
    half_fov = fov_angle / 2
    fov_left_angle = cam_angle - half_fov
    fov_right_angle = cam_angle + half_fov
    fov_length = 500

    left_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_left_angle))
    left_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_left_angle))
    right_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_right_angle))
    right_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_right_angle))

    canvas.create_line(camera_pos[0], camera_pos[1], left_x, left_y, fill='gray', dash=(4, 2))
    canvas.create_line(camera_pos[0], camera_pos[1], right_x, right_y, fill='gray', dash=(4, 2))

    # Draw rays
    ray_intersections = cast_rays(camera_pos, look_at_pos, fov_angle, ray_count, boxes)
    for intersection, _, _ in ray_intersections:  # Unpack all three values but use only intersection here
        if intersection:
            canvas.create_line(camera_pos[0], camera_pos[1], intersection[0], intersection[1], fill='red')

    return ray_intersections


def print_1d_perspective(ray_intersections, ray_count, fov_angle):
    half_fov = fov_angle / 2
    object_perspective = {}

    prev_object_id = None
    start_angle = None

    for i, (_, _, object_id) in enumerate(ray_intersections):
        # Normalize the ray_angle from [-half_fov, half_fov] to [-1, 1]
        ray_angle = (-half_fov + i * (fov_angle / (ray_count - 1))) / half_fov

        if object_id != prev_object_id:
            if prev_object_id is not None:
                # Store the previous object with its start and end angles
                object_name = f"obj{prev_object_id + 1}"
                object_perspective[object_name] = (start_angle, ray_angle)
            start_angle = ray_angle  # New object's start angle

        prev_object_id = object_id

    if prev_object_id is not None:
        # Store the last object
        object_name = f"obj{prev_object_id + 1}"
        object_perspective[object_name] = (start_angle, ray_angle)

    # Print the 1D perspective dictionary with angles between -1 and 1
    print(object_perspective)


# Function to draw the 1D perspective with gap detection
def draw_1d_perspective_with_gaps(canvas, ray_intersections, ray_count, fov_angle):
    canvas.delete("all")

    # Draw origin at camera in 1D space (camera's 1D projection)
    canvas.create_oval(300 - 5, 100 - 5, 300 + 5, 100 + 5, fill="red")
    canvas.create_text(300, 100, text="Cam", anchor="nw", fill="red")

    # Scale factor for distance normalization
    max_distance = max([distance for _, distance, _ in ray_intersections if distance != float('inf')])

    if max_distance == 0:
        max_distance = 1  # Prevent division by zero

    # Plot each intersection along the x-axis of the 1D canvas based on angular position
    half_canvas_width = 300
    half_fov = fov_angle / 2

    prev_x = None
    prev_y = 100
    gap_threshold = 10  # Define a gap threshold for detecting significant distance changes

    prev_object_id = None  # Track the previous object that the ray hit

    for i, (intersection, distance, object_id) in enumerate(ray_intersections):
        if intersection and distance != float('inf'):
            # Calculate the angular position relative to the center of the FOV
            ray_angle = (-half_fov + i * (fov_angle / (ray_count - 1)))  # Angular position
            normalized_angle = ray_angle / fov_angle  # Normalize angle to [-0.5, 0.5]

            # Project based on angular position, centering around the canvas
            proj_x = 300 + normalized_angle * half_canvas_width

            if prev_x is not None:
                if abs(distance - prev_distance) < gap_threshold and object_id == prev_object_id:
                    # Connect the previous point to the current point if no gap is detected
                    canvas.create_line(prev_x, prev_y, proj_x, 100, fill="blue")
                else:
                    # If a gap is detected, do not draw a connecting line
                    pass

            prev_x = proj_x
            prev_distance = distance
            prev_object_id = object_id  # Update the last object hit



# Function to handle mouse clicks on the canvas for selecting camera position and look direction
def on_canvas_click(event):
    global camera_position, look_at_position, click_count

    if click_count == 0:
        camera_position = (event.x, event.y)
        click_count += 1
    elif click_count == 1:
        look_at_position = (event.x, event.y)
        click_count = 0

        ray_intersections = draw_2d_perspective(canvas_2d, boxes, camera_position, look_at_position, fov_angle,
                                                ray_count)
        draw_1d_perspective_with_gaps(canvas_1d, ray_intersections, ray_count, fov_angle)

        # Call the 1D perspective printer
        print_1d_perspective(ray_intersections, ray_count, fov_angle)


# Tkinter setup
root = tk.Tk()
root.title("2D and 1D Perspective with Gap Detection")

# Create two canvases for 2D and 1D perspectives
canvas_2d = tk.Canvas(root, width=600, height=400, bg="white")
canvas_2d.grid(row=0, column=0, padx=20, pady=20)

canvas_1d = tk.Canvas(root, width=600, height=200, bg="white")
canvas_1d.grid(row=1, column=0, padx=20, pady=20)

# Define 3 boxes with different sizes and rotations
boxes = [
    create_box(center=(150, 150), size=100, angle=0),
    create_box(center=(300, 200), size=80, angle=45),
    create_box(center=(450, 100), size=60, angle=30)
]

# Initial setup
camera_position = (50, 200)  # Initial camera position
look_at_position = (150, 200)  # Initial look-at position
fov_angle = 60  # FOV in degrees
ray_count = 100  # Number of rays to cast
click_count = 0  # To track mouse clicks

# Draw the initial 2D and 1D perspectives
ray_intersections = draw_2d_perspective(canvas_2d, boxes, camera_position, look_at_position, fov_angle, ray_count)
draw_1d_perspective_with_gaps(canvas_1d, ray_intersections, ray_count, fov_angle)


# Bind mouse click to the 2D canvas
canvas_2d.bind("<Button-1>", on_canvas_click)

root.mainloop()
