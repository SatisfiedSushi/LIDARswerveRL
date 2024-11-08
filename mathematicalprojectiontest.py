import tkinter as tk
import math

# Function to compute 2D to 1D perspective projection
def perspective_2d_to_1d(point, camera_pos, look_direction, fov):
    x, y = point
    cam_x, cam_y = camera_pos
    look_x, look_y = look_direction

    # Calculate the angle of the point relative to the camera's position
    angle_to_point = math.atan2(y - cam_y, x - cam_x)

    # Calculate the angle of the look direction vector
    angle_of_look_direction = math.atan2(look_y - cam_y, look_x - cam_x)

    # Check if the point is within the field of view (FOV)
    half_fov = fov / 2
    angle_diff = angle_to_point - angle_of_look_direction

    # Normalize angle difference to range [-π, π]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    if -half_fov <= angle_diff <= half_fov:
        # If the point is within the FOV, compute the perspective projection onto the 1D line
        distance = math.hypot(x - cam_x, y - cam_y)
        projected_x = distance * math.cos(angle_diff)
        return projected_x
    return None  # Outside FOV

# Draw 2D and 1D projections
def draw_projection(canvas, points, camera_pos, look_direction, fov):
    canvas.delete("all")

    # Draw the 1D projection line
    canvas.create_line(50, 200, 550, 200, fill='gray', dash=(4, 2))  # Line at y = 200 representing the 1D projection line

    # Draw the camera position
    canvas.create_oval(camera_pos[0] - 5, camera_pos[1] - 5, camera_pos[0] + 5, camera_pos[1] + 5, fill="red")
    canvas.create_text(camera_pos[0], camera_pos[1], text="Cam", anchor="nw", fill="red")

    # Draw the look direction
    canvas.create_line(camera_pos[0], camera_pos[1], look_direction[0], look_direction[1], fill="blue", dash=(5, 5))
    canvas.create_text(look_direction[0], look_direction[1], text="Look", anchor="nw", fill="blue")

    for point in points:
        x, y = point
        projected_x = perspective_2d_to_1d(point, camera_pos, look_direction, fov)

        # Draw original 2D point
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")
        canvas.create_text(x, y, text=f"({x}, {y})", anchor="nw", fill="red")

        # Draw 1D projection if within FOV
        if projected_x is not None:
            # Scale the projection for visualization purposes
            canvas.create_oval(300 + projected_x * 10 - 3, 197, 300 + projected_x * 10 + 3, 203, fill="blue")
            canvas.create_text(300 + projected_x * 10, 210, text=f"{projected_x:.2f}", anchor="nw", fill="blue")

# Tkinter setup
root = tk.Tk()
root.title("2D to 1D Perspective Projection with FOV")

canvas = tk.Canvas(root, width=600, height=300, bg="white")
canvas.pack(padx=20, pady=20)

# Example 2D points
points = [(100, 150), (200, 100), (300, 200), (400, 250), (500, 100)]

# Camera position and look direction
camera_pos = (300, 150)  # Camera is at (300, 150)
look_direction = (300, -50)  # Camera is looking straight up

# Field of view (in radians)
fov = math.radians(90)  # 90 degree field of view

# Draw the initial projection
draw_projection(canvas, points, camera_pos, look_direction, fov)

root.mainloop()
