# utils/geometry.py

import math

def rotate_point(cx, cy, x, y, angle):
    """
    Rotate a point around a center by a given angle.

    :param cx: X-coordinate of the center.
    :param cy: Y-coordinate of the center.
    :param x: X-coordinate of the point.
    :param y: Y-coordinate of the point.
    :param angle: Rotation angle in degrees.
    :return: Tuple of rotated (x, y).
    """
    radians = math.radians(angle)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    # Translate point back to origin
    x -= cx
    y -= cy

    # Rotate point
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    # Translate point back
    x_rotated = x_new + cx
    y_rotated = y_new + cy

    return (x_rotated, y_rotated)

def create_rotated_rectangle(canvas, x, y, size, angle, fill_color='gray'):
    """
    Create a rotated rectangle on the canvas.

    :param canvas: Tkinter Canvas widget.
    :param x: Top-left X-coordinate.
    :param y: Top-left Y-coordinate.
    :param size: Size of the square obstacle.
    :param angle: Rotation angle in degrees.
    :param fill_color: Color to fill the rectangle.
    :return: Canvas object ID for the polygon.
    """
    # Calculate center of the rectangle
    cx = x + size / 2
    cy = y + size / 2

    # Define the four corners of the rectangle
    corners = [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size)
    ]

    # Rotate each corner
    rotated_corners = [rotate_point(cx, cy, px, py, angle) for px, py in corners]

    # Flatten the list of tuples
    points = [coord for point in rotated_corners for coord in point]

    # Create the polygon
    return canvas.create_polygon(points, fill=fill_color, outline='black')
