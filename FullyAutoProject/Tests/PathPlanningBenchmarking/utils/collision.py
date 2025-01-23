# utils/collision.py
import math


def rectangles_overlap(rect1, rect2):
    """
    Determine if two rotated rectangles overlap using SAT.

    :param rect1: List of four (x, y) tuples representing the first rectangle's corners.
    :param rect2: List of four (x, y) tuples representing the second rectangle's corners.
    :return: True if rectangles overlap, False otherwise.
    """

    def get_axes(rect):
        axes = []
        for i in range(len(rect)):
            p1 = rect[i]
            p2 = rect[(i + 1) % len(rect)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            length = math.hypot(*normal)
            axes.append((normal[0] / length, normal[1] / length))
        return axes

    def project(rect, axis):
        min_proj = float('inf')
        max_proj = float('-inf')
        for point in rect:
            projection = point[0] * axis[0] + point[1] * axis[1]
            min_proj = min(min_proj, projection)
            max_proj = max(max_proj, projection)
        return min_proj, max_proj

    axes1 = get_axes(rect1)
    axes2 = get_axes(rect2)
    axes = axes1 + axes2

    for axis in axes:
        min1, max1 = project(rect1, axis)
        min2, max2 = project(rect2, axis)
        if max1 < min2 or max2 < min1:
            return False  # No overlap on this axis
    return True  # Overlap on all axes
