# utils/heuristic.py

def heuristic(a, b):
    """
    Compute the Euclidean distance between two points a and b.

    :param a: Tuple (x1, y1)
    :param b: Tuple (x2, y2)
    :return: Euclidean distance (float)
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
