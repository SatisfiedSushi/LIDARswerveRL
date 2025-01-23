# utils/Vertices.py

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return False  # Necessary for priority queues
