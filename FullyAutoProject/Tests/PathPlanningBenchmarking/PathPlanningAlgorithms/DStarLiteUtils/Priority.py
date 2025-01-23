# PathPlanningAlgorithms/DStarLiteUtils/Priority.py

class Priority:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __lt__(self, other):
        if self.k1 == other.k1:
            return self.k2 < other.k2
        return self.k1 < other.k1

    def __eq__(self, other):
        return self.k1 == other.k1 and self.k2 == other.k2
