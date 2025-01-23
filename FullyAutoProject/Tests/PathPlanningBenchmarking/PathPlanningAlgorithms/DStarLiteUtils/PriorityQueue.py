# PathPlanningAlgorithms/DStarLiteUtils/PriorityQueue.py

import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # Map from item to entry
        self.REMOVED = '<removed-task>'

    def insert(self, item, priority):
        """Add a new item or update the priority of an existing item."""
        if item in self.entry_finder:
            self.remove(item)
        entry = [priority, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def remove(self, item):
        """Mark an existing item as removed."""
        entry = self.entry_finder.pop(item)
        entry[1] = self.REMOVED

    def pop(self):
        """Remove and return the lowest priority item."""
        while self.heap:
            priority, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def top(self):
        """Return the lowest priority item without removing it."""
        while self.heap:
            priority, item = self.heap[0]
            if item is self.REMOVED:
                heapq.heappop(self.heap)
                continue
            return item
        raise KeyError('top from an empty priority queue')

    def top_key(self):
        """Return the priority of the lowest priority item."""
        while self.heap:
            priority, item = self.heap[0]
            if item is self.REMOVED:
                heapq.heappop(self.heap)
                continue
            return priority
        return None

    def update(self, item, new_priority):
        """Update the priority of an existing item."""
        self.insert(item, new_priority)

    def __contains__(self, item):
        return item in self.entry_finder
