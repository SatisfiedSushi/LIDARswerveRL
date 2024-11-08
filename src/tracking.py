# src/tracking.py

import numpy as np
from collections import defaultdict

class Track:
    def __init__(self, id, initial_position):
        self.id = id
        self.position = np.array(initial_position)
        self.history = [self.position.copy()]
        self.age = 1
        self.hits = 1
        self.missed = 0

    def update(self, new_position):
        self.position = np.array(new_position)
        self.history.append(self.position.copy())
        self.age += 1
        self.hits += 1
        self.missed = 0

    def predict(self, dt):
        # Simple prediction: assume constant velocity
        if len(self.history) >= 2:
            velocity = self.history[-1] - self.history[-2]
            self.position += velocity  # dt is assumed to be 1 for simplicity
        self.history.append(self.position.copy())
        self.age += 1
        self.missed += 1

class MHTTracker:
    def __init__(self, max_missed=5):
        self.tracks = {}
        self.max_missed = max_missed

    def add_or_update_track(self, object_id, position):
        if object_id in self.tracks:
            self.tracks[object_id].update(position)
        else:
            self.tracks[object_id] = Track(object_id, position)

    def predict_tracks(self, dt):
        for track in self.tracks.values():
            track.predict(dt)

    def remove_stale_tracks(self):
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.missed > self.max_missed:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]

    def update(self, detections, object_ids, dt=0.1):
        """
        Update tracks with new detections.
        detections: list of [x, y]
        object_ids: list of corresponding object IDs
        dt: time step
        """
        # Update existing tracks or create new ones
        for det, obj_id in zip(detections, object_ids):
            self.add_or_update_track(obj_id, det)

        # Predict for all tracks
        self.predict_tracks(dt)

        # Remove tracks that have missed too many times
        self.remove_stale_tracks()

    def get_active_tracks(self):
        return list(self.tracks.values())
