import json
import os
import numpy as np


class StatisticalEventHandler:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, "backend", "event_handlers", "event_impacts.json")

        with open(path, "r") as f:
            self.event_impacts = json.load(f)

    def apply(self, values, feature_cols, event_type=None):
        if event_type is None:
            return values

        if event_type not in self.event_impacts:
            return values

        impact_dict = self.event_impacts[event_type]

        # Convert dict → ordered vector
        impact_vector = np.array([
            impact_dict.get(col, 0) for col in feature_cols
        ])

        return values + impact_vector