import numpy as np
from .base import EventHandler


class RuleBasedEventHandler(EventHandler):

    def apply(self, values, feature_cols, event_type=None):

        if event_type == "festival":
            factor = 1.25
        elif event_type == "traffic":
            factor = 1.15
        elif event_type == "industrial":
            factor = 1.20
        else:
            factor = 1.0

        return values * factor