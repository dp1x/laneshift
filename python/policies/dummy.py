"""Dummy scripted policy for bridge verification."""
from policy import BasePolicy


class DummyPolicy(BasePolicy):
    label_source = "synthetic"

    def __init__(self):
        self._counter = 0

    def choose_action(self, state):
        self._counter += 1
        # Make the first command visibly exercise the bridge, then return to center.
        if self._counter == 1:
            return "LEFT"
        if self._counter % 20 == 0:
            lane = state.get("player", {}).get("lane", 1)
            return "RIGHT" if lane == 0 else "LEFT"
        return "STAY"
