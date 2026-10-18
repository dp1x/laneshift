"""Replaceable boundary for a future decision-model implementation."""

from policy import BasePolicy


class ModelAdapterPolicy(BasePolicy):
    label_source = "original_model"

    def __init__(self, predictor=None, label_source="original_model"):
        self.predictor = predictor
        self.label_source = label_source

    def choose_action(self, state):
        if self.predictor is None:
            return "STAY"
        action = self.predictor(state)
        if action not in {"LEFT", "STAY", "RIGHT", "BRAKE"}:
            return "STAY"
        return action
