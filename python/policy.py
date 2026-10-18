"""Policy interface for LaneShift controllers."""
from __future__ import annotations
from typing import Protocol, Dict, Any


class Policy(Protocol):
    def choose_action(self, state: Dict[str, Any]) -> str:
        ...


class BasePolicy:
    def choose_action(self, state):
        return "STAY"
