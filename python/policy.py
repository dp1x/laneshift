"""Policy interface for LaneShift controllers."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Protocol, Dict, Any


@dataclass
class PolicyDecision:
    action: str
    label_source: str
    policy_name: str
    model_id: str | None = None
    model_alias: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    model_latency_ms: float | None = None
    client_latency_ms: float | None = None
    request_started_unix_ms: float | None = None
    response_received_unix_ms: float | None = None
    request_id: str | None = None
    request_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] | None = None
    fallback_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class Policy(Protocol):
    def choose_action(self, state: Dict[str, Any]) -> str | PolicyDecision:
        ...


class BasePolicy:
    def choose_action(self, state):
        return "STAY"

    @staticmethod
    def legal_fallback(state):
        legal = state.get("legal_actions", ["LEFT", "STAY", "RIGHT", "BRAKE"])
        return "STAY" if "STAY" in legal else legal[0]
