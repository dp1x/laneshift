"""OpenRouter decisions-model policy (respan/span-01-lite:free).

Span-01-lite accepts only `noul` (yes/no) questions. A four-way action is
therefore split into independent yes/no questions for brake, left and right;
the highest probability above 0.5 wins, otherwise the rider stays.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import httpx

from kev_state import compact_kev_state
from policy import PolicyDecision

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "respan/span-01-lite:free"
DEFAULT_KEY_ENV = "DECISIONS_FREE_OPENROUTER_API_KEY"

QUESTIONS = {
    "BRAKE": (
        "brake",
        "The decision takes about 1.2 seconds to apply. Should the rider brake now? Brake early when a vehicle in the rider's lane is closing, especially when time_to_collision is under 3 seconds.",
        {"true": "Braking now prevents a collision or makes the situation safer.", "false": "Braking now is unnecessary or unsafe."},
    ),
    "LEFT": (
        "left",
        "The decision takes about 1.2 seconds to apply. Should the rider move one lane to the left now? Only say yes if the left lane is clear of vehicles near the rider.",
        {"true": "The left lane is safe and moving left helps now.", "false": "The left lane is unsafe or moving left does not help now."},
    ),
    "RIGHT": (
        "right",
        "The decision takes about 1.2 seconds to apply. Should the rider move one lane to the right now? Only say yes if the right lane is clear of vehicles near the rider.",
        {"true": "The right lane is safe and moving right helps now.", "false": "The right lane is unsafe or moving right does not help now."},
    ),
}


class OpenRouterDecisionPolicy:
    def __init__(self, client: httpx.AsyncClient, config: dict[str, Any]):
        self.client = client
        self.url = str(config.get("url") or DEFAULT_URL)
        self.model = str(config.get("model") or DEFAULT_MODEL)
        self.key_env = str(config.get("key_env") or DEFAULT_KEY_ENV)
        self.api_key = os.getenv(self.key_env, "")
        self.timeout = float(config.get("timeout", 15.0))
        self.label_source = "external_ai"
        self.policy_name = "openrouter_decisions"
        self.model_id = ""
        self.model_alias = self.model
        self.last_response: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.api_key:
            raise ValueError(f"{self.key_env} is not set")
        if not self.url.startswith("https://"):
            raise ValueError("OpenRouter decisions URL must use HTTPS")

    @staticmethod
    def _nearest(state: dict[str, Any], limit: int = 3) -> dict[str, Any]:
        compact = compact_kev_state(state)
        vehicles = sorted(
            compact["traffic"],
            key=lambda vehicle: float(vehicle.get("relative_distance") or 0.0),
        )[:limit]
        compact["traffic"] = vehicles
        return compact

    async def choose_action(self, state: dict[str, Any]) -> PolicyDecision:
        started = time.perf_counter()
        compact = self._nearest(state)
        legal = compact["legal_actions"]
        questions: dict[str, Any] = {}
        mapping: dict[str, str] = {}
        for action in ("BRAKE", "LEFT", "RIGHT"):
            if action not in legal:
                continue
            qid, instructions, criteria = QUESTIONS[action]
            questions[qid] = {"type": "noul", "instructions": instructions, "criteria": criteria}
            mapping[qid] = action
        payload = {
            "model": self.model,
            "state": json.dumps(compact, separators=(",", ":")),
            "questions": questions,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("decisions response is not an object")
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("decisions response is missing answers")
        probs: dict[str, float] = {}
        for qid, action in mapping.items():
            item = answers.get(qid)
            if not isinstance(item, dict) or item.get("type") != "noul":
                raise ValueError(f"decisions response is missing noul answer {qid!r}")
            value = item.get("noul")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"decisions answer {qid!r} is not a finite number")
            probs[action] = min(1.0, max(0.0, float(value)))
        best_action = "STAY"
        best_p = 0.0
        for action, p in probs.items():
            if p > best_p:
                best_action, best_p = action, p
        if best_p < 0.5:
            best_action = "STAY"
        action_probs = {action: probs.get(action, 0.0) for action in legal}
        if "STAY" in action_probs:
            action_probs["STAY"] = max(0.0, 1.0 - best_p)
        self.model_id = str(body.get("model") or self.model)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        self.last_response = body
        return PolicyDecision(
            action=best_action,
            label_source=self.label_source,
            policy_name=self.policy_name,
            model_id=self.model_id,
            model_alias=self.model_alias,
            probabilities=action_probs,
            confidence=best_p,
            model_latency_ms=float(usage.get("latency_ms", 0.0) or 0.0),
            client_latency_ms=(time.perf_counter() - started) * 1000.0,
            request_payload=payload,
            response_payload=body,
        )
