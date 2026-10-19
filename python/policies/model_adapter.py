"""Async Kev System One policy adapter; no model behavior is simulated locally."""

from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from policy import PolicyDecision

ACTION_CRITERIA = {
    "LEFT": "Move one lane to the left.",
    "STAY": "Remain in the current lane.",
    "RIGHT": "Move one lane to the right.",
    "BRAKE": "Reduce speed using the brake.",
}
LABEL_SOURCES = {"original_model", "fine_tuned_model"}
STATE_FIELDS = (
    "protocol_version",
    "episode_id",
    "scenario_seed",
    "sim_time",
    "physics_tick",
    "state_id",
    "player",
    "traffic",
    "legal_actions",
    "score",
    "distance",
    "survival_time",
    "collision",
    "collision_count",
    "episode_status",
    "control_source",
)


class KevInferenceError(RuntimeError):
    """A model call or its typed response could not produce a safe decision."""

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


class KevIdentityError(KevInferenceError):
    """The running Kev service is not serving the configured checkpoint."""


class KevResponseError(KevInferenceError):
    """The System One response is malformed or selects an invalid action."""


class KevSystemOnePolicy:
    def __init__(self, client: httpx.AsyncClient, config: dict[str, Any]):
        self.client = client
        self.base_url = str(config.get("base_url") or os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009")).rstrip("/")
        self.model_alias = str(config.get("model") or os.getenv("KEV_MODEL", "kev-latest"))
        self.expected_run = str(config.get("expected_run", "")).strip()
        self.label_source = str(config.get("label_source", ""))
        self.timeout = float(config.get("timeout", os.getenv("KEV_INFERENCE_TIMEOUT", "0.4")))
        self.api_key_env = str(config.get("api_key_env", "KEV_API_KEY"))
        self.api_key = os.getenv(self.api_key_env)
        if not self.expected_run:
            raise ValueError("Kev policy requires expected_run so the loaded checkpoint can be verified")
        if self.label_source not in LABEL_SOURCES:
            raise ValueError("Kev policy requires explicit label_source: original_model or fine_tuned_model")
        if self.timeout <= 0.0 or not math.isfinite(self.timeout):
            raise ValueError("Kev inference timeout must be a positive finite number")
        self.actual_run = ""
        self.model_metadata: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def start(self) -> None:
        """Verify the service alias resolves to the checkpoint declared by the run config."""
        try:
            response = await self.client.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise KevIdentityError(f"could not inspect Kev model service at {self.base_url}: {exc}") from exc
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            raise KevIdentityError("Kev /v1/models response has no models list")
        matching = [item for item in models if isinstance(item, dict) and item.get("name") == self.model_alias]
        if not matching:
            raise KevIdentityError(f"Kev service does not advertise model alias {self.model_alias!r}")
        metadata = matching[0]
        actual_run = metadata.get("run")
        if not isinstance(actual_run, str) or actual_run != self.expected_run:
            raise KevIdentityError(
                f"Kev alias {self.model_alias!r} resolves to {actual_run!r}, expected {self.expected_run!r}"
            )
        self.actual_run = actual_run
        self.model_metadata = metadata

    @staticmethod
    def build_request(state: dict[str, Any], model_alias: str) -> dict[str, Any]:
        legal_actions = state.get("legal_actions")
        if (
            not isinstance(legal_actions, list)
            or not legal_actions
            or "STAY" not in legal_actions
            or any(action not in ACTION_CRITERIA for action in legal_actions)
            or len(set(legal_actions)) != len(legal_actions)
        ):
            raise KevResponseError("decision state has no valid authoritative legal_actions set")
        if not isinstance(state.get("player"), dict) or not isinstance(state.get("traffic"), list):
            raise KevResponseError("decision state is missing structured player or traffic data")
        return {
            "state": {key: state[key] for key in STATE_FIELDS if key in state},
            "model": model_alias,
            "questions": {
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Choose the safest useful next action from the currently legal choices. "
                        "Only select an option present in criteria."
                    ),
                    "criteria": {action: ACTION_CRITERIA[action] for action in legal_actions},
                }
            },
        }

    async def choose_action(self, state: dict[str, Any]) -> PolicyDecision:
        if not self.actual_run:
            raise KevIdentityError("KevSystemOnePolicy.start() must verify the loaded model before inference")
        request_payload = self.build_request(state, self.model_alias)
        started_unix_ms = time.time() * 1000.0
        started = time.perf_counter()
        response: httpx.Response | None = None
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/systemone",
                json=request_payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raw_response = None
            request_id = None
            if response is not None:
                raw_response = {"status_code": response.status_code, "body": response.text}
                request_id = response.headers.get("x-typesafe-request-id")
            raise KevInferenceError(
                f"Kev System One request failed: {exc}",
                request_payload=request_payload,
                response_payload=raw_response,
                model_id=self.actual_run,
                model_alias=self.model_alias,
                request_started_unix_ms=started_unix_ms,
                response_received_unix_ms=time.time() * 1000.0,
                client_latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                request_id=request_id,
            ) from exc
        completed_unix_ms = time.time() * 1000.0
        client_latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        legal_actions = request_payload["state"]["legal_actions"]
        if not isinstance(body, dict) or body.get("model") != self.model_alias:
            raise self._response_error("Kev response is missing the requested model alias", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        answers = body.get("answers")
        answer = answers.get("next_action") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise self._response_error("Kev response is missing answers.next_action choice", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        action = answer.get("choice")
        if not isinstance(action, str) or action not in legal_actions:
            raise self._response_error(f"Kev selected illegal or unknown choice {action!r}", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        probabilities_raw = answer.get("probabilities")
        if not isinstance(probabilities_raw, dict) or set(probabilities_raw) != set(legal_actions):
            raise self._response_error("Kev choice probabilities must cover exactly the offered legal actions", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        probabilities: dict[str, float] = {}
        for choice, value in probabilities_raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise self._response_error(f"probability for {choice!r} is not numeric", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
            value = float(value)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise self._response_error(f"probability for {choice!r} is outside [0, 1]", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
            probabilities[choice] = value
        if abs(sum(probabilities.values()) - 1.0) > 0.02:
            raise self._response_error("Kev choice probabilities do not sum to approximately 1", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        model_latency = body.get("latency_ms")
        if isinstance(model_latency, bool) or not isinstance(model_latency, (int, float)) or not math.isfinite(float(model_latency)):
            raise self._response_error("Kev response is missing finite top-level latency_ms", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
        confidence = answer.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                raise self._response_error("Kev choice confidence is outside [0, 1]", request_payload, body, started_unix_ms, completed_unix_ms, client_latency_ms, response)
            confidence = float(confidence)
        return PolicyDecision(
            action=action,
            label_source=self.label_source,
            policy_name="kev_system_one",
            model_id=self.actual_run,
            model_alias=self.model_alias,
            probabilities=probabilities,
            confidence=confidence,
            model_latency_ms=float(model_latency),
            client_latency_ms=client_latency_ms,
            request_started_unix_ms=started_unix_ms,
            response_received_unix_ms=completed_unix_ms,
            request_id=response.headers.get("x-typesafe-request-id"),
            request_payload=request_payload,
            response_payload=body,
        )

    def _response_error(self, message, request_payload, response_payload, started_unix_ms, completed_unix_ms, client_latency_ms, response):
        return KevResponseError(
            message,
            request_payload=request_payload,
            response_payload=response_payload,
            model_id=self.actual_run,
            model_alias=self.model_alias,
            request_started_unix_ms=started_unix_ms,
            response_received_unix_ms=completed_unix_ms,
            client_latency_ms=client_latency_ms,
            request_id=response.headers.get("x-typesafe-request-id"),
        )
