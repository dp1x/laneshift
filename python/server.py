"""Localhost-only WebSocket bridge between LaneShift and a Python policy."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import websockets
import httpx

from policy import PolicyDecision
from policies.dummy import DummyPolicy
from policies.model_adapter import KevSystemOnePolicy
from policies.rule_based import RuleBasedPolicy

PROTOCOL_VERSION = 1
ALLOWED_ACTIONS = frozenset({"LEFT", "STAY", "RIGHT", "BRAKE"})
POLICY_SOURCES = {
    "dummy": "synthetic",
    "rule_based": "rule_based",
}


def validate_state_message(message):
    """Return whether a message contains a minimally usable decision state."""
    if not isinstance(message, dict):
        return False
    if message.get("type") != "state" or message.get("protocol_version") != PROTOCOL_VERSION:
        return False
    if not isinstance(message.get("episode_id"), str) or not message["episode_id"]:
        return False
    tick = message.get("physics_tick")
    if not isinstance(tick, int) or tick < 0:
        return False
    player = message.get("player")
    traffic = message.get("traffic")
    legal_actions = message.get("legal_actions")
    if not isinstance(player, dict) or not isinstance(traffic, list):
        return False
    if (
        not isinstance(legal_actions, list)
        or "STAY" not in legal_actions
        or any(not isinstance(action, str) or action not in ALLOWED_ACTIONS for action in legal_actions)
        or len(set(legal_actions)) != len(legal_actions)
    ):
        return False
    lane = player.get("lane")
    speed = player.get("speed")
    if not isinstance(lane, int) or lane not in (0, 1, 2):
        return False
    if not isinstance(speed, (int, float)) or not math.isfinite(speed):
        return False
    return True


def make_action_message(state, seq, action, duration=0.35, label_source="external_ai", decision_metadata=None):
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {action!r}")
    if action not in state.get("legal_actions", []):
        raise ValueError(f"action is not legal in its decision state: {action!r}")
    return {
        "type": "action",
        "protocol_version": PROTOCOL_VERSION,
        "episode_id": state["episode_id"],
        "seq": int(seq),
        "tick": int(state["physics_tick"]),
        "action": action,
        "duration": max(0.05, min(float(duration), 5.0)),
        "label_source": label_source,
        "decision_metadata": decision_metadata or {},
    }


def load_policy(name, model_configs=None, http_client=None):
    if name == "dummy":
        return DummyPolicy()
    if name == "rule_based":
        return RuleBasedPolicy()
    config = (model_configs or {}).get(name)
    if config is None:
        raise ValueError(
            f"unknown policy {name!r}; model policies must be explicitly configured and identity-verified"
        )
    if http_client is None:
        raise ValueError("a Kev policy requires an HTTP client")
    return KevSystemOnePolicy(http_client, config)


class ControllerServer:
    def __init__(
        self,
        policy_name="rule_based",
        duration=0.35,
        expected_episodes=0,
        model_configs=None,
        http_client=None,
        decision_timeout=0.4,
    ):
        self.policy_name = policy_name
        self.policy = load_policy(policy_name, model_configs, http_client)
        self.duration = duration
        self.expected_episodes = expected_episodes
        self.decision_timeout = decision_timeout
        self.max_state_age_s = 0.4
        self.shutdown_event = asyncio.Event()
        self.sequence = 0
        self.decisions = 0
        self.invalid_states = 0
        self.invalid_actions = 0
        self.controller_failures = 0
        self.protocol_rejections = 0
        self.completed_episodes = 0
        self.actions = Counter()
        self.applied_actions = Counter()
        self.rejected_actions = 0
        self.action_outcomes_seen = set()
        self.latencies_ms = []
        self.model_latencies_ms = []
        self.ack_rtts_ms = []
        self.pending_acks = {}
        self.last_episode_id = ""
        self.superseded_states = 0
        self.stale_decisions = 0

    async def start(self):
        start = getattr(self.policy, "start", None)
        if start is not None:
            await start()

    async def handle_connection(self, websocket):
        state_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        latest_state: dict[str, Any] = {"state": None}
        decision_task = asyncio.create_task(self._decision_loop(websocket, state_queue, latest_state))
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self.invalid_states += 1
                    await websocket.send(json.dumps({
                        "type": "reject", "protocol_version": PROTOCOL_VERSION,
                        "seq": -1, "reason": "malformed_json"
                    }))
                    continue
                kind = message.get("type") if isinstance(message, dict) else None
                if kind == "state":
                    if not validate_state_message(message):
                        self.invalid_states += 1
                        await websocket.send(json.dumps({
                            "type": "reject", "protocol_version": PROTOCOL_VERSION,
                            "seq": -1, "reason": "invalid_state"
                        }))
                        continue
                    latest_state["state"] = message
                    if state_queue.full():
                        try:
                            state_queue.get_nowait()
                            state_queue.task_done()
                            self.superseded_states += 1
                        except asyncio.QueueEmpty:
                            pass
                    state_queue.put_nowait(message)
                elif kind == "event":
                    self._handle_event(message.get("event", {}))
                elif kind == "ack":
                    sent = self.pending_acks.pop(message.get("seq"), None)
                    if sent is not None:
                        self.ack_rtts_ms.append((time.perf_counter() - sent) * 1000.0)
                elif kind == "reject":
                    self.protocol_rejections += 1
                    self.pending_acks.pop(message.get("seq"), None)
        except websockets.ConnectionClosed:
            pass
        finally:
            decision_task.cancel()
            try:
                await decision_task
            except asyncio.CancelledError:
                pass

    async def _decision_loop(self, websocket, state_queue: asyncio.Queue, latest_state: dict[str, Any]):
        while True:
            state = await state_queue.get()
            try:
                await self._handle_state(websocket, state, latest_state)
            except websockets.ConnectionClosed:
                return
            finally:
                state_queue.task_done()

    async def _choose(self, state):
        chooser = self.policy.choose_action
        if inspect.iscoroutinefunction(chooser):
            return await asyncio.wait_for(chooser(state), timeout=self.decision_timeout)
        return await asyncio.wait_for(asyncio.to_thread(chooser, state), timeout=self.decision_timeout)

    async def _handle_state(self, websocket, state, latest_state=None):
        if not validate_state_message(state):
            self.invalid_states += 1
            return
        started = time.perf_counter()
        fallback_reason = None
        decision = None
        error_details = {}
        try:
            raw_decision = await self._choose(state)
            if isinstance(raw_decision, PolicyDecision):
                decision = raw_decision
            else:
                decision = PolicyDecision(
                    action=str(raw_decision),
                    label_source=getattr(self.policy, "label_source", POLICY_SOURCES.get(self.policy_name, "external_ai")),
                    policy_name=self.policy_name,
                )
        except asyncio.TimeoutError:
            self.controller_failures += 1
            fallback_reason = "decision_timeout"
        except Exception as exc:
            self.controller_failures += 1
            fallback_reason = f"inference_error:{type(exc).__name__}:{exc}"
            error_details = getattr(exc, "details", {})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latencies_ms.append(elapsed_ms)

        if latest_state is not None:
            current = latest_state.get("state")
            if not isinstance(current, dict) or current.get("episode_id") != state.get("episode_id"):
                self.stale_decisions += 1
                return
            age = float(current.get("sim_time", 0.0)) - float(state.get("sim_time", 0.0))
            if age < 0.0 or age > self.max_state_age_s:
                self.stale_decisions += 1
                return

        legal_actions = state["legal_actions"]
        if decision is None:
            decision = PolicyDecision(
                action="STAY" if "STAY" in legal_actions else legal_actions[0],
                label_source="controller_fallback",
                policy_name=self.policy_name,
                model_id=error_details.get("model_id", getattr(self.policy, "actual_run", None)),
                model_alias=error_details.get("model_alias", getattr(self.policy, "model_alias", None)),
                probabilities=error_details.get("probabilities"),
                request_started_unix_ms=error_details.get("request_started_unix_ms"),
                response_received_unix_ms=error_details.get("response_received_unix_ms"),
                request_id=error_details.get("request_id"),
                request_payload=error_details.get("request_payload"),
                response_payload=error_details.get("response_payload"),
                fallback_reason=fallback_reason,
                client_latency_ms=error_details.get("client_latency_ms", elapsed_ms),
            )
        elif decision.action not in ALLOWED_ACTIONS or decision.action not in legal_actions:
            self.invalid_actions += 1
            fallback_reason = "action_not_legal_for_decision_state"
            decision = PolicyDecision(
                action="STAY" if "STAY" in legal_actions else legal_actions[0],
                label_source="controller_fallback",
                policy_name=self.policy_name,
                model_id=decision.model_id,
                model_alias=decision.model_alias,
                probabilities=decision.probabilities,
                confidence=decision.confidence,
                model_latency_ms=decision.model_latency_ms,
                client_latency_ms=decision.client_latency_ms or elapsed_ms,
                request_started_unix_ms=decision.request_started_unix_ms,
                response_received_unix_ms=decision.response_received_unix_ms,
                request_id=decision.request_id,
                request_payload=decision.request_payload,
                response_payload=decision.response_payload,
                fallback_reason=fallback_reason,
            )

        self.sequence += 1
        metadata = decision.to_metadata()
        metadata.update({
            "decision_state_id": state.get("state_id"),
            "decision_tick": state.get("physics_tick"),
            "controller_decision_latency_ms": round(elapsed_ms, 2),
        })
        self.actions[decision.action] += 1
        self.decisions += 1
        self.last_episode_id = state["episode_id"]
        if decision.model_latency_ms is not None:
            self.model_latencies_ms.append(float(decision.model_latency_ms))
        self.pending_acks[self.sequence] = time.perf_counter()
        outgoing = make_action_message(
            state,
            self.sequence,
            decision.action,
            self.duration,
            decision.label_source,
            metadata,
        )
        await websocket.send(json.dumps(outgoing, separators=(",", ":")))

    def _handle_event(self, event):
        if not isinstance(event, dict):
            return
        if event.get("type") == "action_decision":
            seq = event.get("controller_seq", -1)
            key = (event.get("episode_id", ""), seq)
            if seq >= 0 and key in self.action_outcomes_seen:
                return
            if seq >= 0:
                self.action_outcomes_seen.add(key)
            if event.get("accepted") is True:
                self.applied_actions[str(event.get("applied_action", "STAY"))] += 1
            elif seq >= 0:
                self.rejected_actions += 1
        elif event.get("type") in {"external_action_ignored", "external_action_rejected"} and event.get("controller_seq", -1) >= 0:
            seq = event["controller_seq"]
            key = (event.get("episode_id", ""), seq)
            if key not in self.action_outcomes_seen:
                self.action_outcomes_seen.add(key)
                self.rejected_actions += 1
        if event.get("type") == "episode_end":
            self.completed_episodes += 1
            if self.expected_episodes and self.completed_episodes >= self.expected_episodes:
                self.shutdown_event.set()

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    def metrics(self):
        model_id = getattr(self.policy, "actual_run", "")
        model_alias = getattr(self.policy, "model_alias", "")
        policy_kind = "model" if hasattr(self.policy, "actual_run") else "baseline"
        return {
            "policy": self.policy_name,
            "policy_kind": policy_kind,
            "model_id": model_id,
            "model_alias": model_alias,
            "decisions": self.decisions,
            "actions": dict(self.actions),
            "applied_actions": dict(self.applied_actions),
            "rejected_actions": self.rejected_actions,
            "invalid_states": self.invalid_states,
            "invalid_actions": self.invalid_actions,
            "protocol_rejections": self.protocol_rejections,
            "controller_failures": self.controller_failures,
            "completed_episodes": self.completed_episodes,
            "superseded_states": self.superseded_states,
            "stale_decisions": self.stale_decisions,
            "decision_latency_ms_mean": statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0,
            "decision_latency_ms_p50": self._percentile(self.latencies_ms, 0.50),
            "decision_latency_ms_p95": self._percentile(self.latencies_ms, 0.95),
            "decision_latency_ms_p99": self._percentile(self.latencies_ms, 0.99),
            "model_latency_ms_mean": statistics.fmean(self.model_latencies_ms) if self.model_latencies_ms else 0.0,
            "model_latency_ms_p50": self._percentile(self.model_latencies_ms, 0.50),
            "model_latency_ms_p95": self._percentile(self.model_latencies_ms, 0.95),
            "ack_rtt_ms_mean": statistics.fmean(self.ack_rtts_ms) if self.ack_rtts_ms else 0.0,
        }


async def run_server(args):
    model_configs = {}
    if args.kev_config:
        config_path = Path(args.kev_config)
        try:
            model_configs = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Kev model registry {config_path}: {exc}") from exc
        if not isinstance(model_configs, dict):
            raise ValueError("Kev model registry must be a JSON object keyed by controller policy name")
        if "models" in model_configs and isinstance(model_configs["models"], dict):
            model_configs = model_configs["models"]
    if args.policy == "kev" and "kev" not in model_configs:
        if not args.kev_expected_run or not args.kev_label_source:
            raise ValueError("--policy kev requires --kev-expected-run and --kev-label-source, or --kev-config")
        model_configs["kev"] = {
            "base_url": args.kev_base_url,
            "model": args.kev_model,
            "expected_run": args.kev_expected_run,
            "label_source": args.kev_label_source,
            "timeout": args.inference_timeout,
            "api_key_env": args.kev_api_key_env,
        }
    async with httpx.AsyncClient(trust_env=False) as http_client:
        server = ControllerServer(
            args.policy,
            args.duration,
            args.expected_episodes,
            model_configs=model_configs,
            http_client=http_client,
            decision_timeout=args.decision_timeout,
        )
        await server.start()
        async with websockets.serve(
            server.handle_connection,
            "127.0.0.1",
            args.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
            max_queue=2,
        ):
            print(f"LaneShift controller listening at ws://127.0.0.1:{args.port} ({args.policy})", flush=True)
            if args.expected_episodes:
                await server.shutdown_event.wait()
    result = server.metrics()
    print("LANESHIFT_METRICS " + json.dumps(result, separators=(",", ":")), flush=True)
    if args.metrics_out:
        output = Path(args.metrics_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="rule_based", help="dummy, rule_based, or a named model-registry entry")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=float, default=0.35)
    parser.add_argument("--decision-timeout", type=float, default=0.45)
    parser.add_argument("--expected-episodes", type=int, default=0)
    parser.add_argument("--metrics-out")
    parser.add_argument("--kev-config", help="JSON model registry keyed by policy name")
    parser.add_argument("--kev-base-url", default=os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009"))
    parser.add_argument("--kev-model", default=os.getenv("KEV_MODEL", "kev-latest"))
    parser.add_argument("--kev-expected-run")
    parser.add_argument("--kev-label-source", choices=("original_model", "fine_tuned_model"))
    parser.add_argument("--kev-api-key-env", default="KEV_API_KEY")
    parser.add_argument("--inference-timeout", type=float, default=0.4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(run_server(parse_args()))
    except KeyboardInterrupt:
        pass
