"""Localhost-only WebSocket bridge between LaneShift and a Python policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import websockets

from policies.dummy import DummyPolicy
from policies.model_adapter import ModelAdapterPolicy
from policies.rule_based import RuleBasedPolicy

PROTOCOL_VERSION = 1
ALLOWED_ACTIONS = frozenset({"LEFT", "STAY", "RIGHT", "BRAKE"})
POLICY_SOURCES = {
    "dummy": "synthetic",
    "rule_based": "rule_based",
    "model": "original_model",
    "fine_tuned_model": "fine_tuned_model",
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
    if not isinstance(player, dict) or not isinstance(traffic, list):
        return False
    lane = player.get("lane")
    speed = player.get("speed")
    if not isinstance(lane, int) or lane not in (0, 1, 2):
        return False
    if not isinstance(speed, (int, float)) or not math.isfinite(speed):
        return False
    return True


def make_action_message(state, seq, action, duration=0.35, label_source="external_ai"):
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {action!r}")
    return {
        "type": "action",
        "protocol_version": PROTOCOL_VERSION,
        "episode_id": state["episode_id"],
        "seq": int(seq),
        "tick": int(state["physics_tick"]),
        "action": action,
        "duration": max(0.05, min(float(duration), 5.0)),
        "label_source": label_source,
    }


def load_policy(name):
    policies = {
        "dummy": DummyPolicy,
        "rule_based": RuleBasedPolicy,
        "model": ModelAdapterPolicy,
        "fine_tuned_model": lambda: ModelAdapterPolicy(label_source="fine_tuned_model"),
    }
    try:
        return policies[name]()
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {', '.join(policies)}") from exc


class ControllerServer:
    def __init__(self, policy_name="rule_based", duration=0.35, expected_episodes=0):
        self.policy_name = policy_name
        self.policy = load_policy(policy_name)
        self.duration = duration
        self.expected_episodes = expected_episodes
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
        self.latencies_ms = []
        self.ack_rtts_ms = []
        self.pending_acks = {}
        self.last_episode_id = ""

    async def handle_connection(self, websocket):
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self.invalid_states += 1
                    continue
                kind = message.get("type") if isinstance(message, dict) else None
                if kind == "state":
                    await self._handle_state(websocket, message)
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

    async def _handle_state(self, websocket, state):
        if not validate_state_message(state):
            self.invalid_states += 1
            return
        started = time.perf_counter()
        try:
            action = self.policy.choose_action(state)
        except Exception:
            self.controller_failures += 1
            action = "STAY"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latencies_ms.append(elapsed_ms)
        if action not in ALLOWED_ACTIONS:
            self.invalid_actions += 1
            action = "STAY"
        self.sequence += 1
        label_source = getattr(self.policy, "label_source", POLICY_SOURCES.get(self.policy_name, "external_ai"))
        outgoing = make_action_message(state, self.sequence, action, self.duration, label_source)
        self.actions[action] += 1
        self.decisions += 1
        self.last_episode_id = state["episode_id"]
        self.pending_acks[self.sequence] = time.perf_counter()
        await websocket.send(json.dumps(outgoing, separators=(",", ":")))

    def _handle_event(self, event):
        if not isinstance(event, dict):
            return
        if event.get("type") == "action_decision":
            if event.get("accepted") is True:
                self.applied_actions[str(event.get("applied_action", "STAY"))] += 1
            elif event.get("controller_seq", -1) >= 0:
                self.rejected_actions += 1
        elif event.get("type") in {"external_action_ignored", "external_action_rejected"} and event.get("controller_seq", -1) >= 0:
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
        return {
            "policy": self.policy_name,
            "decisions": self.decisions,
            "actions": dict(self.actions),
            "applied_actions": dict(self.applied_actions),
            "rejected_actions": self.rejected_actions,
            "invalid_states": self.invalid_states,
            "invalid_actions": self.invalid_actions,
            "protocol_rejections": self.protocol_rejections,
            "controller_failures": self.controller_failures,
            "completed_episodes": self.completed_episodes,
            "decision_latency_ms_mean": statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0,
            "decision_latency_ms_p50": self._percentile(self.latencies_ms, 0.50),
            "decision_latency_ms_p95": self._percentile(self.latencies_ms, 0.95),
            "decision_latency_ms_p99": self._percentile(self.latencies_ms, 0.99),
            "ack_rtt_ms_mean": statistics.fmean(self.ack_rtts_ms) if self.ack_rtts_ms else 0.0,
        }


async def run_server(args):
    server = ControllerServer(args.policy, args.duration, args.expected_episodes)
    async with websockets.serve(
        server.handle_connection,
        "127.0.0.1",
        args.port,
        ping_interval=20,
        ping_timeout=20,
        max_size=2**20,
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
    parser.add_argument("--policy", choices=("dummy", "rule_based", "model", "fine_tuned_model"), default="rule_based")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=float, default=0.35)
    parser.add_argument("--expected-episodes", type=int, default=0)
    parser.add_argument("--metrics-out")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(run_server(parse_args()))
    except KeyboardInterrupt:
        pass
