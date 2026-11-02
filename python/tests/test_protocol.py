import unittest
import asyncio
import json
import tempfile
import threading
from pathlib import Path

from policy import PolicyDecision
from server import ALLOWED_ACTIONS, ControllerServer, load_policy, make_action_message, parse_args, validate_state_message


def sample_state():
    return {
        "type": "state",
        "protocol_version": 1,
        "episode_id": "ep-test",
        "physics_tick": 42,
        "sim_time": 0.7,
        "player": {"lane": 1, "speed": 20.0},
        "traffic": [],
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


class CaptureSocket:
    def __init__(self):
        self.messages = []

    async def send(self, payload):
        self.messages.append(json.loads(payload))


class IllegalPolicy:
    label_source = "synthetic"

    def choose_action(self, _state):
        return "LEFT"


class SlowPolicy:
    label_source = "original_model"

    async def choose_action(self, _state):
        await asyncio.sleep(0.05)
        return "LEFT"


class BlockingPolicy:
    """A synchronous policy that holds the inference worker open."""

    label_source = "original_model"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def choose_action(self, _state):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        self.release.wait(timeout=2.0)
        with self._lock:
            self.active -= 1
        return "LEFT"


class UncertainStayPolicy:
    label_source = "original_model"

    def choose_action(self, _state):
        return PolicyDecision(
            action="STAY",
            label_source="original_model",
            policy_name="onnx_kev",
            probabilities={"LEFT": 0.22, "STAY": 0.40, "RIGHT": 0.21, "BRAKE": 0.17},
            confidence=0.40,
        )


class ProtocolTests(unittest.TestCase):
    def test_controller_default_port_matches_game_default(self):
        self.assertEqual(parse_args([]).port, 8766)

    def test_accepts_well_formed_state_and_rejects_missing_identity(self):
        self.assertTrue(validate_state_message(sample_state()))
        invalid = sample_state()
        invalid.pop("episode_id")
        self.assertFalse(validate_state_message(invalid))

    def test_state_requires_a_valid_authoritative_legal_action_set(self):
        self.assertTrue(validate_state_message(sample_state()))
        missing = sample_state()
        missing.pop("legal_actions")
        self.assertFalse(validate_state_message(missing))
        invalid = sample_state()
        invalid["legal_actions"] = ["LEFT", "TELEPORT"]
        self.assertFalse(validate_state_message(invalid))

    def test_action_envelope_uses_the_decision_state_identity(self):
        state = sample_state()
        message = make_action_message(
            state, seq=3, action="LEFT", duration=0.4, label_source="rule_based"
        )
        self.assertEqual(message["protocol_version"], 1)
        self.assertEqual(message["episode_id"], "ep-test")
        self.assertEqual(message["tick"], 42)
        self.assertEqual(message["seq"], 3)
        self.assertEqual(message["action"], "LEFT")
        self.assertEqual(message["duration"], 0.4)
        self.assertIn(message["action"], ALLOWED_ACTIONS)

    def test_each_named_policy_returns_a_supported_action(self):
        for name in ("dummy", "rule_based"):
            with self.subTest(policy=name):
                action = load_policy(name).choose_action(sample_state())
                self.assertIn(action, ALLOWED_ACTIONS)

    def test_unconfigured_model_policy_fails_instead_of_simulating_stay(self):
        for name in ("model", "fine_tuned_model"):
            with self.subTest(policy=name), self.assertRaises(ValueError):
                load_policy(name)

    def test_dummy_policy_starts_with_a_lane_change_for_bridge_smoke_tests(self):
        self.assertEqual(load_policy("dummy").choose_action(sample_state()), "LEFT")

    def test_server_separates_requested_and_simulation_accepted_actions(self):
        server = ControllerServer("dummy")
        server._handle_event({"type": "action_decision", "accepted": True, "applied_action": "LEFT", "controller_seq": 1})
        server._handle_event({"type": "action_decision", "accepted": False, "applied_action": "STAY", "controller_seq": 2})
        server._handle_event({"type": "external_action_ignored", "controller_seq": 3})
        metrics = server.metrics()
        self.assertEqual(metrics["applied_actions"], {"LEFT": 1})
        self.assertEqual(metrics["rejected_actions"], 2)

    def test_action_result_and_rejection_event_are_counted_once(self):
        server = ControllerServer("dummy")
        server._handle_event({
            "type": "action_decision", "episode_id": "ep-a", "accepted": False,
            "applied_action": "STAY", "controller_seq": 7,
        })
        server._handle_event({
            "type": "external_action_rejected", "episode_id": "ep-a", "controller_seq": 7,
        })
        server._handle_event({
            "type": "external_action_rejected", "episode_id": "ep-a", "controller_seq": 8,
        })
        self.assertEqual(server.metrics()["rejected_actions"], 2)

    def test_server_never_sends_an_action_illegal_for_the_decision_state(self):
        server = ControllerServer("dummy")
        server.policy = IllegalPolicy()
        state = sample_state()
        state["legal_actions"] = ["STAY", "BRAKE"]
        socket = CaptureSocket()

        asyncio.run(server._handle_state(socket, state))

        self.assertEqual(socket.messages[0]["action"], "STAY")
        self.assertEqual(socket.messages[0]["label_source"], "controller_fallback")
        self.assertEqual(server.invalid_actions, 1)

    def test_evaluation_mode_disables_safety_substitution(self):
        server = ControllerServer("rule_based", safety_fallback=True, evaluation_mode=True)
        self.assertIsNone(server.safety_policy)
        self.assertTrue(server.evaluation_mode)

    def test_watch_safety_vetoes_an_uncertain_stay_when_a_lane_is_clear(self):
        server = ControllerServer("rule_based", decision_timeout=0.2, safety_fallback=True)
        server.policy_name = "onnx_kev"
        server.policy = UncertainStayPolicy()
        state = sample_state()
        state["traffic"] = [{
            "id": "veh_1", "lane": 1, "relative_distance_z": 10.0,
            "relative_speed_z": 2.0, "type": "car",
        }]
        socket = CaptureSocket()
        asyncio.run(server._handle_state(socket, state))
        self.assertIn(socket.messages[0]["action"], {"LEFT", "RIGHT", "BRAKE"})
        self.assertEqual(socket.messages[0]["label_source"], "controller_fallback")
        self.assertEqual(socket.messages[0]["decision_metadata"]["fallback_reason"], "low_confidence_safety")

    def test_slow_policy_times_out_to_unlabelled_safe_fallback(self):
        server = ControllerServer("rule_based", decision_timeout=0.005)
        server.policy = SlowPolicy()
        socket = CaptureSocket()

        asyncio.run(server._handle_state(socket, sample_state()))

        self.assertEqual(socket.messages[0]["action"], "STAY")
        self.assertEqual(socket.messages[0]["label_source"], "controller_fallback")
        self.assertEqual(socket.messages[0]["decision_metadata"]["fallback_reason"], "decision_timeout")

    def test_decision_from_previous_episode_is_discarded_after_reset(self):
        server = ControllerServer("rule_based", decision_timeout=0.2)
        server.policy = SlowPolicy()
        state = sample_state()
        latest = {"state": state}
        socket = CaptureSocket()

        async def run_decision():
            decision = asyncio.create_task(server._handle_state(socket, state, latest))
            await asyncio.sleep(0.005)
            latest["state"] = {**state, "episode_id": "ep-next", "sim_time": 0.0, "physics_tick": 0}
            await decision

        asyncio.run(run_decision())

        self.assertEqual(socket.messages, [])
        self.assertEqual(server.stale_decisions, 1)

    def test_non_decision_state_updates_latest_without_queueing(self):
        server = ControllerServer("dummy")
        queue = asyncio.Queue(maxsize=1)
        latest = {}
        update = sample_state()
        update["decision_request"] = False
        self.assertEqual(server._accept_state(update, queue, latest), "ignored")
        self.assertTrue(queue.empty())
        self.assertEqual(server.states_ignored, 1)
        self.assertEqual(latest["state"]["physics_tick"], 42)

        request = sample_state()
        request["physics_tick"] = 43
        request["state_id"] = "ep-test:43"
        self.assertEqual(server._accept_state(request, queue, latest), "queued")
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(server.decision_requests, 1)

        invalid = sample_state()
        invalid.pop("episode_id")
        self.assertEqual(server._accept_state(invalid, queue, latest), "invalid")

    def test_live_mode_reports_timeout_and_never_sends_a_fallback(self):
        server = ControllerServer("rule_based", decision_timeout=0.005, live_mode=True)
        server.policy = SlowPolicy()
        socket = CaptureSocket()

        asyncio.run(server._handle_state(socket, sample_state()))

        self.assertEqual(len(socket.messages), 1)
        self.assertEqual(socket.messages[0]["type"], "model_error")
        self.assertEqual(socket.messages[0]["reason"], "decision_timeout")
        metrics = server.metrics()
        self.assertEqual(metrics["decisions"], 0)
        self.assertEqual(metrics["decision_timeouts"], 1)
        self.assertTrue(metrics["live_mode"])

    def test_live_mode_disables_safety_substitution(self):
        server = ControllerServer("rule_based", safety_fallback=True, live_mode=True)
        self.assertIsNone(server.safety_policy)

    def test_evaluation_mode_reports_timeout_and_never_sends_a_fallback(self):
        server = ControllerServer("rule_based", decision_timeout=0.005, evaluation_mode=True)
        server.policy = SlowPolicy()
        socket = CaptureSocket()

        asyncio.run(server._handle_state(socket, sample_state()))

        self.assertEqual(socket.messages[0]["type"], "model_error")
        self.assertEqual(socket.messages[0]["reason"], "decision_timeout")

    def test_metrics_are_written_on_demand_for_a_disconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            server = ControllerServer("dummy", metrics_out=path)
            server._handle_event({"type": "episode_end", "termination_reason": "timeout"})
            server.write_metrics()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["policy"], "dummy")
            self.assertEqual(payload["completed_episodes"], 1)
            self.assertEqual(payload["decisions"], 0)

    def test_single_inference_worker_refuses_an_overlapping_decision(self):
        async def run():
            server = ControllerServer("rule_based", decision_timeout=5.0, live_mode=True)
            policy = BlockingPolicy()
            server.policy = policy
            socket = CaptureSocket()
            first = sample_state()
            second = sample_state()
            second["physics_tick"] = 43
            second["state_id"] = "ep-test:43"

            first_task = asyncio.create_task(server._handle_state(socket, first))
            while not policy.started.is_set():
                await asyncio.sleep(0.01)
            second_task = asyncio.create_task(server._handle_state(socket, second))
            await asyncio.sleep(0.05)
            policy.release.set()
            await asyncio.gather(first_task, second_task)
            server.close()
            return server, policy, socket

        server, policy, socket = asyncio.run(run())

        self.assertEqual(policy.max_active, 1)
        self.assertEqual(server.inference_busy, 1)
        self.assertTrue(any(message.get("type") == "model_error" for message in socket.messages))
        self.assertTrue(any(message.get("type") == "action" for message in socket.messages))


if __name__ == "__main__":
    unittest.main()
