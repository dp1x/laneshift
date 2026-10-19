import unittest
import asyncio
import json

from server import ALLOWED_ACTIONS, ControllerServer, load_policy, make_action_message, validate_state_message


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


class ProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
