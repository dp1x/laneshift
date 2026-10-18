import unittest

from server import ALLOWED_ACTIONS, ControllerServer, load_policy, make_action_message, validate_state_message


def sample_state():
    return {
        "type": "state",
        "protocol_version": 1,
        "episode_id": "ep-test",
        "physics_tick": 42,
        "player": {"lane": 1, "speed": 20.0},
        "traffic": [],
    }


class ProtocolTests(unittest.TestCase):
    def test_accepts_well_formed_state_and_rejects_missing_identity(self):
        self.assertTrue(validate_state_message(sample_state()))
        invalid = sample_state()
        invalid.pop("episode_id")
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
        for name in ("dummy", "rule_based", "model"):
            with self.subTest(policy=name):
                action = load_policy(name).choose_action(sample_state())
                self.assertIn(action, ALLOWED_ACTIONS)

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


if __name__ == "__main__":
    unittest.main()
