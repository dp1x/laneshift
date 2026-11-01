import unittest

from kev_state import lane_observation, render_lane_state


def sample_state():
    return {
        "episode_id": "ep-secret",
        "scenario_seed": 42,
        "sim_time": 3.0,
        "player": {"lane": 1, "speed": 20.0, "braking": False, "target_lane": 1},
        "traffic": [
            {"id": "left-ahead", "lane": 0, "relative_distance_z": 24.0, "relative_speed_z": -2.0},
            {"id": "left-behind", "lane": 0, "relative_distance_z": -15.0, "relative_speed_z": -3.0},
            {"id": "center-ahead", "lane": 1, "relative_distance_z": 11.0, "relative_speed_z": 4.0},
            {"id": "right-ahead", "lane": 2, "relative_distance_z": 35.0, "relative_speed_z": -1.0},
        ],
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


class LaneObservationTests(unittest.TestCase):
    def test_keeps_front_and_rear_vehicle_in_every_lane(self):
        observation = lane_observation(sample_state())
        left, center, right = observation["lanes"]
        self.assertEqual(left["lane"], "LEFT")
        self.assertEqual(left["ahead"]["distance_m"], 24.0)
        self.assertEqual(left["ahead"]["movement"], "opening")
        self.assertEqual(left["behind"]["distance_m"], 15.0)
        self.assertEqual(left["behind"]["movement"], "approaching from behind")
        self.assertEqual(left["behind"]["time_to_contact_s"], 5.0)
        self.assertEqual(center["ahead"]["movement"], "closing")
        self.assertEqual(center["ahead"]["time_to_contact_s"], 2.75)
        self.assertEqual(right["ahead"]["movement"], "opening")
        self.assertIsNone(right["behind"])

    def test_render_is_natural_language_and_has_no_provenance(self):
        text = render_lane_state(lane_observation(sample_state()))
        self.assertIn("Player: lane CENTER", text)
        self.assertIn("Nearest vehicle ahead: 11.0 m", text)
        self.assertIn("Nearest vehicle behind: 15.0 m", text)
        self.assertIn("Legal actions: LEFT, STAY, RIGHT, BRAKE", text)
        self.assertNotIn("ep-secret", text)
        self.assertNotIn("42", text)
        self.assertNotIn("left-ahead", text)

    def test_relative_motion_signs(self):
        observation = lane_observation(sample_state())
        left = observation["lanes"][0]
        center = observation["lanes"][1]
        self.assertLess(left["ahead"]["relative_speed_mps"], 0)
        self.assertEqual(left["ahead"]["movement"], "opening")
        self.assertGreater(center["ahead"]["relative_speed_mps"], 0)
        self.assertEqual(center["ahead"]["movement"], "closing")


if __name__ == "__main__":
    unittest.main()
