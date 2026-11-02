import unittest

from policies.safe_teacher import SafeTeacherPolicy


def vehicle(lane, distance, relative_speed):
    return {
        "lane": lane,
        "relative_distance_z": distance,
        "relative_speed_z": relative_speed,
        "type": "car",
    }


def state(traffic, lane=1, speed=20.0, legal=None):
    return {
        "player": {"lane": lane, "target_lane": lane, "speed": speed},
        "traffic": traffic,
        "legal_actions": legal or ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


class SafeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.policy = SafeTeacherPolicy()

    def test_open_road_stays(self):
        self.assertEqual(self.policy.choose_action(state([])), "STAY")

    def test_blocked_centre_with_clear_left_changes_left(self):
        traffic = [vehicle(1, 10.0, 6.0), vehicle(2, 9.0, 7.0)]
        self.assertEqual(self.policy.choose_action(state(traffic)), "LEFT")

    def test_blocked_centre_with_clear_right_changes_right(self):
        traffic = [vehicle(1, 10.0, 6.0), vehicle(0, 9.0, 7.0)]
        self.assertEqual(self.policy.choose_action(state(traffic)), "RIGHT")

    def test_all_lanes_blocked_brakes(self):
        traffic = [
            vehicle(1, 10.0, 6.0),
            vehicle(0, 7.0, 5.0),
            vehicle(2, 6.0, 4.0),
        ]
        self.assertEqual(self.policy.choose_action(state(traffic)), "BRAKE")

    def test_safe_current_lane_stays_even_with_clear_sides(self):
        traffic = [vehicle(1, 30.0, 1.0), vehicle(0, 50.0, 0.0), vehicle(2, 50.0, 0.0)]
        self.assertEqual(self.policy.choose_action(state(traffic)), "STAY")

    def test_fast_car_behind_makes_target_lane_unsafe(self):
        traffic = [
            vehicle(1, 10.0, 6.0),
            vehicle(0, -10.0, -8.0),
            vehicle(2, 40.0, 0.0),
        ]
        self.assertEqual(self.policy.choose_action(state(traffic)), "RIGHT")

    def test_only_legal_actions_are_returned(self):
        traffic = [vehicle(1, 10.0, 6.0), vehicle(0, 7.0, 5.0), vehicle(2, 6.0, 4.0)]
        legal = ["STAY", "BRAKE"]
        self.assertEqual(self.policy.choose_action(state(traffic, legal=legal)), "BRAKE")
        self.assertIn(self.policy.choose_action(state(traffic, legal=legal)), legal)

    def test_rear_floor_blocks_a_tight_lane_change(self):
        evaluation = self.policy._lane_eval([vehicle(1, -5.0, -1.0)], 1, 20.0)
        self.assertFalse(evaluation["safe"])


if __name__ == "__main__":
    unittest.main()
