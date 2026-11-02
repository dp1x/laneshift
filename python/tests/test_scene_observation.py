import unittest

from kev_state import (
    OBSERVATION_SCHEMA_VERSION,
    render_scene_state,
    scene_observation,
)


def state():
    return {
        "episode_id": "ep",
        "scenario_seed": 7,
        "player": {
            "lane": 1,
            "target_lane": 1,
            "lane_change_in_progress": False,
            "lane_change_cooldown": 0.0,
            "current_action": "STAY",
            "speed": 20.0,
            "acceleration": 3.0,
            "braking": False,
            "position": {"x": 0.0, "z": 100.0},
        },
        "traffic": [
            {"id": "same_near", "lane": 1, "relative_distance_z": 10.0,
             "relative_speed_z": 5.0, "speed": 15.0, "on_screen": True,
             "type": "car", "length": 4.5, "width": 1.8},
            {"id": "same_far", "lane": 1, "relative_distance_z": 40.0,
             "relative_speed_z": 8.0, "speed": 12.0, "on_screen": True,
             "type": "truck", "length": 7.0, "width": 2.4},
            {"id": "left_behind", "lane": 0, "relative_distance_z": -14.0,
             "relative_speed_z": -4.0, "speed": 24.0, "on_screen": False,
             "type": "car", "length": 4.5, "width": 1.8},
            {"id": "right_on", "lane": 2, "relative_distance_z": 25.0,
             "relative_speed_z": 2.0, "speed": 18.0, "on_screen": True,
             "type": "car", "length": 4.5, "width": 1.8},
            {"id": "ahead_off", "lane": 0, "relative_distance_z": 130.0,
             "relative_speed_z": 9.0, "speed": 11.0, "on_screen": False,
             "type": "car", "length": 4.5, "width": 1.8},
            {"id": "way_ahead", "lane": 2, "relative_distance_z": 300.0,
             "relative_speed_z": 1.0, "speed": 19.0, "on_screen": False,
             "type": "car", "length": 4.5, "width": 1.8},
            {"id": "way_behind", "lane": 2, "relative_distance_z": -30.0,
             "relative_speed_z": -2.0, "speed": 22.0, "on_screen": False,
             "type": "car", "length": 4.5, "width": 1.8},
        ],
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


class SceneObservationTests(unittest.TestCase):
    def test_all_on_screen_and_relevant_off_screen_vehicles_are_present(self):
        observation = scene_observation(state())
        self.assertEqual(observation["schema_version"], OBSERVATION_SCHEMA_VERSION)
        on_ids = {row["vehicle_id"] for row in observation["on_screen"]}
        off_ids = {row["vehicle_id"] for row in observation["off_screen"]}
        self.assertEqual(on_ids, {"same_near", "same_far", "right_on"})
        self.assertEqual(off_ids, {"left_behind", "ahead_off"})
        all_ids = [row["vehicle_id"] for row in observation["on_screen"] + observation["off_screen"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertNotIn("way_ahead", all_ids)
        self.assertNotIn("way_behind", all_ids)
        visibility = observation["visibility"]
        self.assertEqual(visibility["active_vehicle_count"], 7)
        self.assertEqual(visibility["on_screen_vehicle_count"], 3)
        self.assertEqual(visibility["off_screen_relevant_vehicle_count"], 2)
        self.assertEqual(visibility["off_screen_beyond_range_count"], 2)

    def test_every_vehicle_has_direction_speed_and_collision_dimensions(self):
        observation = scene_observation(state())
        for row in observation["on_screen"] + observation["off_screen"]:
            self.assertIn("relative_distance_z", row)
            self.assertIn("relative_speed_z", row)
            self.assertIn("closing_speed_mps", row)
            self.assertIn("absolute_speed_mps", row)
            self.assertIn("on_screen", row)
            self.assertIn("ahead", row)
            self.assertIn("length_m", row)
            self.assertIn("width_m", row)
        near = next(row for row in observation["on_screen"] if row["vehicle_id"] == "same_near")
        self.assertEqual(near["time_to_collision_s"], 2.0)
        self.assertEqual(near["closing_speed_mps"], 5.0)
        self.assertEqual(near["absolute_speed_mps"], 15.0)
        behind = next(row for row in observation["off_screen"] if row["vehicle_id"] == "left_behind")
        self.assertFalse(behind["ahead"])
        self.assertFalse(behind["on_screen"])
        self.assertEqual(behind["closing_speed_mps"], 4.0)
        self.assertEqual(behind["time_to_collision_s"], 3.5)
        truck = next(row for row in observation["on_screen"] if row["vehicle_id"] == "same_far")
        self.assertEqual((truck["length_m"], truck["width_m"]), (7.0, 2.4))

    def test_lane_summaries_use_only_included_vehicles(self):
        observation = scene_observation(state())
        summaries = {row["lane"]: row for row in observation["lane_summaries"]}
        self.assertEqual(summaries["CENTER"]["vehicle_count"], 2)
        self.assertEqual(summaries["CENTER"]["ahead_count"], 2)
        self.assertEqual(summaries["CENTER"]["behind_count"], 0)
        self.assertEqual(summaries["CENTER"]["nearest_ahead"]["vehicle_id"], "same_near")
        self.assertEqual(summaries["LEFT"]["vehicle_count"], 2)
        self.assertEqual(summaries["LEFT"]["ahead_count"], 1)
        self.assertEqual(summaries["LEFT"]["behind_count"], 1)
        self.assertEqual(summaries["LEFT"]["nearest_behind"]["vehicle_id"], "left_behind")
        self.assertEqual(summaries["RIGHT"]["vehicle_count"], 1)
        self.assertIsNone(summaries["RIGHT"]["nearest_behind"])

    def test_player_manoeuvre_state_is_preserved(self):
        data = state()
        data["player"].update({
            "lane": 1,
            "target_lane": 2,
            "lane_change_in_progress": True,
            "lane_change_cooldown": 0.08,
            "position": {"x": -1.6, "z": 100.0},
            "braking": True,
            "acceleration": -14.0,
            "previous_action": "RIGHT",
            "recent_actions": ["LEFT", "RIGHT"],
        })
        observation = scene_observation(data)
        player = observation["player"]
        self.assertEqual(player["lane"], "CENTER")
        self.assertEqual(player["target_lane"], "RIGHT")
        self.assertEqual(player["lane_change_direction"], "RIGHT")
        self.assertTrue(player["lane_change_in_progress"])
        self.assertAlmostEqual(player["lane_change_progress"], 0.4, places=2)
        self.assertAlmostEqual(player["lane_change_remaining_estimate_s"], 0.16, places=2)
        self.assertTrue(player["braking"])
        self.assertEqual(player["acceleration_mps2"], -14.0)
        self.assertEqual(player["previous_action"], "RIGHT")
        self.assertEqual(player["recent_actions"], ["LEFT", "RIGHT"])

    def test_render_contains_each_included_vehicle_exactly_once(self):
        observation = scene_observation(state())
        text = render_scene_state(observation)
        lines = text.splitlines()
        self.assertEqual(lines[0], "v2")
        self.assertTrue(lines[1].startswith("P "))
        self.assertIn("close abs type", text)
        on_start = lines.index("ON")
        off_start = lines.index("OFF")
        on_rows = [line for line in lines[on_start + 1:off_start] if line != "none"]
        # The last line is the lane summary, so drop it before comparing.
        off_rows = [line for line in lines[off_start + 1:-1] if line != "none"]
        self.assertEqual(len(on_rows), len(observation["on_screen"]))
        self.assertEqual(len(off_rows), len(observation["off_screen"]))
        self.assertEqual(len(on_rows) + len(off_rows), 5)
        self.assertIn("C+10 +5 15 c", on_rows)
        self.assertIn("C+40 +8 12 t", on_rows)
        self.assertIn("L-14 +4 24 c", off_rows)
        self.assertIn("L+130 +9 11 c", off_rows)
        self.assertNotIn("way_ahead", text)
        self.assertNotIn("300", text)

    def test_open_road_render(self):
        data = state()
        data["traffic"] = []
        observation = scene_observation(data)
        text = render_scene_state(observation)
        self.assertIn("ON\nnone", text)
        self.assertIn("OFF\nnone", text)
        self.assertTrue(text.splitlines()[-1].startswith("L0/0 C0/0 R0/0"))

    def test_no_silent_vehicle_drop(self):
        data = state()
        data["traffic"] = [
            {"id": f"v{i}", "lane": i % 3, "relative_distance_z": 5.0 + i * 6.0,
             "relative_speed_z": 2.0, "speed": 18.0, "on_screen": True,
             "type": "car", "length": 4.5, "width": 1.8}
            for i in range(9)
        ]
        observation = scene_observation(data)
        text = render_scene_state(observation)
        self.assertNotIn("close abs type", text)
        self.assertIn("X0", text)
        on_start = text.splitlines().index("ON")
        off_start = text.splitlines().index("OFF")
        on_rows = [line for line in text.splitlines()[on_start + 1:off_start] if line != "none"]
        self.assertEqual(len(on_rows), 9)
        self.assertEqual(len({row["vehicle_id"] for row in observation["on_screen"]}), 9)
        self.assertEqual(len(observation["on_screen"]), 9)

    def test_more_than_the_limit_is_explicitly_counted(self):
        data = state()
        data["traffic"] = [
            {"id": f"v{i}", "lane": i % 3, "relative_distance_z": 5.0 + i * 6.0,
             "relative_speed_z": 2.0, "speed": 18.0, "on_screen": True,
             "type": "car", "length": 4.5, "width": 1.8}
            for i in range(12)
        ]
        observation = scene_observation(data)
        text = render_scene_state(observation)
        lines = text.splitlines()
        on_start = lines.index("ON")
        off_start = lines.index("OFF")
        on_rows = [line for line in lines[on_start + 1:off_start] if line != "none"]
        self.assertEqual(len(on_rows), 9)
        self.assertIn("X3", text)
        self.assertEqual(len(observation["on_screen"]), 12)

    def test_controlled_scene_changes_are_represented(self):
        def render(traffic):
            data = state()
            data["traffic"] = traffic
            return render_scene_state(scene_observation(data))

        def vehicle(lane, distance, relative_speed, kind="car"):
            return {
                "id": f"{kind}-{lane}-{distance}", "lane": lane,
                "relative_distance_z": distance, "relative_speed_z": relative_speed,
                "speed": 18.0, "on_screen": True, "type": kind,
                "length": 7.0 if kind == "truck" else 4.5,
                "width": 2.4 if kind == "truck" else 1.8,
            }

        open_text = render([])
        center_blocked = render([vehicle(1, 10.0, 6.0, "truck")])
        left_safe = render([vehicle(1, 16.0, 3.0), vehicle(2, 9.0, 7.0)])
        right_safe = render([vehicle(1, 16.0, 3.0), vehicle(0, 9.0, 7.0)])
        rear = render([vehicle(1, -10.0, -8.0)])
        two_center = render([vehicle(1, 8.0, 6.0), vehicle(1, 24.0, 2.0)])

        self.assertIn("ON\nnone", open_text)
        self.assertEqual(open_text.splitlines()[-1], "L0/0 C0/0 R0/0 X0")
        self.assertIn("C+10 +6 18 t", center_blocked)
        self.assertIn("C1/0", center_blocked)
        self.assertIn("C+16 +3 18 c", left_safe)
        self.assertIn("R+9 +7 18 c", left_safe)
        self.assertNotIn("L+", left_safe)
        self.assertIn("L+9 +7 18 c", right_safe)
        self.assertIn("C-10 +8 18 c", rear)
        self.assertIn("C0/1", rear)
        self.assertEqual(two_center.count("C+"), 2)
        self.assertNotEqual(open_text, center_blocked)
        self.assertNotEqual(left_safe, right_safe)


if __name__ == "__main__":
    unittest.main()
