import os
import unittest
from pathlib import Path

from kev_state import (
    ACTION_CRITERIA_SCENE,
    KEV_SCENE_INSTRUCTIONS,
    render_scene_state,
    scene_observation,
)
from policies.onnx_kev import SPECIAL

LENGTH = 256


def tokenizer_path():
    candidates = [
        os.environ.get("LANESHIFT_ONNX_TOKENIZER", ""),
        r"C:\LANESHIFT2D-Kev-model\tokenizer.json",
        str(Path(__file__).resolve().parents[2] / "runtime" / "tokenizer.json"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def state(on_screen=9, off_screen=2):
    traffic = []
    for i in range(on_screen):
        traffic.append({
            "id": f"on_{i}",
            "lane": i % 3,
            "relative_distance_z": 6.0 + i * 11.0,
            "relative_speed_z": 3.0 - (i % 7),
            "speed": 15.0 + (i % 6),
            "on_screen": True,
            "type": "truck" if i % 5 == 4 else "car",
            "length": 7.0 if i % 5 == 4 else 4.5,
            "width": 2.4 if i % 5 == 4 else 1.8,
        })
    for i in range(off_screen):
        traffic.append({
            "id": f"off_{i}",
            "lane": i % 3,
            "relative_distance_z": 120.0 + i * 15.0,
            "relative_speed_z": 4.0,
            "speed": 16.0,
            "on_screen": False,
            "type": "car",
            "length": 4.5,
            "width": 1.8,
        })
    return {
        "player": {
            "lane": 1, "target_lane": 1, "position": {"x": 0.0, "z": 100.0},
            "speed": 20.0, "acceleration": 3.0, "braking": False,
            "lane_change_in_progress": False, "lane_change_cooldown": 0.0,
            "previous_action": "STAY", "recent_actions": ["LEFT", "STAY"],
        },
        "traffic": traffic,
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


class SceneTokenBudgetTests(unittest.TestCase):
    def setUp(self):
        path = tokenizer_path()
        if path is None:
            self.skipTest("Kev tokenizer not available")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)

    def tokens(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def branch_length(self):
        q_id = self.tok.token_to_id(SPECIAL[1])
        o_id = self.tok.token_to_id(SPECIAL[2])
        c_id = self.tok.token_to_id(SPECIAL[3])
        d_id = self.tok.token_to_id(SPECIAL[4])
        branch = [q_id] + self.tokens(KEV_SCENE_INSTRUCTIONS)
        for action in ["LEFT", "STAY", "RIGHT", "BRAKE"]:
            branch += [o_id] + self.tokens(f"{action}: {ACTION_CRITERIA_SCENE[action]}") + [c_id]
        branch.append(d_id)
        return len(branch)

    def test_eleven_visible_and_relevant_vehicles_fit_the_graph(self):
        text = render_scene_state(scene_observation(state(9, 2)))
        state_tokens = len(self.tokens(text))
        max_state = LENGTH - self.branch_length()
        self.assertLessEqual(state_tokens + 1, max_state,
                             f"{state_tokens} state tokens do not fit {max_state}")

    def test_seven_visible_vehicles_fit_with_absolute_speed(self):
        text = render_scene_state(scene_observation(state(7, 0)))
        self.assertIn("close abs type", text)
        state_tokens = len(self.tokens(text))
        max_state = LENGTH - self.branch_length()
        self.assertLessEqual(state_tokens + 1, max_state)

    def test_prompt_uses_less_than_full_context(self):
        text = render_scene_state(scene_observation(state(9, 2)))
        total = 1 + len(self.tokens(text)) + self.branch_length()
        self.assertLess(total, LENGTH)

    def test_normal_scene_is_compact(self):
        text = render_scene_state(scene_observation(state(2, 0)))
        self.assertLess(len(self.tokens(text)), 100)


if __name__ == "__main__":
    unittest.main()
