import json
import tempfile
import unittest
from pathlib import Path

import build_kev_v2_scene as builder


def make_state(seed=900001, lane=1, traffic=None, legal=None):
    return {
        "episode_id": f"ep_{seed}_1",
        "scenario_seed": seed,
        "observation_schema": 2,
        "physics_tick": 60,
        "sim_time": 1.0,
        "player": {
            "lane": lane,
            "target_lane": lane,
            "speed": 20.0,
            "acceleration": 0.0,
            "base_speed": 20.0,
            "max_speed": 28.0,
            "braking": False,
            "lane_change_in_progress": False,
            "lane_change_cooldown": 0.0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "previous_action": "STAY",
            "current_action": "STAY",
            "recent_actions": [],
        },
        "traffic": traffic or [],
        "legal_actions": legal or ["LEFT", "STAY", "RIGHT", "BRAKE"],
        "score": 0,
        "distance": 0,
        "survival_time": 1.0,
        "collision": False,
        "collision_count": 0,
        "episode_status": "running",
        "control_source": "safe_teacher",
        "protocol_version": 2,
    }


def make_event(seed, action, seq, source="safe_teacher", state=None):
    return {
        "type": "action_decision",
        "event_id": f"ep_{seed}_1:action:{seq}",
        "episode_id": f"ep_{seed}_1",
        "scenario_seed": seed,
        "decision_seq": seq,
        "physics_tick": seq * 60,
        "decision_sim_time": float(seq),
        "label_source": source,
        "accepted": True,
        "human_action": action if source == "human" else None,
        "requested_action": action,
        "human_adjudicated_action": None,
        "recent_actions": [],
        "state_before": state or make_state(seed),
    }


def make_policy():
    from tokenizers import Tokenizer
    from policies.onnx_kev import OnnxKevPolicy

    tokenizer = Path(__file__).resolve().parents[2] / "models" / "kev-original" / "tokenizer.json"
    if not tokenizer.exists():
        raise unittest.SkipTest(f"tokenizer not present: {tokenizer}")
    policy = OnnxKevPolicy(
        model_path="", tokenizer_path=str(tokenizer), length=256, prompt_mode="scene"
    )
    policy.tok = Tokenizer.from_file(str(tokenizer))
    return policy


class BuilderTests(unittest.TestCase):
    def test_turn_safety_flags_close_target_lane_vehicle(self):
        state = make_state(
            lane=2,
            traffic=[{"lane": 1, "relative_distance_z": 2.87, "relative_speed_z": 14.57}],
        )
        safe, reason = builder.turn_safety(state, "LEFT")
        self.assertFalse(safe)
        self.assertIn("target_lane_front", reason)

    def test_turn_safety_accepts_clear_target_lane(self):
        state = make_state(
            lane=2,
            traffic=[{"lane": 1, "relative_distance_z": 40.0, "relative_speed_z": 2.0}],
        )
        safe, reason = builder.turn_safety(state, "LEFT")
        self.assertTrue(safe)
        self.assertIsNone(reason)

    def test_extract_uses_exact_encoder_and_stride(self):
        policy = make_policy()
        plan = {
            900001: {
                "seed": 900001,
                "split": "train",
                "source": "safe_teacher",
                "cell": "B",
                "traffic_density": 1.0,
                "duration_s": 60,
            }
        }
        with tempfile.TemporaryDirectory(prefix="kev-v2-test-", dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            session = root / "recordings" / "session_test"
            session.mkdir(parents=True)
            events = [make_event(900001, "STAY", seq) for seq in range(1, 5)]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
            )
            (session / "episodes.jsonl").write_text(
                json.dumps({
                    "type": "episode_end",
                    "episode_id": "ep_900001_1",
                    "scenario_seed": 900001,
                    "termination_reason": "timeout",
                    "survival_time": 60.0,
                    "collision_count": 0,
                }) + "\n",
                encoding="utf-8",
            )
            candidates, counts = builder.extract_candidates(
                root / "recordings", plan, policy, 256, 2
            )
            self.assertEqual(len(candidates), 2)
            for candidate in candidates:
                meta = candidate["meta"]
                self.assertTrue(meta["included"])
                self.assertLessEqual(meta["total_tokens"], 256)
                self.assertEqual(meta["legal_actions"], ["LEFT", "STAY", "RIGHT", "BRAKE"])
                self.assertEqual(
                    list(candidate["record"]["questions"]["next_action"]["criteria"]),
                    meta["legal_actions"],
                )
                policy._encode(
                    candidate["meta"]["state_before"]
                    if "state_before" in candidate["meta"]
                    else make_state(900001)
                )
                self.assertEqual(
                    candidate["record"]["state"], policy.last_encoding["state_text"]
                )
            self.assertEqual(counts["counts"].get("stride_skipped:safe_teacher"), 2)

    def test_human_unsafe_turn_is_excluded_but_preserved(self):
        policy = make_policy()
        plan = {
            644099890: {
                "seed": 644099890,
                "split": "train",
                "source": "human",
                "cell": "H",
                "traffic_density": 0.7,
                "duration_s": None,
            }
        }
        unsafe_state = make_state(
            seed=644099890,
            lane=2,
            traffic=[{"lane": 1, "relative_distance_z": 2.87, "relative_speed_z": 14.57}],
        )
        with tempfile.TemporaryDirectory(prefix="kev-v2-test-", dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            session = root / "recordings" / "session_human"
            session.mkdir(parents=True)
            (session / "events.jsonl").write_text(
                json.dumps(make_event(644099890, "LEFT", 1, "human", unsafe_state)) + "\n",
                encoding="utf-8",
            )
            (session / "episodes.jsonl").write_text(
                json.dumps({
                    "type": "episode_end",
                    "episode_id": "ep_644099890_1",
                    "scenario_seed": 644099890,
                    "termination_reason": "crash",
                    "survival_time": 3.5,
                    "collision_count": 1,
                }) + "\n",
                encoding="utf-8",
            )
            candidates, _ = builder.extract_candidates(root / "recordings", plan, policy, 256, 2)
            self.assertEqual(len(candidates), 1)
            meta = candidates[0]["meta"]
            self.assertFalse(meta["included"])
            self.assertEqual(meta["label_quality"], "unsafe")
            self.assertIn("target_lane_front", meta["excluded_reason"])
            self.assertIn("state_before", meta)

    def test_validate_rejects_seed_leak_and_human_holdout(self):
        policy = make_policy()
        plan = {
            1: {"seed": 1, "split": "train", "source": "human", "cell": "H",
                "traffic_density": 0.7, "duration_s": None},
            2: {"seed": 2, "split": "train", "source": "safe_teacher", "cell": "B",
                "traffic_density": 1.0, "duration_s": 60},
        }
        with tempfile.TemporaryDirectory(prefix="kev-v2-test-", dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            session = root / "recordings" / "session_test"
            session.mkdir(parents=True)
            events = [
                make_event(1, "STAY", 1, "human"),
                make_event(2, "STAY", 1, "safe_teacher"),
                make_event(2, "STAY", 2, "safe_teacher"),
            ]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
            )
            (session / "episodes.jsonl").write_text(
                "\n".join(json.dumps({
                    "type": "episode_end",
                    "episode_id": f"ep_{seed}_1",
                    "scenario_seed": seed,
                    "termination_reason": "timeout",
                    "survival_time": 60.0,
                    "collision_count": 0,
                }) for seed in (1, 2)) + "\n",
                encoding="utf-8",
            )
            candidates, _ = builder.extract_candidates(root / "recordings", plan, policy, 256, 1)
            self.assertEqual(len(candidates), 3)
            candidates[1]["meta"]["split"] = "test"
            validation = builder.validate(candidates, plan, 256)
            self.assertTrue(any("seed 2 appears" in problem for problem in validation["problems"]))
            candidates[0]["meta"]["split"] = "test"
            validation = builder.validate(candidates, plan, 256)
            self.assertTrue(any("human rows appear in test" in problem for problem in validation["problems"]))


if __name__ == "__main__":
    unittest.main()
