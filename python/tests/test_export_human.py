import json
import tempfile
import unittest
from pathlib import Path

from export_human_decisions import export


def event(seed, session_suffix, action, episode, seq):
    state = {
        "episode_id": episode,
        "scenario_seed": seed,
        "state_id": f"{episode}:{seq}",
        "physics_tick": seq * 60,
        "sim_time": float(seq),
        "player": {"lane": 1, "target_lane": 1, "lane_change_in_progress": False,
                   "lane_change_cooldown": 0.0, "current_action": "STAY",
                   "speed": 20.0, "braking": False},
        "traffic": [{"id": "v1", "lane": 1, "relative_distance_z": 20.0,
                     "relative_speed_z": 3.0}],
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
        "observation_schema": 2,
        "score": 99,
        "distance": 200,
    }
    return {
        "type": "action_decision",
        "event_id": f"{episode}:action:{seq}",
        "episode_id": episode,
        "scenario_seed": seed,
        "decision_seq": seq,
        "physics_tick": seq * 60,
        "decision_sim_time": float(seq),
        "human_action": action,
        "requested_action": action,
        "human_adjudicated_action": None,
        "accepted": True,
        "label_source": "human",
        "recent_actions": [],
        "state_before": state,
    }


class HumanExportTests(unittest.TestCase):
    def test_export_separates_raw_training_and_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "recordings" / "session_test"
            session.mkdir(parents=True)
            events = [
                event(1, 1, "LEFT", "ep_1", 1),
                {"type": "action_decision", "label_source": "rule_based",
                 "accepted": True, "requested_action": "STAY"},
                {"type": "action_decision", "label_source": "human",
                 "accepted": True, "human_action": "RIGHT",
                 "episode_id": "ep_1", "decision_seq": 2,
                 "state_before": event(1, 1, "RIGHT", "ep_1", 2)["state_before"]},
            ]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
            )
            (session / "episodes.jsonl").write_text(
                json.dumps({"type": "episode_end", "episode_id": "ep_1",
                            "termination_reason": "timeout", "score": 10}) + "\n",
                encoding="utf-8",
            )
            card = export(root / "recordings", root / "out", "lane", None)
            self.assertEqual(card["decisions"], 2)
            self.assertEqual(card["action_counts"], {"LEFT": 1, "RIGHT": 1})
            self.assertEqual(card["nonhuman_action_counts"], {"rule_based": 1})
            raw = [json.loads(line) for line in (root / "out" / "raw_human.jsonl").read_text().splitlines()]
            self.assertEqual(raw[0]["human_action"], "LEFT")
            self.assertEqual(raw[0]["state_before"]["player"]["lane"], 1)
            self.assertIn("score", raw[0]["state_before"])
            train = [json.loads(line) for line in (root / "out" / "kev_train.jsonl").read_text().splitlines()]
            self.assertEqual(len(train), 2)
            self.assertIn("questions", train[0])
            self.assertNotIn("score", train[0]["state"])
            self.assertNotIn("termination", train[0]["state"])
            self.assertEqual(train[0]["questions"]["next_action"]["label"], "LEFT")
            meta = [json.loads(line) for line in (root / "out" / "kev_train.meta.jsonl").read_text().splitlines()]
            self.assertEqual(meta[0]["episode_outcome"]["termination_reason"], "timeout")

    def test_three_seed_groups_produce_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "recordings" / "session_test"
            session.mkdir(parents=True)
            events = [event(seed, seed, "LEFT", f"ep_{seed}", 1) for seed in (1, 2, 3)]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
            )
            card = export(root / "recordings", root / "out", "lane", None)
            self.assertEqual(card["splits"], {"train": 1, "validation": 1, "test": 1})
            seeds = {}
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (root / "out" / f"kev_{split}.meta.jsonl").read_text().splitlines()]
                seeds[split] = {row["scenario_seed"] for row in rows}
            self.assertTrue(seeds["train"].isdisjoint(seeds["validation"]))
            self.assertTrue(seeds["train"].isdisjoint(seeds["test"]))
            self.assertTrue(seeds["validation"].isdisjoint(seeds["test"]))


if __name__ == "__main__":
    unittest.main()
