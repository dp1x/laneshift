import json
import tempfile
import unittest
from pathlib import Path

import export_kev_pilot as exporter


def make_row(
    example_id="ep_1:action:1",
    seed=1,
    split="train",
    action="STAY",
    legal="LEFT|STAY|RIGHT|BRAKE",
    source="rule_based",
    accepted=1,
):
    row = {
        "row_id": f"session:ep_{seed}_1:{example_id.rsplit(':', 1)[-1]}",
        "example_id": example_id,
        "session_id": "session",
        "episode_id": f"session:ep_{seed}_1",
        "scenario_seed": seed,
        "split": split,
        "decision_seq": int(example_id.rsplit(":", 1)[-1]),
        "tick": 7,
        "sim_time": 0.2333,
        "player_lane": 1,
        "player_target_lane": 1,
        "player_speed": 20.7,
        "lane_change_in_progress": 0,
        "lane_change_cooldown": 0.0,
        "legal_actions": legal,
        "legal_LEFT": 1 if "LEFT" in legal else 0,
        "legal_STAY": 1 if "STAY" in legal else 0,
        "legal_RIGHT": 1 if "RIGHT" in legal else 0,
        "legal_BRAKE": 1 if "BRAKE" in legal else 0,
        "v1_present": 1,
        "v1_lane": 0,
        "v1_distance": 30.62,
        "v1_rel_speed": 5.0,
        "v1_ttc": None,
        "v1_in_player_lane": 0,
        "v2_present": 1,
        "v2_lane": 2,
        "v2_distance": -10.0,
        "v2_rel_speed": -8.0,
        "v2_ttc": None,
        "v2_in_player_lane": 0,
        "v3_present": 0,
        "v3_lane": None,
        "v3_distance": None,
        "v3_rel_speed": None,
        "v3_ttc": None,
        "v3_in_player_lane": 0,
        "nearest_same_lane_gap": None,
        "nearest_left_gap": 30.62,
        "nearest_right_gap": None,
        "min_ttc": None,
        "action": action,
        "label_source": source,
        "prob_LEFT": None,
        "prob_STAY": None,
        "prob_RIGHT": None,
        "prob_BRAKE": None,
        "accepted": accepted,
        "applied_action": action,
        "application_latency_ticks": 0,
        "episode_score": 248.5267,
        "episode_distance": 346.9333,
        "episode_survival_time": 12.7667,
        "episode_collisions": 1,
        "episode_overtaken": 6,
        "episode_termination": "crash",
        "episode_success": 0,
    }
    return row


class ExportKevPilotTests(unittest.TestCase):
    def test_compact_prompt_reconstruction(self):
        row = make_row(action="LEFT")
        record, prompt, state = exporter.row_to_training_record(row)
        self.assertEqual(
            prompt,
            "c1\nP 1 1 21 0\nV 2 -10 -8 0 -\nV 0 +31 +5 0 -",
        )
        self.assertEqual(state["legal_actions"], ["LEFT", "STAY", "RIGHT", "BRAKE"])
        self.assertEqual(record["state"], prompt)
        question = record["questions"]["next_action"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["label"], "LEFT")
        self.assertEqual(list(question["criteria"]), ["LEFT", "STAY", "RIGHT", "BRAKE"])
        self.assertNotIn("episode_score", record["state"])

    def test_training_record_contains_no_outcome_fields(self):
        record, _, _ = exporter.row_to_training_record(make_row())
        self.assertEqual(set(record), {"state", "questions"})
        self.assertEqual(set(record["questions"]["next_action"]), {"type", "instructions", "criteria", "label"})

    def test_illegal_action_is_rejected(self):
        row = make_row(action="BRAKE", legal="LEFT|STAY|RIGHT")
        report = exporter.validate_rows([row], {"rule_based"})
        self.assertTrue(any("not legal" in problem for problem in report["problems"]))

    def test_seed_in_two_splits_is_rejected(self):
        rows = [make_row("ep:action:1", seed=1, split="train"), make_row("ep:action:2", seed=1, split="test")]
        report = exporter.validate_rows(rows, {"rule_based"})
        self.assertTrue(any("more than one split" in problem for problem in report["problems"]))

    def test_source_allowlist_is_enforced(self):
        row = make_row(source="controller_fallback")
        report = exporter.validate_rows([row], {"rule_based"})
        self.assertEqual(report["accepted_rows"], 0)
        self.assertEqual(report["source_counts"], {"controller_fallback": 1})

    def test_export_writes_seed_disjoint_files(self):
        rows = [
            make_row("ep_1:action:1", seed=1, split="train", action="STAY"),
            make_row("ep_2:action:1", seed=2, split="validation", action="BRAKE"),
            make_row("ep_3:action:1", seed=3, split="test", action="LEFT"),
        ]
        with tempfile.TemporaryDirectory(prefix="kev-pilot-", dir=Path(__file__).resolve().parents[1]) as tmp:
            source = Path(tmp) / "decisions.jsonl"
            exporter.write_jsonl(source, rows)
            output = Path(tmp) / "out"
            report = exporter.export(source, output, {"rule_based"})
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["splits"], {"train": 1, "validation": 1, "test": 1})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["totals"]["rows"], 3)
            self.assertEqual(
                manifest["quality_gates"]["source_allowlist"],
                ["rule_based"],
            )
            train = exporter.read_jsonl(output / "kev_train.jsonl")
            self.assertEqual(train[0]["questions"]["next_action"]["label"], "STAY")
            self.assertTrue((output / "approvals.jsonl").exists())

    def test_actual_pilot_export(self):
        source = Path(__file__).resolve().parents[3] / "cris@ds-training" / ".portable-dev" / "pilot" / "decisions.jsonl"
        if not source.exists():
            self.skipTest(f"pilot dataset not present: {source}")
        with tempfile.TemporaryDirectory(prefix="kev-pilot-real-", dir=Path(__file__).resolve().parents[1]) as tmp:
            report = exporter.export(source, Path(tmp) / "out", {"rule_based"}, source.with_name("decisions.card.json"))
            self.assertEqual(report["rows"], 1348)
            self.assertEqual(report["splits"], {"train": 764, "validation": 238, "test": 346})
            self.assertEqual(
                report["action_counts"],
                {"BRAKE": 111, "LEFT": 24, "RIGHT": 21, "STAY": 1192},
            )


if __name__ == "__main__":
    unittest.main()
