import json
import tempfile
import unittest
from pathlib import Path

from dataset_prep import read_jsonl
from export_decisions import (
    ACTIONS,
    FEATURE_NAMES,
    OUTCOME_COLUMNS,
    ROW_COLUMNS,
    assign_splits,
    build_rows,
    feature_vector,
    file_sha256,
    lane_gap,
    main,
    nearest_vehicles,
    project_decision,
    time_to_collision,
    write_csv,
    write_jsonl,
)


def temp_root(test):
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return Path(directory.name)


def vehicle(vehicle_id, lane, distance, closing_speed, kind="car"):
    return {
        "id": vehicle_id,
        "lane": lane,
        "type": kind,
        "position": {"x": 0.0, "y": 0.0, "z": distance},
        "velocity": {"x": 0.0, "y": 0.0, "z": 16.0},
        "relative_distance_z": distance,
        "relative_speed_z": closing_speed,
    }


def snapshot(episode_id, seed, tick, traffic=None, lane=1, legal=None, speed=20.0):
    return {
        "protocol_version": 1,
        "episode_id": episode_id,
        "scenario_seed": seed,
        "sim_time": tick / 60.0,
        "physics_tick": tick,
        "state_id": f"{episode_id}:{tick}",
        "player": {
            "lane": lane,
            "target_lane": lane,
            "lane_change_in_progress": False,
            "lane_change_cooldown": 0.0,
            "speed": speed,
            "position": {"x": 0.0, "y": 0.0, "z": tick / 60.0 * speed},
            "velocity": {"x": 0.0, "y": 0.0, "z": speed},
            "current_action": "STAY",
            "braking": False,
        },
        "traffic": traffic or [],
        "legal_actions": legal or ["LEFT", "STAY", "RIGHT", "BRAKE"],
        "score": 0.0,
        "distance": tick / 3.0,
        "survival_time": tick / 60.0,
        "collision_count": 0,
        "collision": False,
        "episode_status": "running",
        "control_source": "external_ai",
    }


def decision_event(episode_id, seq, tick, action, state, accepted=True, probability=None):
    metadata = {"policy_name": "rule_based", "label_source": "rule_based"}
    if probability is not None:
        metadata["probabilities"] = probability
    return {
        "type": "action_decision",
        "event_id": f"{episode_id}:action:{seq}",
        "decision_seq": seq,
        "controller_seq": seq,
        "episode_id": episode_id,
        "physics_tick": tick,
        "decision_tick": tick,
        "application_tick": tick + 1,
        "state_id": f"{episode_id}:{tick}",
        "requested_action": action,
        "applied_action": action if accepted else "STAY",
        "accepted": accepted,
        "control_source": "rule_based",
        "label_source": "rule_based",
        "decision_metadata": metadata,
        "state_before": state,
    }


def episode_summary(episode_id, seed, termination="timeout", score=100.0):
    return {
        "episode_id": episode_id,
        "scenario_seed": seed,
        "score": score,
        "distance": 200.0,
        "survival_time": 30.0,
        "collision_count": 0 if termination == "timeout" else 1,
        "overtaken_count": 4,
        "termination_reason": termination,
    }


def write_session(root, session_id, seed, decisions):
    """decisions: list of (seq, tick, action, state, accepted)."""
    session = Path(root) / session_id
    session.mkdir(parents=True, exist_ok=True)
    events = []
    snapshots = []
    for seq, tick, action, state, accepted in decisions:
        events.append(decision_event(state["episode_id"], seq, tick, action, state, accepted=accepted))
        snapshots.append(state)
    # Telemetry snapshots are a superset; the join must prefer the event's own state.
    telemetry = [snapshot(state["episode_id"], seed, state["physics_tick"] - 1) for state in snapshots]
    episodes = [episode_summary(decisions[0][3]["episode_id"], seed)]
    write_jsonl(session / "telemetry.jsonl", telemetry)
    write_jsonl(session / "events.jsonl", events)
    write_jsonl(session / "episodes.jsonl", episodes)
    return session


class DecisionExportTests(unittest.TestCase):
    def test_one_row_per_accepted_legal_decision(self):
        state = snapshot("ep_a", 5, 10, traffic=[vehicle("v0", 0, 30.0, 5.0)])
        session = write_session(temp_root(self), "session_a", 5, [
            (1, 10, "LEFT", state, True),
            (2, 11, "BRAKE", snapshot("ep_a", 5, 11), False),
        ])
        illegal_state = snapshot("ep_a", 5, 12, legal=["STAY", "BRAKE"])
        events, _ = read_jsonl(session / "events.jsonl")
        events.append(decision_event("ep_a", 3, 12, "LEFT", illegal_state))
        write_jsonl(session / "events.jsonl", events)
        telemetry, _ = read_jsonl(session / "telemetry.jsonl")
        telemetry.append(illegal_state)
        write_jsonl(session / "telemetry.jsonl", telemetry)

        rows, report = build_rows([session])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "LEFT")
        self.assertEqual(report["rejected_actions"], 1)
        self.assertEqual(report["illegal_labels"], 1)

    def test_outcome_columns_are_recorded_but_never_enter_the_features(self):
        state = snapshot("ep_a", 5, 10, traffic=[vehicle("v0", 1, 12.0, 6.0)])
        rows, _report = build_rows([write_session(
            temp_root(self), "session_a", 5, [(1, 10, "BRAKE", state, True)]
        )])
        row = rows[0]
        for column in OUTCOME_COLUMNS:
            self.assertIn(column, row)
        self.assertEqual(row["episode_termination"], "timeout")

        baseline = feature_vector(row)
        tampered = dict(row)
        tampered.update({
            "episode_score": 1e9, "episode_distance": 1e9, "episode_survival_time": 0.0,
            "episode_collisions": 99, "episode_termination": "crash", "episode_success": 0,
            "episode_overtaken": 999,
        })
        self.assertEqual(baseline, feature_vector(tampered))

    def test_feature_names_exclude_outcomes_ids_and_labels(self):
        forbidden = set(OUTCOME_COLUMNS) | {
            "row_id", "example_id", "session_id", "episode_id", "scenario_seed", "split",
            "decision_seq", "tick", "sim_time", "action", "label_source", "accepted",
            "applied_action", "application_latency_ticks", "legal_actions",
        }
        self.assertFalse(set(FEATURE_NAMES) & forbidden)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_time_to_collision_only_applies_to_a_closing_vehicle_ahead(self):
        self.assertAlmostEqual(time_to_collision(20.0, 5.0), 4.0)
        self.assertIsNone(time_to_collision(20.0, 0.0))
        self.assertIsNone(time_to_collision(-20.0, 5.0))
        self.assertAlmostEqual(time_to_collision(10_000.0, 0.5), 60.0)

    def test_nearest_vehicles_are_ordered_by_absolute_distance_and_capped(self):
        traffic = [vehicle("far", 0, 80.0, 5.0), vehicle("behind", 1, -5.0, -3.0),
                   vehicle("near", 2, 10.0, 2.0), vehicle("mid", 1, 40.0, 4.0)]
        ordered = nearest_vehicles(traffic)
        self.assertEqual([item[2]["id"] for item in ordered], ["behind", "near", "mid"])

    def test_nearest_vehicles_tie_break_matches_rounded_distance_ordering(self):
        # Raw distances put "b" first; both round to 5.0, so the stable sort must
        # keep the snapshot order, exactly like kev_state.compact_kev_state.
        traffic = [vehicle("a", 0, 5.004, 1.0), vehicle("b", 1, 5.003, 1.0), vehicle("c", 2, 9.0, 1.0)]
        ordered = nearest_vehicles(traffic)
        self.assertEqual([item[2]["id"] for item in ordered], ["a", "b", "c"])

    def test_lane_gap_ignores_vehicles_behind_and_missing_lanes(self):
        traffic = [vehicle("ahead", 1, 25.0, 5.0), vehicle("behind", 1, -10.0, 5.0),
                   vehicle("other", 2, 5.0, 5.0)]
        self.assertEqual(lane_gap(traffic, 1), 25.0)
        self.assertEqual(lane_gap(traffic, 2), 5.0)
        self.assertIsNone(lane_gap(traffic, 0))
        self.assertIsNone(lane_gap(traffic, -1))
        self.assertIsNone(lane_gap(traffic, 3))

    def test_rows_record_probabilities_only_when_the_controller_supplied_them(self):
        state = snapshot("ep_a", 5, 10)
        with_probabilities = decision_event("ep_a", 1, 10, "STAY", state,
                                            probability={"LEFT": 0.1, "STAY": 0.7, "RIGHT": 0.1, "BRAKE": 0.1})
        without = decision_event("ep_a", 2, 11, "STAY", state)
        session = temp_root(self) / "session_a"
        session.mkdir(parents=True, exist_ok=True)
        write_jsonl(session / "telemetry.jsonl", [state])
        write_jsonl(session / "events.jsonl", [with_probabilities, without])
        write_jsonl(session / "episodes.jsonl", [episode_summary("ep_a", 5)])

        rows, _report = build_rows([session])

        self.assertEqual(rows[0]["prob_STAY"], 0.7)
        self.assertIsNone(rows[1]["prob_STAY"])

    def test_two_sessions_with_the_same_episode_name_stay_distinct(self):
        state = snapshot("ep_1_1", 1, 10)
        root = temp_root(self)
        first = write_session(root, "session_a", 1, [(1, 10, "STAY", state, True)])
        second = write_session(root, "session_b", 1, [(1, 10, "STAY", state, True)])

        rows, report = build_rows([first, second])

        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["episode_id"] for row in rows}), 2)
        self.assertEqual(len(report["sessions"]), 2)

    def test_three_way_split_is_seed_grouped_and_deterministic(self):
        rows = []
        for seed in range(1, 9):
            for decision in range(3):
                rows.append({
                    "episode_id": f"session:{seed}:{seed}",
                    "scenario_seed": seed,
                    "action": ACTIONS[decision % len(ACTIONS)],
                    "split": None,
                })
        first = assign_splits([dict(row) for row in rows], 0.2, 0.2, 7)
        second = assign_splits([dict(row) for row in rows], 0.2, 0.2, 7)
        self.assertEqual(first, second)
        seeds = {name: set(summary["scenario_seeds"]) for name, summary in first.items()}
        self.assertFalse(seeds["train"] & seeds["validation"])
        self.assertFalse(seeds["train"] & seeds["test"])
        self.assertFalse(seeds["validation"] & seeds["test"])

        assigned = [dict(row) for row in rows]
        assign_splits(assigned, 0.2, 0.2, 7)
        by_seed = {}
        for row in assigned:
            by_seed.setdefault(row["scenario_seed"], set()).add(row["split"])
        self.assertTrue(all(len(names) == 1 for names in by_seed.values()))

    def test_split_requires_at_least_three_seed_groups(self):
        rows = [{"episode_id": "a", "scenario_seed": 1, "action": "STAY", "split": None},
                {"episode_id": "b", "scenario_seed": 2, "action": "STAY", "split": None}]
        with self.assertRaises(ValueError):
            assign_splits(rows, 0.2, 0.2, 7)

    def test_split_rejects_an_episode_shared_by_validation_and_test(self):
        rows = [
            {"episode_id": f"session:ep_{seed}", "scenario_seed": seed, "action": "STAY", "split": None}
            for seed in range(1, 11)
        ]
        summary = assign_splits([dict(row) for row in rows], 0.2, 0.2, 7)
        validation_seed = summary["validation"]["scenario_seeds"][0]
        test_seed = summary["test"]["scenario_seeds"][0]
        leaky = [dict(row) for row in rows]
        # The same episode id under a validation seed and a test seed must be refused.
        leaky.append({
            "episode_id": f"session:ep_{validation_seed}",
            "scenario_seed": test_seed,
            "action": "STAY",
            "split": None,
        })
        with self.assertRaises(SystemExit) as caught:
            assign_splits(leaky, 0.2, 0.2, 7)
        self.assertIn("episode", str(caught.exception))
        self.assertIn("validation", str(caught.exception))
        self.assertIn("test", str(caught.exception))

    def test_csv_and_jsonl_share_one_schema(self):
        state = snapshot("ep_a", 5, 10)
        rows, _report = build_rows([write_session(
            temp_root(self), "session_a", 5, [(1, 10, "STAY", state, True)]
        )])
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "rows.csv"
            jsonl_path = Path(directory) / "rows.jsonl"
            write_csv(csv_path, rows)
            write_jsonl(jsonl_path, rows)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            loaded = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        self.assertEqual(header, list(ROW_COLUMNS))
        self.assertEqual(set(loaded), set(ROW_COLUMNS))

    def test_main_writes_dataset_and_data_card(self):
        root = temp_root(self)
        for seed in (11, 22, 33, 44, 55):
            state = snapshot(f"ep_{seed}", seed, 10, traffic=[vehicle("v0", 1, 15.0, 5.0)])
            write_session(root, f"session_{seed}", seed, [(1, 10, "BRAKE", state, True)])
        output = root / "decisions.jsonl"

        code = main([
            "--recordings", str(root), "--output", str(output),
            "--validation-fraction", "0.2", "--test-fraction", "0.2", "--split-seed", "3",
        ])

        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        card = json.loads(output.with_name("decisions.card.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 5)
        self.assertTrue(output.with_name("decisions.csv").is_file())
        self.assertEqual(card["counts"]["rows"], 5)
        self.assertEqual(card["counts"]["episodes"], 5)
        self.assertEqual(sum(card["action_counts"].values()), 5)
        self.assertEqual(card["label_source_counts"], {"rule_based": 5})
        self.assertEqual(card["conditions"]["policy"], "rule_based")
        self.assertEqual(card["decisions_sha256"], file_sha256(output))
        self.assertIn("commit", card["code_commit"])
        self.assertIn("rule_based", " ".join(card["limits"]))
        self.assertEqual(sum(split["rows"] for split in card["splits"].values()), 5)
        split_seeds = [set(split["scenario_seeds"]) for split in card["splits"].values()]
        self.assertFalse(split_seeds[0] & split_seeds[1])
        self.assertFalse(split_seeds[0] & split_seeds[2])
        self.assertFalse(split_seeds[1] & split_seeds[2])

    def test_project_decision_rejects_rows_without_a_usable_state(self):
        self.assertIsNone(project_decision({"state": {}, "action": "STAY", "accepted": True}, "s"))
        self.assertIsNone(project_decision({"state": None, "action": "STAY", "accepted": True}, "s"))


if __name__ == "__main__":
    unittest.main()
