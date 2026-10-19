import unittest

from dataset_prep import (
    export_kev_examples,
    make_kev_training_record,
    prepare_records,
    split_by_episode,
    split_by_scenario,
)


def state(tick, collision=False):
    return {
        "protocol_version": 1,
        "episode_id": "episode-a",
        "scenario_seed": 17,
        "physics_tick": tick,
        "sim_time": tick / 60,
        "player": {"lane": 1, "speed": 20.0},
        "traffic": [],
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
        "score": float(tick),
        "distance": tick / 3,
        "survival_time": tick / 60,
        "collision": collision,
        "episode_status": "crashed" if collision else "running",
    }


class DatasetPreparationTests(unittest.TestCase):
    def test_joins_latest_pre_action_state_and_keeps_outcome_out_of_features(self):
        telemetry = [state(10), state(15, collision=True)]
        events = [{
            "type": "action_decision",
            "episode_id": "episode-a",
            "physics_tick": 12,
            "decision_seq": 1,
            "requested_action": "LEFT",
            "applied_action": "LEFT",
            "accepted": True,
            "control_source": "human",
        }]
        episodes = [{
            "episode_id": "episode-a",
            "scenario_seed": 17,
            "score": 50,
            "distance": 100,
            "survival_time": 5,
            "collision_count": 1,
            "termination_reason": "crash",
        }]

        candidates, report = prepare_records(telemetry, events, episodes)

        self.assertEqual(report["candidate_count"], 1)
        candidate = candidates[0]
        self.assertEqual(candidate["state"]["physics_tick"], 10)
        self.assertFalse(candidate["state"]["collision"] if "collision" in candidate["state"] else False)
        self.assertNotIn("outcome", candidate["state"])
        self.assertEqual(candidate["label_source"], "human")
        self.assertEqual(candidate["outcome"]["termination_reason"], "crash")

    def test_split_keeps_all_records_from_an_episode_together(self):
        records = [
            {"episode_id": "a", "n": 1},
            {"episode_id": "a", "n": 2},
            {"episode_id": "b", "n": 3},
            {"episode_id": "c", "n": 4},
        ]
        split = split_by_episode(records, eval_fraction=0.34, seed=3)
        train_ids = {row["episode_id"] for row in split["train"]}
        eval_ids = {row["episode_id"] for row in split["eval"]}
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(len(split["train"]) + len(split["eval"]), len(records))
        self.assertEqual(split, split_by_episode(records, eval_fraction=0.34, seed=3))

    def test_split_keeps_repeated_scenario_seeds_together(self):
        records = [
            {"episode_id": "ep-a1", "scenario_seed": 101, "n": 1},
            {"episode_id": "ep-a2", "scenario_seed": 101, "n": 2},
            {"episode_id": "ep-b", "scenario_seed": 202, "n": 3},
            {"episode_id": "ep-c", "scenario_seed": 303, "n": 4},
        ]
        split = split_by_episode(records, eval_fraction=0.5, seed=3)
        train_seeds = {row["scenario_seed"] for row in split["train"]}
        eval_seeds = {row["scenario_seed"] for row in split["eval"]}
        self.assertFalse(train_seeds & eval_seeds)
        seed_101_rows = [row for row in records if row["scenario_seed"] == 101]
        seed_101_in_train = any(row in split["train"] for row in seed_101_rows)
        seed_101_in_eval = any(row in split["eval"] for row in seed_101_rows)
        self.assertFalse(seed_101_in_train and seed_101_in_eval)

    def test_rejected_actions_are_not_supervised_labels(self):
        candidates, report = prepare_records(
            [state(5)],
            [{
                "type": "action_decision",
                "episode_id": "episode-a",
                "physics_tick": 5,
                "requested_action": "LEFT",
                "applied_action": "STAY",
                "accepted": False,
                "control_source": "human",
            }],
            [],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(report["rejected_actions"], 1)

    def test_future_embedded_state_is_replaced_by_preceding_snapshot(self):
        future = state(15, collision=True)
        event = {
            "type": "action_decision",
            "episode_id": "episode-a",
            "physics_tick": 12,
            "requested_action": "RIGHT",
            "applied_action": "RIGHT",
            "accepted": True,
            "control_source": "human",
            "state_before": future,
        }
        candidates, _report = prepare_records([state(10)], [event], [])
        self.assertEqual(candidates[0]["state"]["physics_tick"], 10)

    def test_delayed_action_uses_decision_tick_not_application_tick(self):
        observed_state = state(12)
        observed_state["player"]["speed"] = 19.25
        event = {
            "type": "action_decision",
            "episode_id": "episode-a",
            "physics_tick": 18,
            "decision_tick": 12,
            "application_tick": 18,
            "decision_seq": 8,
            "requested_action": "RIGHT",
            "applied_action": "RIGHT",
            "accepted": True,
            "control_source": "external_ai",
            "state_before": observed_state,
        }
        candidates, _report = prepare_records([state(15), state(20)], [event], [])
        candidate = candidates[0]
        self.assertEqual(candidate["decision_tick"], 12)
        self.assertEqual(candidate["application_tick"], 18)
        self.assertEqual(candidate["state"]["physics_tick"], 12)
        self.assertEqual(candidate["state"]["player"]["speed"], 19.25)

    def test_missing_required_snapshot_fields_are_reported(self):
        malformed_state = {"episode_id": "episode-a", "physics_tick": 5, "traffic": []}
        event = {
            "type": "action_decision",
            "episode_id": "episode-a",
            "physics_tick": 5,
            "requested_action": "LEFT",
            "applied_action": "LEFT",
            "accepted": True,
            "control_source": "human",
        }
        candidates, report = prepare_records([malformed_state], [event], [])
        self.assertEqual(candidates, [])
        self.assertEqual(report["malformed_records"], 1)
        self.assertEqual(report["incomplete_actions"], 1)

    def test_kev_export_uses_labelled_choice_training_schema(self):
        input_state = state(10)
        input_state["legal_actions"] = ["LEFT", "STAY", "BRAKE"]
        candidate = {
            "example_id": "episode-a:3",
            "episode_id": "episode-a",
            "scenario_seed": 17,
            "decision_seq": 3,
            "decision_tick": 10,
            "application_tick": 12,
            "state_id": "episode-a:10",
            "state": input_state,
            "action": "LEFT",
            "label_source": "human",
            "outcome": {"termination_reason": "timeout"},
        }
        training = make_kev_training_record(candidate)

        self.assertEqual(training["questions"]["next_action"]["type"], "choice")
        question = training["questions"]["next_action"]
        self.assertEqual(question["label"], "LEFT")
        self.assertIn("LEFT", question["criteria"])
        self.assertNotIn("RIGHT", question["criteria"])
        self.assertNotIn("episode_id", training["state"])
        self.assertNotIn("outcome", training)

    def test_kev_training_requires_explicit_approval_and_source_selection(self):
        candidate = {
            "example_id": "episode-a:4",
            "episode_id": "episode-a",
            "scenario_seed": 17,
            "decision_seq": 4,
            "decision_tick": 10,
            "application_tick": 10,
            "state_id": "episode-a:10",
            "state": state(10),
            "action": "LEFT",
            "label_source": "controller_fallback",
            "outcome": {"termination_reason": "timeout"},
        }
        approvals = [{"example_id": "episode-a:4", "action": "LEFT", "approved": True, "reviewer": "audit"}]
        rows, _provenance, report = export_kev_examples([candidate], approvals, include_sources=["human"])
        self.assertEqual(rows, [])
        self.assertEqual(report["excluded_source"], 1)

        candidate["label_source"] = "human"
        rows, provenance, report = export_kev_examples([candidate], [], include_sources=["human"])
        self.assertEqual(rows, [])
        self.assertEqual(provenance, [])
        self.assertEqual(report["unapproved"], 1)

    def test_three_way_split_keeps_repeated_scenario_seeds_together(self):
        rows = [
            {"episode_id": "a1", "scenario_seed": 1, "id": 1},
            {"episode_id": "a2", "scenario_seed": 1, "id": 2},
            {"episode_id": "b", "scenario_seed": 2, "id": 3},
            {"episode_id": "c", "scenario_seed": 3, "id": 4},
            {"episode_id": "d", "scenario_seed": 4, "id": 5},
            {"episode_id": "e", "scenario_seed": 5, "id": 6},
        ]
        split = split_by_scenario(rows, validation_fraction=0.2, test_fraction=0.2, seed=19)
        seed_sets = [{row["scenario_seed"] for row in split[name]} for name in ("train", "validation", "test")]
        self.assertFalse(seed_sets[0] & seed_sets[1])
        self.assertFalse(seed_sets[0] & seed_sets[2])
        self.assertFalse(seed_sets[1] & seed_sets[2])
        one_seed = [row for row in rows if row["scenario_seed"] == 1]
        self.assertEqual(sum(any(row in split[name] for row in one_seed) for name in split), 1)


if __name__ == "__main__":
    unittest.main()
