import unittest

from experiment_runner import (
    assert_models_loaded,
    build_policy_spec,
    summarize_policy,
)


class ExperimentSummaryTests(unittest.TestCase):
    def test_comparison_row_aggregates_episode_and_controller_metrics(self):
        episodes = [
            {"scenario_seed": 10, "score": 20, "distance": 40, "survival_time": 2, "collision_count": 1, "termination_reason": "crash", "final_lane": 0},
            {"scenario_seed": 11, "score": 60, "distance": 120, "survival_time": 6, "collision_count": 0, "termination_reason": "timeout", "final_lane": 1},
        ]
        controller = {
            "actions": {"LEFT": 2, "STAY": 8},
            "applied_actions": {"LEFT": 1, "STAY": 7},
            "decisions": 12,
            "rejected_actions": 2,
            "decision_latency_ms_mean": 1.5,
            "decision_latency_ms_p95": 3.0,
            "controller_failures": 1,
            "invalid_actions": 2,
            "protocol_rejections": 1,
        }

        row = summarize_policy("rule_based", episodes, controller)

        self.assertEqual(row["episode_count"], 2)
        self.assertEqual(row["mean_score"], 40)
        self.assertEqual(row["collision_frequency"], 0.5)
        self.assertEqual(row["completion_frequency"], 0.5)
        self.assertEqual(row["runs_ending_outside_start_lane"], 1)
        self.assertEqual(row["actions_taken"], 8)
        self.assertEqual(row["action_decisions_requested"], 12)
        self.assertEqual(row["action_decisions_rejected"], 4)
        self.assertEqual(row["controller_failures"], 1)
        self.assertEqual(row["invalid_commands"], 6)
        self.assertEqual(row["decision_latency_ms_p95"], 3.0)

    def test_model_policy_requires_an_explicit_verified_registry_entry(self):
        with self.assertRaises(ValueError):
            build_policy_spec("kev_original", {})
        with self.assertRaises(ValueError):
            build_policy_spec("kev_original", {"kev_original": {"base_url": "http://x"}})
        spec = build_policy_spec("kev_original", {
            "kev_original": {
                "base_url": "http://127.0.0.1:8009",
                "model": "kev-latest",
                "expected_run": "jaredpalmer/kev-0.8b",
                "label_source": "original_model",
            }
        })
        self.assertEqual(spec["name"], "kev_original")
        self.assertEqual(spec["label_source"], "original_model")
        self.assertEqual(spec["expected_run"], "jaredpalmer/kev-0.8b")

    def test_baselines_never_claim_a_model_identity(self):
        spec = build_policy_spec("rule_based", {})
        self.assertEqual(spec["kind"], "baseline")
        self.assertIsNone(spec.get("label_source"))

    def test_report_refuses_to_publish_rows_for_unloaded_checkpoints(self):
        rows = [{"policy": "kev_original", "model_loaded": "no", "model_id": ""}]
        with self.assertRaises(RuntimeError):
            assert_models_loaded(rows)
        rows = [{"policy": "kev_original", "model_loaded": "yes", "model_id": "jaredpalmer/kev-0.8b"}]
        assert_models_loaded(rows)

    def test_comparison_row_reports_model_identity_when_a_checkpoint_was_loaded(self):
        row = summarize_policy("kev_original", [{
            "scenario_seed": 5, "score": 10, "distance": 20, "survival_time": 1,
            "collision_count": 0, "termination_reason": "timeout", "final_lane": 1,
        }], {
            "applied_actions": {"STAY": 3},
            "decisions": 3,
            "model_id": "jaredpalmer/kev-0.8b",
            "model_alias": "kev-latest",
        })
        self.assertEqual(row["model_id"], "jaredpalmer/kev-0.8b")
        self.assertEqual(row["model_loaded"], "yes")
        self.assertEqual(row["model_latency_ms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
