import unittest

from experiment_runner import summarize_policy


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


if __name__ == "__main__":
    unittest.main()
