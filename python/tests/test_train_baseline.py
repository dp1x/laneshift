import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_decisions import FEATURE_NAMES, feature_vector
from train_baseline import (
    ACTIONS,
    accuracy,
    brier_score,
    class_weight_values,
    class_weights,
    confusion_matrix,
    constant_probabilities,
    evaluate,
    expected_calibration_error,
    log_loss,
    macro_f1,
    main,
    one_hot,
    predict_proba_from_model,
    rows_to_arrays,
    train,
    validate_split_isolation,
)


def make_row(action="STAY", split="train", seed=1, same_lane_gap=200.0, left_gap=200.0, right_gap=200.0,
             lane=1, legal=("LEFT", "STAY", "RIGHT", "BRAKE"), episode="session:ep_1", termination="timeout"):
    row = {
        "row_id": f"{episode}:1",
        "episode_id": episode,
        "session_id": "session",
        "scenario_seed": seed,
        "split": split,
        "tick": 10,
        "sim_time": 0.2,
        "player_lane": lane,
        "player_target_lane": lane,
        "player_speed": 22.0,
        "lane_change_in_progress": 0,
        "lane_change_cooldown": 0.0,
        "legal_actions": "|".join(legal),
        "action": action,
        "label_source": "rule_based",
        "accepted": 1,
        "applied_action": action,
        "application_latency_ticks": 1,
    }
    for index in (1, 2, 3):
        row[f"v{index}_present"] = 0
        row[f"v{index}_lane"] = None
        row[f"v{index}_distance"] = None
        row[f"v{index}_rel_speed"] = None
        row[f"v{index}_ttc"] = None
        row[f"v{index}_in_player_lane"] = 0
    row["nearest_same_lane_gap"] = same_lane_gap
    row["nearest_left_gap"] = left_gap
    row["nearest_right_gap"] = right_gap
    row["min_ttc"] = None
    row["episode_score"] = 100.0
    row["episode_distance"] = 200.0
    row["episode_survival_time"] = 30.0
    row["episode_collisions"] = 0 if termination == "timeout" else 1
    row["episode_overtaken"] = 4
    row["episode_termination"] = termination
    row["episode_success"] = 1 if termination == "timeout" else 0
    return row


def synthetic_rows():
    """A deterministic rule: BRAKE under 18 m in the same lane, otherwise STAY."""
    rows = []
    seeds = {"train": (1, 2, 3, 4), "validation": (5, 6), "test": (7, 8)}
    for split, split_seeds in seeds.items():
        for seed in split_seeds:
            for step in range(12):
                gap = 5.0 + step * 4.0
                action = "BRAKE" if gap < 18.0 else "STAY"
                rows.append(make_row(action=action, split=split, seed=seed, same_lane_gap=gap,
                                     episode=f"session:ep_{seed}"))
    return rows


def parse_args(**overrides):
    values = {
        "decisions": None,
        "output_dir": None,
        "seed": 1234,
        "class_weights": "none",
        "hidden": "8,16",
        "l2": "0.001",
        "learning_rate": 0.02,
        "epochs": 120,
        "batch_size": 32,
        "patience": 20,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MetricTests(unittest.TestCase):
    def test_confusion_accuracy_and_macro_f1_match_hand_computation(self):
        labels = np.array([0, 0, 1, 1, 2])
        predictions = np.array([0, 0, 1, 2, 2])

        matrix = confusion_matrix(labels, predictions, n_classes=3)
        self.assertEqual(matrix.tolist(), [[2, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(accuracy(labels, predictions), 0.8)
        macro, per_class = macro_f1(labels, predictions, n_classes=3)
        self.assertAlmostEqual(macro, (1.0 + 2 / 3 + 2 / 3) / 3, places=6)
        self.assertAlmostEqual(per_class["STAY"]["recall"], 0.5)
        self.assertAlmostEqual(per_class["RIGHT"]["precision"], 0.5)

    def test_log_loss_matches_hand_computation(self):
        labels = np.array([0, 0])
        probabilities = np.array([[0.9, 0.1], [0.6, 0.4]])
        expected = -(np.log(0.9) + np.log(0.6)) / 2
        self.assertAlmostEqual(log_loss(labels, probabilities), expected, places=9)
        self.assertGreater(log_loss(labels, np.array([[0.5, 0.5], [0.5, 0.5]])), 0.0)

    def test_brier_matches_hand_computation(self):
        labels = np.array([0, 0])
        probabilities = np.array([[0.9, 0.1], [0.6, 0.4]])
        expected = (((0.9 - 1.0) ** 2 + 0.1 ** 2) + ((0.6 - 1.0) ** 2 + 0.4 ** 2)) / 2
        self.assertAlmostEqual(brier_score(labels, probabilities, n_classes=2), expected, places=9)

    def test_ece_matches_hand_computation_and_bins_by_confidence(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
            [0.6, 0.3, 0.05, 0.05],
            [0.6, 0.3, 0.05, 0.05],
        ])
        ece, table = expected_calibration_error(labels, probabilities)
        self.assertAlmostEqual(ece, 0.25, places=9)
        self.assertEqual(len(table), 2)
        self.assertEqual([row["count"] for row in table], [2, 2])

    def test_constant_probabilities_pick_the_requested_class(self):
        probabilities = constant_probabilities(ACTIONS.index("STAY"), 3)
        self.assertEqual(probabilities.shape, (3, len(ACTIONS)))
        self.assertTrue(np.all(probabilities[:, ACTIONS.index("STAY")] == 1.0))
        self.assertTrue(np.all(probabilities.sum(axis=1) == 1.0))

    def test_evaluate_reports_confusion_counts_and_per_class_support(self):
        labels = np.array([1, 1, 3])
        probabilities = np.array([
            [0.05, 0.9, 0.03, 0.02],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])
        metrics = evaluate(labels, probabilities)
        self.assertEqual(metrics["rows"], 3)
        self.assertEqual(metrics["confusion_matrix"], [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertNotIn("LEFT", metrics["per_class"])
        self.assertEqual(metrics["per_class"]["STAY"]["support"], 2)
        self.assertEqual(metrics["actual_counts"], {"LEFT": 0, "STAY": 2, "RIGHT": 0, "BRAKE": 1})


class TrainingTests(unittest.TestCase):
    def test_feature_matrix_has_one_row_per_decision_and_no_labels(self):
        rows = synthetic_rows()
        features, labels = rows_to_arrays(rows)
        self.assertEqual(features.shape, (len(rows), len(FEATURE_NAMES)))
        self.assertEqual(len(labels), len(rows))
        self.assertTrue(np.isfinite(features).all())

    def test_model_beats_the_majority_baseline_on_a_learnable_rule(self):
        rows = synthetic_rows()
        model, metrics = train(rows, parse_args())

        test_accuracy = metrics["splits"]["test"]["accuracy"]
        majority_accuracy = metrics["baselines"]["majority"]["splits"]["test"]["accuracy"]
        self.assertGreater(test_accuracy, majority_accuracy)
        self.assertGreater(test_accuracy, 0.9)
        self.assertGreater(metrics["splits"]["test"]["macro_f1"], 0.8)
        self.assertLess(metrics["splits"]["test"]["log_loss"], 0.5)
        self.assertEqual(metrics["splits"]["test"]["rows"], len([row for row in rows if row["split"] == "test"]))

    def test_training_is_deterministic_for_a_fixed_seed(self):
        first_model, first_metrics = train(synthetic_rows(), parse_args())
        second_model, second_metrics = train(synthetic_rows(), parse_args())
        for key in ("W1", "b1", "W2", "b2"):
            self.assertEqual(first_model["weights"][key], second_model["weights"][key])
        self.assertEqual(first_metrics["splits"], second_metrics["splits"])
        self.assertEqual(first_metrics["config"], second_metrics["config"])

    def test_episode_outcomes_do_not_change_the_trained_model(self):
        baseline_model, baseline_metrics = train(synthetic_rows(), parse_args())
        tampered = copy.deepcopy(synthetic_rows())
        for index, row in enumerate(tampered):
            row["episode_score"] = float(index * 1000)
            row["episode_distance"] = float(index)
            row["episode_survival_time"] = 0.0
            row["episode_collisions"] = index % 3
            row["episode_termination"] = "crash" if index % 2 else "timeout"
            row["episode_success"] = 0
        tampered_model, tampered_metrics = train(tampered, parse_args())
        for key in ("W1", "b1", "W2", "b2"):
            self.assertEqual(baseline_model["weights"][key], tampered_model["weights"][key])
        # Only the reporting metadata may differ; every prediction metric must be identical.
        summary = lambda metrics: {
            split: {key: values[key] for key in ("rows", "accuracy", "macro_f1", "log_loss", "brier", "ece")}
            for split, values in metrics["splits"].items()
        }
        self.assertEqual(summary(baseline_metrics), summary(tampered_metrics))

    def test_saved_model_reproduces_its_predictions(self):
        rows = synthetic_rows()
        model, _metrics = train(rows, parse_args())
        test_rows = [row for row in rows if row["split"] == "test"]
        features = np.asarray([feature_vector(row) for row in test_rows], dtype=np.float64)
        mean = np.asarray(model["scaler"]["mean"])
        std = np.asarray(model["scaler"]["std"])
        hidden = np.tanh(((features - mean) / std) @ np.asarray(model["weights"]["W1"]) + np.asarray(model["weights"]["b1"]))
        logits = hidden @ np.asarray(model["weights"]["W2"]) + np.asarray(model["weights"]["b2"])
        expected = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        actual = predict_proba_from_model(model, test_rows)
        self.assertTrue(np.allclose(expected, actual, atol=1e-9))

    def test_balanced_class_weights_reweight_the_rare_actions(self):
        labels = np.array([1, 1, 1, 1, 3])
        weights = class_weights(labels, "balanced")
        self.assertIsNone(class_weights(labels, "none"))
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 4.0, places=6)

    def test_one_hot_has_a_fixed_class_order(self):
        encoded = one_hot(np.array([ACTIONS.index("BRAKE")]))
        self.assertEqual(encoded.shape, (1, len(ACTIONS)))
        self.assertEqual(int(encoded.argmax()), ACTIONS.index("BRAKE"))

    def test_scaler_statistics_come_from_the_train_split_only(self):
        rows = [make_row(split="train", seed=1, same_lane_gap=100.0 + 10.0 * step, episode="s:ep_1")
                for step in range(6)]
        rows += [make_row(split="validation", seed=2, same_lane_gap=5.0 + step, episode="s:ep_2")
                 for step in range(6)]
        rows += [make_row(split="test", seed=3, same_lane_gap=5.0 + step, episode="s:ep_3")
                 for step in range(6)]

        model, _metrics = train(rows, parse_args(hidden="8", l2="0.001", epochs=20, patience=5))

        index = FEATURE_NAMES.index("nearest_same_lane_gap")
        train_mean = float(np.mean([feature_vector(row)[index] for row in rows if row["split"] == "train"]))
        all_mean = float(np.mean([feature_vector(row)[index] for row in rows]))
        self.assertAlmostEqual(model["scaler"]["mean"][index], train_mean, places=9)
        self.assertNotAlmostEqual(model["scaler"]["mean"][index], all_mean, places=3)

    def test_balanced_class_weights_come_from_train_labels_only(self):
        rows = [make_row(action="STAY", split="train", seed=1, episode="s:ep_1") for _ in range(10)]
        rows += [make_row(action="BRAKE", split="train", seed=1, episode="s:ep_1") for _ in range(2)]
        rows += [make_row(action="LEFT", split="validation", seed=2, episode="s:ep_2") for _ in range(4)]
        rows += [make_row(action="STAY", split="validation", seed=2, episode="s:ep_2")]
        rows += [make_row(action="LEFT", split="test", seed=3, episode="s:ep_3") for _ in range(4)]
        rows += [make_row(action="STAY", split="test", seed=3, episode="s:ep_3")]

        _model, metrics = train(rows, parse_args(class_weights="balanced", hidden="8", l2="0.001",
                                                 epochs=10, patience=5))

        values = metrics["config"]["class_weight_values"]
        train_labels = [ACTIONS.index(row["action"]) for row in rows if row["split"] == "train"]
        all_labels = [ACTIONS.index(row["action"]) for row in rows]
        self.assertEqual(values, class_weight_values(train_labels, "balanced"))
        self.assertNotEqual(values, class_weight_values(all_labels, "balanced"))
        self.assertAlmostEqual(values["STAY"], 0.6, places=6)   # 12 / (2 * 10)
        self.assertAlmostEqual(values["BRAKE"], 3.0, places=6)  # 12 / (2 * 2)
        self.assertEqual(values["LEFT"], 0.0)
        self.assertEqual(values["RIGHT"], 0.0)


class SplitIsolationTests(unittest.TestCase):
    @staticmethod
    def rows(*spec):
        return [make_row(action="STAY", split=split, seed=seed, episode=episode)
                for split, seed, episode in spec]

    def test_accepts_disjoint_splits(self):
        by_split = validate_split_isolation(self.rows(
            ("train", 1, "s:ep_1"), ("validation", 2, "s:ep_2"), ("test", 3, "s:ep_3"),
        ))
        self.assertEqual(len(by_split["validation"]), 1)

    def test_rejects_a_validation_test_episode_collision(self):
        with self.assertRaises(SystemExit) as caught:
            validate_split_isolation(self.rows(
                ("train", 1, "s:ep_1"), ("validation", 2, "s:ep_2"), ("test", 3, "s:ep_2"),
            ))
        self.assertIn("episode", str(caught.exception))

    def test_rejects_a_seed_shared_by_validation_and_test(self):
        with self.assertRaises(SystemExit) as caught:
            validate_split_isolation(self.rows(
                ("train", 1, "s:ep_1"), ("validation", 2, "s:ep_2"), ("test", 2, "s:ep_3"),
            ))
        self.assertIn("scenario seed", str(caught.exception))

    def test_main_refuses_a_leaky_decision_file(self):
        rows = self.rows(
            ("train", 1, "s:ep_1"), ("validation", 2, "s:ep_2"), ("test", 3, "s:ep_2"),
        )
        with tempfile.TemporaryDirectory() as directory:
            decisions = Path(directory) / "decisions.jsonl"
            decisions.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                main([
                    "--decisions", str(decisions),
                    "--output-dir", str(Path(directory) / "out"),
                    "--hidden", "8", "--l2", "0.001", "--epochs", "5",
                ])


if __name__ == "__main__":
    unittest.main()
