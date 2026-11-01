"""Train and evaluate a small supervised baseline on decision-level exports.

The model sees only the decision-state features defined by
`export_decisions.feature_vector`. Episode outcomes are evaluation labels, not
inputs. Hyperparameters are selected on validation seeds and the held-out test
seeds are touched once, at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from export_decisions import ACTIONS, FEATURE_NAMES, OUTCOME_COLUMNS, code_commit, feature_vector, file_sha256

SCHEMA_VERSION = 1
N_CLASSES = len(ACTIONS)


def runtime_environment():
    """Record what produced the numbers; floating-point reproduction is environment-bound."""
    blas = {}
    try:
        blas = dict(np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}) or {})
    except (AttributeError, TypeError, ValueError):
        blas = {}
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        },
        "thread_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    return rows


def rows_to_arrays(rows):
    if not rows:
        raise ValueError("no rows to convert")
    features = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    labels = np.asarray([ACTIONS.index(row["action"]) for row in rows], dtype=np.int64)
    return features, labels


def standardizer(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize(features, mean, std):
    return (features - mean) / std


def one_hot(labels, n_classes=N_CLASSES):
    encoded = np.zeros((len(labels), n_classes), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def confusion_matrix(labels, predictions, n_classes=N_CLASSES):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, predicted in zip(labels, predictions):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def accuracy(labels, predictions):
    if len(labels) == 0:
        return 0.0
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def macro_f1(labels, predictions, n_classes=N_CLASSES):
    """Macro F1 over classes that actually occur in `labels`."""
    matrix = confusion_matrix(labels, predictions, n_classes)
    scores = []
    per_class = {}
    for index in range(n_classes):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        support = float(matrix[index, :].sum())
        if support == 0.0:
            continue
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[ACTIONS[index]] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support)}
        scores.append(f1)
    return (float(np.mean(scores)) if scores else 0.0), per_class


def log_loss(labels, probabilities, epsilon=1e-12):
    if len(labels) == 0:
        return 0.0
    clipped = np.clip(probabilities, epsilon, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)])))


def brier_score(labels, probabilities, n_classes=N_CLASSES):
    """Mean over rows of the summed squared error against the one-hot target."""
    if len(labels) == 0:
        return 0.0
    return float(np.mean(np.sum((probabilities - one_hot(np.asarray(labels), n_classes)) ** 2, axis=1)))


def expected_calibration_error(labels, probabilities, n_classes=N_CLASSES, bins=10):
    """Top-1 ECE with equal-width confidence bins, plus the per-bin table."""
    if len(labels) == 0:
        return 0.0, []
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        upper_inclusive = index == bins - 1
        if upper_inclusive:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        weight = count / len(labels)
        ece += weight * abs(bin_accuracy - bin_confidence)
        table.append({
            "bin": [round(low, 2), round(high, 2)],
            "count": count,
            "confidence": round(bin_confidence, 4),
            "accuracy": round(bin_accuracy, 4),
        })
    return float(ece), table


def constant_probabilities(majority, rows, n_classes=N_CLASSES):
    """One-hot probabilities for a baseline that always predicts one class."""
    probabilities = np.zeros((rows, n_classes), dtype=np.float64)
    probabilities[:, int(majority)] = 1.0
    return probabilities


def evaluate(labels, probabilities, n_classes=N_CLASSES):
    predictions = probabilities.argmax(axis=1)
    macro, per_class = macro_f1(labels, predictions, n_classes)
    ece, bins = expected_calibration_error(labels, probabilities, n_classes)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
    predicted_counts = np.bincount(predictions, minlength=n_classes)
    return {
        "rows": int(len(labels)),
        "accuracy": round(accuracy(labels, predictions), 4),
        "macro_f1": round(macro, 4),
        "log_loss": round(log_loss(labels, probabilities), 4),
        "brier": round(brier_score(labels, probabilities, n_classes), 4),
        "ece": round(ece, 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4) if len(labels) else 0.0,
        "per_class": {name: {key: round(value, 4) if isinstance(value, float) else value
                             for key, value in stats.items()}
                      for name, stats in per_class.items()},
        "actual_counts": {ACTIONS[index]: int(counts[index]) for index in range(n_classes)},
        "predicted_counts": {ACTIONS[index]: int(predicted_counts[index]) for index in range(n_classes)},
        "confusion_matrix": confusion_matrix(labels, predictions, n_classes).tolist(),
        "calibration_bins": bins,
    }


class MLP:
    """One tanh hidden layer and a softmax output, trained with Adam."""

    def __init__(self, n_inputs, n_hidden, n_outputs, rng, learning_rate=0.02, l2=1e-4):
        self.rng = rng
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.params = {
            "W1": rng.normal(0.0, math.sqrt(1.0 / n_inputs), (n_inputs, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0.0, math.sqrt(1.0 / n_hidden), (n_hidden, n_outputs)),
            "b2": np.zeros(n_outputs),
        }
        self.moments = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.velocities = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step = 0

    def forward(self, features, params=None):
        params = self.params if params is None else params
        hidden = np.tanh(features @ params["W1"] + params["b1"])
        logits = hidden @ params["W2"] + params["b2"]
        return hidden, softmax(logits)

    def predict_proba(self, features, params=None):
        return self.forward(features, params)[1]

    def _gradients(self, features, hidden, probabilities, targets, weights, params):
        error = (probabilities - targets) * weights[:, None] / max(1.0, float(weights.sum()))
        grad_w2 = hidden.T @ error + self.l2 * params["W2"]
        grad_b2 = error.sum(axis=0)
        hidden_error = (error @ params["W2"].T) * (1.0 - hidden ** 2)
        grad_w1 = features.T @ hidden_error + self.l2 * params["W1"]
        grad_b1 = hidden_error.sum(axis=0)
        return {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}

    def fit(self, features, labels, sample_weights=None, validation=None, epochs=300,
            batch_size=32, patience=30):
        targets = one_hot(labels)
        weights = (np.ones(len(labels), dtype=np.float64) if sample_weights is None
                   else np.asarray(sample_weights, dtype=np.float64))
        best_params = {key: value.copy() for key, value in self.params.items()}
        best_loss = float("inf")
        best_epoch = 0
        history = []
        stale = 0
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(labels))
            for start in range(0, len(labels), batch_size):
                batch = order[start:start + batch_size]
                hidden, probabilities = self.forward(features[batch])
                gradients = self._gradients(features[batch], hidden, probabilities, targets[batch],
                                            weights[batch], self.params)
                self.step += 1
                for key in self.params:
                    self.moments[key] = 0.9 * self.moments[key] + 0.1 * gradients[key]
                    self.velocities[key] = 0.999 * self.velocities[key] + 0.001 * gradients[key] ** 2
                    m_hat = self.moments[key] / (1.0 - 0.9 ** self.step)
                    v_hat = self.velocities[key] / (1.0 - 0.999 ** self.step)
                    self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            train_loss = log_loss(labels, self.predict_proba(features))
            record = {"epoch": epoch, "train_log_loss": round(train_loss, 6)}
            if validation is not None:
                validation_features, validation_labels = validation
                # Selection uses the unweighted mean loss so calibration stays comparable
                # between class-weighted and unweighted models.
                validation_loss = log_loss(validation_labels, self.predict_proba(validation_features))
                record["validation_log_loss"] = round(validation_loss, 6)
                history.append(record)
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_params = {key: value.copy() for key, value in self.params.items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            else:
                history.append(record)
        if validation is not None:
            self.params = best_params
        return {"history": history, "best_epoch": best_epoch, "best_validation_log_loss": best_loss}


def class_weight_table(labels, n_classes=N_CLASSES):
    """Inverse-frequency weights from `labels`; callers must pass training labels only."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


def class_weights(labels, mode, n_classes=N_CLASSES):
    if mode == "none":
        return None
    return class_weight_table(labels, n_classes)[np.asarray(labels, dtype=np.int64)]


def class_weight_values(labels, mode, n_classes=N_CLASSES):
    """The per-class weights as recorded in the metrics, for auditability."""
    if mode == "none":
        return None
    table = class_weight_table(labels, n_classes)
    return {ACTIONS[index]: round(float(table[index]), 6) for index in range(n_classes)}


def summarize_by_seed(rows, probabilities):
    """Compact per-seed view: held-out seeds are the unit that matters, rows are not independent."""
    summary = {}
    seeds = sorted({row["scenario_seed"] for row in rows})
    for seed in seeds:
        indices = [index for index, row in enumerate(rows) if row["scenario_seed"] == seed]
        labels = np.asarray([ACTIONS.index(rows[index]["action"]) for index in indices], dtype=np.int64)
        seed_probabilities = probabilities[indices]
        predictions = seed_probabilities.argmax(axis=1)
        macro, _per_class = macro_f1(labels, predictions)
        summary[str(seed)] = {
            "rows": len(indices),
            "episodes": len({rows[index]["episode_id"] for index in indices}),
            "accuracy": round(accuracy(labels, predictions), 4),
            "macro_f1": round(macro, 4),
            "log_loss": round(log_loss(labels, seed_probabilities), 4),
            "termination": sorted({str(rows[index].get("episode_termination")) for index in indices}),
        }
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True, help="decision JSONL produced by export_decisions.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="none")
    parser.add_argument("--hidden", default="8,16,32", help="comma-separated hidden sizes to try")
    parser.add_argument("--l2", default="0.0001,0.001", help="comma-separated L2 values to try")
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args(argv)


def _split(rows, name):
    return [row for row in rows if row.get("split") == name]


def validate_split_isolation(rows):
    """Refuse to train unless every pair of splits is disjoint in seeds and episodes.

    The exporter already guarantees this; the trainer re-checks because the
    decision file is an external input that may have been edited or produced by
    an older exporter.
    """
    by_split = {name: _split(rows, name) for name in ("train", "validation", "test")}
    for name, split_rows in by_split.items():
        if not split_rows:
            raise SystemExit(f"split {name!r} has no rows")
    for key, label in (("scenario_seed", "scenario seed"), ("episode_id", "episode")):
        owner = {}
        for name in ("train", "validation", "test"):
            for value in {row.get(key) for row in by_split[name] if row.get(key) is not None}:
                if value in owner:
                    raise SystemExit(f"{label} {value!r} appears in both {owner[value]} and {name}")
                owner[value] = name
    return by_split


def _distribution(rows):
    return {action: sum(1 for row in rows if row["action"] == action) for action in ACTIONS}


def _dataset_identity(args):
    path = getattr(args, "decisions", None)
    if path and Path(path).is_file():
        return {"path": str(path), "decisions_sha256": file_sha256(path)}
    return {"path": str(path) if path else None, "decisions_sha256": None}


def train(rows, args):
    splits = {name: _split(rows, name) for name in ("train", "validation", "test")}
    for name, split_rows in splits.items():
        if not split_rows:
            raise SystemExit(f"split {name!r} has no rows")
    identity = _dataset_identity(args)
    train_features, train_labels = rows_to_arrays(splits["train"])
    validation_features, validation_labels = rows_to_arrays(splits["validation"])
    mean, std = standardizer(train_features)
    train_features = standardize(train_features, mean, std)
    validation_features = standardize(validation_features, mean, std)
    train_weights = class_weights(train_labels, args.class_weights)

    hidden_sizes = [int(value) for value in args.hidden.split(",") if value.strip()]
    l2_values = [float(value) for value in args.l2.split(",") if value.strip()]
    grid = []
    best = None
    for hidden in hidden_sizes:
        for l2 in l2_values:
            rng = np.random.default_rng(args.seed)
            model = MLP(train_features.shape[1], hidden, N_CLASSES, rng, args.learning_rate, l2)
            fit = model.fit(
                train_features, train_labels,
                sample_weights=train_weights,
                validation=(validation_features, validation_labels),
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            entry = {
                "hidden": hidden,
                "l2": l2,
                "learning_rate": args.learning_rate,
                "best_epoch": fit["best_epoch"],
                "validation_log_loss": round(fit["best_validation_log_loss"], 6),
                "epochs_run": len(fit["history"]),
            }
            grid.append(entry)
            if best is None or entry["validation_log_loss"] < best["entry"]["validation_log_loss"]:
                best = {"entry": entry, "model": model, "fit": fit}

    model = best["model"]
    split_metrics = {}
    split_labels = {}
    for name, split_rows in splits.items():
        features, labels = rows_to_arrays(split_rows)
        probabilities = model.predict_proba(standardize(features, mean, std))
        split_labels[name] = labels
        split_metrics[name] = evaluate(labels, probabilities)
        split_metrics[name]["per_seed"] = summarize_by_seed(split_rows, probabilities)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": runtime_environment(),
        "code": code_commit(),
        "determinism_note": (
            "Weights and metrics reproduce exactly only on the same Python, NumPy and BLAS build "
            "with the same thread settings; other builds can change floating-point summation order."
        ),
        "dataset": {
            "rows": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "path": identity["path"],
            "decisions_sha256": identity["decisions_sha256"],
            "label_sources": sorted({str(row.get("label_source")) for row in rows}),
            "episode_outcomes": dict(Counter(
                str(row["episode_termination"]) for row in {
                    row["episode_id"]: row for row in rows
                }.values()
            )),
            "crashes": sum(1 for row in {
                row["episode_id"]: row for row in rows
            }.values() if row.get("episode_termination") == "crash"),
            "splits": {
                name: {
                    "rows": len(split_rows),
                    "episodes": len({row["episode_id"] for row in split_rows}),
                    "scenario_seeds": sorted({row["scenario_seed"] for row in split_rows}),
                    "action_counts": _distribution(split_rows),
                }
                for name, split_rows in splits.items()
            },
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "count": len(FEATURE_NAMES),
            "excluded_outcome_columns": list(OUTCOME_COLUMNS),
        },
        "config": {
            "hidden": best["entry"]["hidden"],
            "hidden_units": best["entry"]["hidden"],
            "l2": best["entry"]["l2"],
            "learning_rate": best["entry"]["learning_rate"],
            "batch_size": args.batch_size,
            "class_weights": args.class_weights,
            "class_weight_values": class_weight_values(train_labels, args.class_weights),
            "seed": args.seed,
            "best_epoch": best["entry"]["best_epoch"],
            "epochs_run": best["entry"]["epochs_run"],
            "selection_metric": "validation log loss (unweighted)",
        },
        "grid": grid,
        "splits": split_metrics,
    }

    train_majority = int(np.bincount(train_labels, minlength=N_CLASSES).argmax())
    stay_index = ACTIONS.index("STAY")
    baselines = {}
    for name, index in (("majority", train_majority), ("stay", stay_index)):
        if name == "stay" and index == train_majority:
            continue
        baselines[name] = {
            "class": ACTIONS[index],
            "splits": {
                split_name: evaluate(
                    split_labels[split_name],
                    constant_probabilities(index, len(split_labels[split_name])),
                )
                for split_name in splits
            },
        }
    metrics["baselines"] = baselines

    model_document = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(ACTIONS),
        "config": metrics["config"],
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "weights": {key: value.tolist() for key, value in model.params.items()},
        "dataset_path": identity["path"],
        "decisions_sha256": identity["decisions_sha256"],
        "code": metrics["code"],
        "environment": metrics["environment"],
        "train_scenario_seeds": metrics["dataset"]["splits"]["train"]["scenario_seeds"],
        "validation_scenario_seeds": metrics["dataset"]["splits"]["validation"]["scenario_seeds"],
        "test_scenario_seeds": metrics["dataset"]["splits"]["test"]["scenario_seeds"],
    }
    return model_document, metrics


def predict_proba_from_model(model, rows):
    """Reproduce predictions from a saved model document."""
    features = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    mean = np.asarray(model["scaler"]["mean"], dtype=np.float64)
    std = np.asarray(model["scaler"]["std"], dtype=np.float64)
    features = (features - mean) / std
    hidden = np.tanh(features @ np.asarray(model["weights"]["W1"]) + np.asarray(model["weights"]["b1"]))
    logits = hidden @ np.asarray(model["weights"]["W2"]) + np.asarray(model["weights"]["b2"])
    return softmax(logits)


def format_report(metrics):
    lines = []
    lines.append("# Supervised baseline pilot")
    lines.append("")
    dataset = metrics["dataset"]
    lines.append(f"Generated by `python/train_baseline.py`. Label sources: {', '.join(dataset['label_sources'])}.")
    if "human" not in dataset["label_sources"]:
        lines.append("The labels are policy actions, not human driving decisions.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- rows: {dataset['rows']}, episodes: {dataset['episodes']}, endings {dataset['episode_outcomes']}")
    if dataset.get("decisions_sha256"):
        lines.append(f"- decisions sha256: {dataset['decisions_sha256']}")
    code = metrics.get("code") or {}
    if code.get("commit"):
        lines.append(f"- code commit: {code['commit']} (dirty: {code.get('dirty')})")
    for name, split in dataset["splits"].items():
        lines.append(f"- {name}: {split['rows']} rows, {split['episodes']} episodes, seeds {split['scenario_seeds']}, actions {split['action_counts']}")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    config = metrics["config"]
    lines.append(f"- MLP {metrics['features']['count']} -> {config['hidden']} (tanh) -> {N_CLASSES} (softmax)")
    lines.append(
        f"- learning rate {config['learning_rate']}, L2 {config['l2']}, batch {config['batch_size']}, "
        f"class weights {config['class_weights']}"
    )
    lines.append(f"- selected on validation log loss at epoch {config['best_epoch']}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    header = "| split | model | rows | accuracy | macro F1 | log loss | Brier | ECE |"
    lines.append(header)
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ("train", "validation", "test"):
        split = metrics["splits"][name]
        lines.append(
            f"| {name} | MLP | {split['rows']} | {split['accuracy']:.4f} | {split['macro_f1']:.4f} | "
            f"{split['log_loss']:.4f} | {split['brier']:.4f} | {split['ece']:.4f} |"
        )
        for baseline_name, baseline in metrics["baselines"].items():
            baseline_split = baseline["splits"][name]
            lines.append(
                f"| {name} | {baseline_name} ({baseline['class']}) | {baseline_split['rows']} | "
                f"{baseline_split['accuracy']:.4f} | {baseline_split['macro_f1']:.4f} | "
                f"{baseline_split['log_loss']:.4f} | {baseline_split['brier']:.4f} | {baseline_split['ece']:.4f} |"
            )
    lines.append("")
    lines.append("## Held-out seeds")
    lines.append("")
    lines.append("| split | seed | rows | accuracy | macro F1 | log loss | episode ending |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for name in ("validation", "test"):
        for seed, seed_metrics in metrics["splits"][name]["per_seed"].items():
            lines.append(
                f"| {name} | {seed} | {seed_metrics['rows']} | {seed_metrics['accuracy']:.4f} | "
                f"{seed_metrics['macro_f1']:.4f} | {seed_metrics['log_loss']:.4f} | "
                f"{', '.join(seed_metrics['termination'])} |"
            )
    lines.append("")
    lines.append("## Test confusion matrix")
    lines.append("")
    lines.append("| true \\ predicted | " + " | ".join(ACTIONS) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in ACTIONS) + " |")
    for index, action in enumerate(ACTIONS):
        row = metrics["splits"]["test"]["confusion_matrix"][index]
        lines.append(f"| {action} | " + " | ".join(str(value) for value in row) + " |")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    environment = metrics.get("environment") or {}
    blas = environment.get("blas") or {}
    lines.append(
        f"- python {environment.get('python')} ({environment.get('python_implementation')}), "
        f"numpy {environment.get('numpy')}, {environment.get('platform')}, {environment.get('machine')}"
    )
    lines.append(f"- BLAS {blas.get('name')} {blas.get('version')}, cpu_count {environment.get('cpu_count')}")
    lines.append(f"- thread env {environment.get('thread_env')}")
    lines.append(
        "- Weights and metrics reproduce exactly only on this Python/NumPy/BLAS build and thread setup; "
        "other builds can change floating-point summation order."
    )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append("- The target is a deterministic rule, so a high score measures imitation, not driving skill.")
    lines.append("- Rows inside an episode are 0.1 s apart; the effective sample size is far below the row count.")
    lines.append("- No probabilities are recorded for rule-based labels, so calibration is of this MLP only.")
    lines.append("- A human-labelled dataset and a held-out human session are still required before fine-tuning Kev.")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.decisions)
    if not rows:
        raise SystemExit(f"no decision rows in {args.decisions}")
    unknown = {row.get("split") for row in rows} - {"train", "validation", "test"}
    if unknown:
        raise SystemExit(f"decision rows contain unsupported split labels: {sorted(map(str, unknown))}")
    validate_split_isolation(rows)
    model_document, metrics = train(rows, args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(model_document, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(format_report(metrics), encoding="utf-8")
    summary = {
        "output_dir": str(output),
        "selected": metrics["config"],
        "test": metrics["splits"]["test"],
        "baselines": {name: value["splits"]["test"] for name, value in metrics["baselines"].items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
