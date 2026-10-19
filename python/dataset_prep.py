"""Join LaneShift JSONL logs into leakage-safe decision candidates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kev_state import ACTION_CRITERIA, compact_kev_state

ALLOWED_ACTIONS = {"LEFT", "STAY", "RIGHT", "BRAKE"}
STATE_FEATURES = (
    "protocol_version",
    "episode_id",
    "scenario_seed",
    "sim_time",
    "physics_tick",
    "player",
    "traffic",
    "legal_actions",
    "score",
    "distance",
    "survival_time",
    "control_source",
)
SOURCE_ALIASES = {
    "human": "human",
    "rule": "rule_based",
    "rule_based": "rule_based",
    "external": "external_ai",
    "external_ai": "external_ai",
    "original_model": "original_model",
    "fine_tuned_model": "fine_tuned_model",
    "synthetic": "synthetic",
    "dummy": "synthetic",
    "controller_fallback": "controller_fallback",
}
TRAINABLE_SOURCES = {"human", "rule_based", "original_model", "fine_tuned_model", "synthetic"}
KEV_INPUT_FIELDS = ("sim_time", "player", "traffic", "legal_actions", "score", "distance", "survival_time")
KEV_ACTION_CRITERIA = ACTION_CRITERIA
REQUIRED_EPISODE_FIELDS = {
    "episode_id", "scenario_seed", "score", "distance",
    "survival_time", "collision_count", "termination_reason",
}


def read_jsonl(path):
    records = []
    malformed = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(value, dict):
                malformed.append({"line": line_number, "error": "record must be a JSON object"})
                continue
            records.append(value)
    return records, malformed


def _state_is_usable(state, episode_id):
    return (
        isinstance(state, dict)
        and state.get("episode_id") == episode_id
        and isinstance(state.get("physics_tick"), int)
        and isinstance(state.get("player"), dict)
        and isinstance(state.get("traffic"), list)
    )


def _pre_action_state(event, snapshots, episode_id, decision_tick):
    embedded = event.get("state_before")
    if (
        _state_is_usable(embedded, episode_id)
        and embedded["physics_tick"] <= decision_tick
    ):
        return embedded
    preceding = [
        row for row in snapshots
        if row.get("physics_tick", -1) <= decision_tick
        and _state_is_usable(row, episode_id)
    ]
    if not preceding:
        return None
    return max(preceding, key=lambda row: row["physics_tick"])


def prepare_records(telemetry_records, event_records, episode_records):
    snapshots_by_episode = defaultdict(list)
    episodes_by_id = {}
    report = {
        "telemetry_records": 0,
        "action_events": 0,
        "candidate_count": 0,
        "rejected_actions": 0,
        "incomplete_actions": 0,
        "malformed_records": 0,
        "episodes_without_summary": 0,
        "illegal_labels": 0,
        "duplicate_actions": 0,
        "illegal_labels": 0,
    }

    for row in telemetry_records:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("episode_id"), str)
            or not _state_is_usable(row, row.get("episode_id"))
        ):
            report["malformed_records"] += 1
            continue
        snapshots_by_episode[row["episode_id"]].append(row)
        report["telemetry_records"] += 1
    for rows in snapshots_by_episode.values():
        rows.sort(key=lambda row: row["physics_tick"])

    for row in episode_records:
        if (
            isinstance(row, dict)
            and isinstance(row.get("episode_id"), str)
            and REQUIRED_EPISODE_FIELDS.issubset(row)
        ):
            episodes_by_id[row["episode_id"]] = row
        else:
            report["malformed_records"] += 1

    candidates = []
    seen_example_ids = set()
    for event_index, event in enumerate(event_records, 1):
        if not isinstance(event, dict):
            report["malformed_records"] += 1
            continue
        if event.get("type") != "action_decision":
            continue
        report["action_events"] += 1
        if "accepted" not in event or "applied_action" not in event:
            report["incomplete_actions"] += 1
            continue
        episode_id = event.get("episode_id")
        decision_tick = event.get("decision_tick", event.get("physics_tick"))
        application_tick = event.get("application_tick", event.get("physics_tick", decision_tick))
        action = event.get("applied_action")
        if event.get("accepted") is not True or action not in ALLOWED_ACTIONS:
            report["rejected_actions"] += 1
            continue
        if (
            not isinstance(episode_id, str)
            or not isinstance(decision_tick, int)
            or decision_tick < 0
            or not isinstance(application_tick, int)
            or application_tick < decision_tick
        ):
            report["incomplete_actions"] += 1
            continue
        before = _pre_action_state(
            event, snapshots_by_episode.get(episode_id, []), episode_id, decision_tick
        )
        if before is None:
            report["incomplete_actions"] += 1
            continue
        legal_actions = before.get("legal_actions")
        if not isinstance(legal_actions, list) or action not in legal_actions:
            report["illegal_labels"] += 1
            continue
        source = event.get("label_source") or event.get("control_source")
        if not isinstance(source, str) or not source:
            report["incomplete_actions"] += 1
            continue
        label_source = SOURCE_ALIASES.get(str(source), str(source))
        outcome = episodes_by_id.get(episode_id)
        if outcome is None:
            report["episodes_without_summary"] += 1
        # Only pre-decision fields enter `state`; outcomes remain a separate target.
        features = {key: before[key] for key in STATE_FEATURES if key in before}
        if "control_source" not in features:
            features["control_source"] = event.get("control_source", "unknown")
        decision_seq = event.get("decision_seq", event.get("seq", event_index))
        example_id = str(event.get("event_id", "%s:action:%s" % (episode_id, decision_seq)))
        if example_id in seen_example_ids:
            report["duplicate_actions"] += 1
            continue
        seen_example_ids.add(example_id)
        candidates.append({
            "record_type": "decision_candidate",
            "protocol_version": 1,
            "example_id": example_id,
            "episode_id": episode_id,
            "scenario_seed": before.get("scenario_seed", event.get("scenario_seed")),
            "decision_seq": decision_seq,
            "decision_tick": decision_tick,
            "application_tick": application_tick,
            "state_id": before.get("state_id", f"{episode_id}:{before['physics_tick']}"),
            "state": features,
            "requested_action": event.get("requested_action", action),
            "action": action,
            "accepted": True,
            "control_source": event.get("control_source", "unknown"),
            "label_source": label_source,
            "decision_metadata": event.get("decision_metadata", {}),
            "outcome": outcome,
            "successful_episode": bool(
                outcome and outcome.get("termination_reason") in {"timeout", "completed", "episode_complete"}
            ),
        })
    report["candidate_count"] = len(candidates)
    report["action_counts"] = dict(Counter(row["action"] for row in candidates))
    report["label_source_counts"] = dict(Counter(row["label_source"] for row in candidates))
    report["episode_count"] = len({row["episode_id"] for row in candidates})
    report["scenario_seed_count"] = len({row["scenario_seed"] for row in candidates if row.get("scenario_seed") is not None})
    return candidates, report


def split_by_episode(records, eval_fraction=0.2, seed=0):
    """Split by scenario seed, keeping every episode for a seed together."""
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    group_keys = []
    for index, row in enumerate(records):
        scenario_seed = row.get("scenario_seed")
        if scenario_seed is None and isinstance(row.get("state"), dict):
            scenario_seed = row["state"].get("scenario_seed")
        if scenario_seed is not None:
            group_keys.append(("scenario", str(scenario_seed)))
        elif row.get("episode_id") is not None:
            group_keys.append(("episode", str(row["episode_id"])))
        else:
            group_keys.append(("record", str(index)))
    shuffled = sorted(set(group_keys))
    random.Random(seed).shuffle(shuffled)
    if eval_fraction == 0.0 or len(shuffled) < 2:
        eval_groups = set()
    else:
        eval_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * eval_fraction)))
        eval_groups = set(shuffled[:eval_count])
    return {
        "train": [row for row, key in zip(records, group_keys) if key not in eval_groups],
        "eval": [row for row, key in zip(records, group_keys) if key in eval_groups],
    }


def split_by_scenario(records, validation_fraction=0.1, test_fraction=0.1, seed=0):
    """Create seed-disjoint train/validation/test sets; require at least 3 scenario groups."""
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must each be in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1")
    group_keys = []
    for index, row in enumerate(records):
        scenario_seed = row.get("scenario_seed")
        if scenario_seed is None and isinstance(row.get("state"), dict):
            scenario_seed = row["state"].get("scenario_seed")
        if scenario_seed is not None:
            group_keys.append(("scenario", str(scenario_seed)))
        elif row.get("episode_id") is not None:
            group_keys.append(("episode", str(row["episode_id"])))
        else:
            group_keys.append(("record", str(index)))
    groups = sorted(set(group_keys))
    if len(groups) < 3:
        raise ValueError("at least 3 distinct scenario-seed groups are required for train/validation/test splits")
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(groups) * test_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    while test_count + validation_count >= len(groups):
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("not enough distinct scenario seeds for non-empty train/validation/test splits")
    test_groups = set(shuffled[:test_count])
    validation_groups = set(shuffled[test_count:test_count + validation_count])
    result = {"train": [], "validation": [], "test": []}
    for record, key in zip(records, group_keys):
        name = "test" if key in test_groups else "validation" if key in validation_groups else "train"
        result[name].append(record)
    return result


def make_kev_training_record(candidate):
    """Convert an explicitly approved candidate into Kev's labelled choice schema."""
    state = candidate.get("state")
    action = candidate.get("action")
    if not isinstance(state, dict) or action not in ALLOWED_ACTIONS:
        raise ValueError("candidate is missing a valid state or action label")
    legal_actions = state.get("legal_actions")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or any(value not in KEV_ACTION_CRITERIA for value in legal_actions)
        or len(set(legal_actions)) != len(legal_actions)
        or action not in legal_actions
    ):
        raise ValueError("candidate label must be one of its unique, valid legal_actions")
    # The training state is the same projection the model sees at inference time.
    training_state = compact_kev_state(state)
    return {
        "state": training_state,
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": "Choose the safest useful action. Only choose a currently legal option.",
                "criteria": {choice: KEV_ACTION_CRITERIA[choice] for choice in legal_actions},
                "label": action,
            }
        },
    }


def export_kev_examples(candidates, approval_records, include_sources):
    """Export only explicitly approved labels from explicitly selected sources."""
    allowed_sources = set(include_sources)
    if not allowed_sources or not allowed_sources.issubset(TRAINABLE_SOURCES):
        raise ValueError(f"select one or more training sources from: {', '.join(sorted(TRAINABLE_SOURCES))}")
    report = {
        "approved_count": 0,
        "unapproved": 0,
        "excluded_source": 0,
        "invalid_approvals": 0,
        "duplicate_example_ids": 0,
        "invalid_candidates": 0,
        "action_counts": {},
        "label_source_counts": {},
        "episode_count": 0,
        "scenario_seed_count": 0,
    }
    approvals = {}
    invalid_approval_ids = set()
    for approval in approval_records:
        example_id = approval.get("example_id") if isinstance(approval, dict) else None
        if not isinstance(example_id, str) or not example_id or example_id in approvals or example_id in invalid_approval_ids:
            report["invalid_approvals"] += 1
            if isinstance(example_id, str) and example_id:
                invalid_approval_ids.add(example_id)
                approvals.pop(example_id, None)
            continue
        approvals[example_id] = approval

    training_rows = []
    provenance_rows = []
    seen_ids = set()
    for candidate in candidates:
        example_id = candidate.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
            report["duplicate_example_ids"] += 1
            continue
        seen_ids.add(example_id)
        source = candidate.get("label_source")
        if source not in allowed_sources:
            report["excluded_source"] += 1
            continue
        approval = approvals.get(example_id)
        if approval is None or approval.get("approved") is not True:
            report["unapproved"] += 1
            continue
        action = candidate.get("action")
        if approval.get("action") != action:
            report["invalid_approvals"] += 1
            continue
        try:
            training = make_kev_training_record(candidate)
        except (TypeError, ValueError):
            report["invalid_candidates"] += 1
            continue
        training_rows.append(training)
        provenance_rows.append({
            "example_id": example_id,
            "episode_id": candidate.get("episode_id"),
            "scenario_seed": candidate.get("scenario_seed"),
            "decision_seq": candidate.get("decision_seq"),
            "decision_tick": candidate.get("decision_tick"),
            "application_tick": candidate.get("application_tick"),
            "state_id": candidate.get("state_id"),
            "action": action,
            "label_source": source,
            "approval": approval,
            "outcome": candidate.get("outcome"),
            "successful_episode": candidate.get("successful_episode", False),
            "decision_metadata": candidate.get("decision_metadata", {}),
        })

    report["approved_count"] = len(training_rows)
    report["action_counts"] = dict(Counter(row["questions"]["next_action"]["label"] for row in training_rows))
    report["label_source_counts"] = dict(Counter(row["label_source"] for row in provenance_rows))
    report["episode_count"] = len({row["episode_id"] for row in provenance_rows})
    report["scenario_seed_count"] = len({row["scenario_seed"] for row in provenance_rows if row.get("scenario_seed") is not None})
    return training_rows, provenance_rows, report


def _write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _distribution(rows, key):
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))


def _write_kev_splits(output_path, training_rows, provenance_rows, validation_fraction, test_fraction, split_seed):
    grouped = [
        {
            "episode_id": provenance["episode_id"],
            "scenario_seed": provenance["scenario_seed"],
            "training": training,
            "provenance": provenance,
        }
        for training, provenance in zip(training_rows, provenance_rows)
    ]
    splits = split_by_scenario(grouped, validation_fraction, test_fraction, split_seed)
    base = output_path.with_suffix("")
    split_files = {}
    for name, records in splits.items():
        training_path = base.with_name(base.name + f".{name}.jsonl")
        provenance_path = base.with_name(base.name + f".{name}.provenance.jsonl")
        training = [row["training"] for row in records]
        provenance = [row["provenance"] for row in records]
        _write_jsonl(training_path, training)
        _write_jsonl(provenance_path, provenance)
        split_files[name] = {
            "training_file": str(training_path),
            "provenance_file": str(provenance_path),
            "records": len(training),
            "action_counts": _distribution(
                [{"action": row["questions"]["next_action"]["label"]} for row in training],
                "action",
            ),
            "source_counts": _distribution(provenance, "label_source"),
            "episode_count": len({row["episode_id"] for row in provenance}),
            "scenario_seeds": sorted({row["scenario_seed"] for row in provenance if row.get("scenario_seed") is not None}),
        }
    return split_files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("candidates", "kev"), default="candidates")
    parser.add_argument("--successful-only", action="store_true", help="filter exported candidates, retaining the report of all candidates")
    parser.add_argument("--approved-records", type=Path, help="review JSONL with example_id, action, approved=true")
    parser.add_argument("--include-sources", default="", help="comma-separated label sources explicitly approved for training")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    telemetry, bad_telemetry = read_jsonl(args.telemetry)
    events, bad_events = read_jsonl(args.events)
    episodes, bad_episodes = read_jsonl(args.episodes)
    candidates, report = prepare_records(telemetry, events, episodes)
    report["malformed_lines"] = {
        "telemetry": bad_telemetry,
        "events": bad_events,
        "episodes": bad_episodes,
    }
    report["malformed_records"] += sum(map(len, report["malformed_lines"].values()))
    exported = [row for row in candidates if row["successful_episode"]] if args.successful_only else candidates
    _write_jsonl(args.output, exported)
    if args.format == "candidates" and args.eval_fraction:
        split = split_by_episode(exported, args.eval_fraction, args.split_seed)
        _write_jsonl(args.output.with_name(args.output.stem + ".train.jsonl"), split["train"])
        _write_jsonl(args.output.with_name(args.output.stem + ".eval.jsonl"), split["eval"])
        report["split"] = {
            "train": len(split["train"]),
            "eval": len(split["eval"]),
            "episode_disjoint": True,
            "scenario_seed_disjoint": True,
        }
    if args.format == "kev":
        if args.approved_records is None:
            raise SystemExit("--format kev requires --approved-records; actions are never auto-labelled as training data")
        include_sources = [source.strip() for source in args.include_sources.split(",") if source.strip()]
        if not include_sources:
            raise SystemExit("--format kev requires an explicit --include-sources allowlist")
        approvals, malformed_approvals = read_jsonl(args.approved_records)
        training_rows, provenance_rows, export_report = export_kev_examples(exported, approvals, include_sources)
        export_report["malformed_approval_lines"] = malformed_approvals
        if malformed_approvals:
            export_report["invalid_approvals"] += len(malformed_approvals)
        report["kev_export"] = export_report
        if not training_rows:
            raise SystemExit("no approved, valid Kev training labels; candidate and report files were still written")
        # The primary output holds Kev-format training rows; raw logs are never modified.
        _write_jsonl(args.output, training_rows)
        provenance_path = args.output.with_name(args.output.stem + ".provenance.jsonl")
        _write_jsonl(provenance_path, provenance_rows)
        try:
            split_files = _write_kev_splits(
                args.output,
                training_rows,
                provenance_rows,
                args.validation_fraction,
                args.test_fraction,
                args.split_seed,
            )
            split_error = None
        except ValueError as exc:
            split_files = {}
            split_error = str(exc)
        manifest = {
            "format": "kev-systemone-labelled-choice-jsonl-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_inputs": {"telemetry": str(args.telemetry), "events": str(args.events), "episodes": str(args.episodes)},
            "approval_file": str(args.approved_records),
            "included_label_sources": include_sources,
            "split_policy": "whole scenario_seed groups; no seed overlap between train, validation and test",
            "split_seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "total_approved_records": len(training_rows),
            "total_action_counts": _distribution(
                [{"action": row["questions"]["next_action"]["label"]} for row in training_rows], "action"
            ),
            "total_source_counts": _distribution(provenance_rows, "label_source"),
            "provenance_file": str(provenance_path),
            "splits": split_files,
            "split_error": split_error,
            "quality_note": "Examples are included only when explicitly approved in the supplied review JSONL.",
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report["kev_manifest"] = str(manifest_path)
        if split_error:
            print(json.dumps({"warning": split_error, "manifest": str(manifest_path)}, indent=2))
    else:
        _write_jsonl(args.output, exported)
    if args.format == "candidates" and args.eval_fraction:
        split = split_by_episode(exported, args.eval_fraction, args.split_seed)
        _write_jsonl(args.output.with_name(args.output.stem + ".train.jsonl"), split["train"])
        _write_jsonl(args.output.with_name(args.output.stem + ".eval.jsonl"), split["eval"])
        report["split"] = {
            "train": len(split["train"]),
            "eval": len(split["eval"]),
            "episode_disjoint": True,
            "scenario_seed_disjoint": True,
        }
    report["exported_count"] = len(exported)
    report["incomplete_records"] = report["incomplete_actions"] + report["episodes_without_summary"]
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["malformed_records"] or report["incomplete_records"] else 0


if __name__ == "__main__":
    sys.exit(main())
