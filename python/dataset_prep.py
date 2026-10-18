"""Join LaneShift JSONL logs into leakage-safe decision candidates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ALLOWED_ACTIONS = {"LEFT", "STAY", "RIGHT", "BRAKE"}
STATE_FEATURES = (
    "protocol_version",
    "episode_id",
    "scenario_seed",
    "sim_time",
    "physics_tick",
    "player",
    "traffic",
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
}
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
        candidates.append({
            "record_type": "decision_candidate",
            "protocol_version": 1,
            "episode_id": episode_id,
            "scenario_seed": before.get("scenario_seed", event.get("scenario_seed")),
            "decision_seq": event.get("decision_seq", event.get("seq", event_index)),
            "decision_tick": decision_tick,
            "application_tick": application_tick,
            "state_id": before.get("state_id", f"{episode_id}:{before['physics_tick']}"),
            "state": features,
            "requested_action": event.get("requested_action", action),
            "action": action,
            "accepted": True,
            "control_source": event.get("control_source", "unknown"),
            "label_source": label_source,
            "outcome": outcome,
            "successful_episode": bool(
                outcome and outcome.get("termination_reason") in {"timeout", "completed", "episode_complete"}
            ),
        })
    report["candidate_count"] = len(candidates)
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


def _write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--successful-only", action="store_true", help="filter exported candidates, retaining the report of all candidates")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
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
    if args.eval_fraction:
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
