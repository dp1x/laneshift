"""Export one row per controller decision from LaneShift recording sessions.

The game writes 30 Hz telemetry plus an `action_decision` event for every
controller action. This script keeps the decision events, joins each one to
its pre-action state and its episode summary, and writes one row per decision.

Feature columns describe the decision state only. Episode outcomes are copied
next to the row for evaluation and are never part of `feature_vector`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dataset_prep import ALLOWED_ACTIONS, prepare_records, read_jsonl, split_by_scenario

SCHEMA_VERSION = 1
ACTIONS = ("LEFT", "STAY", "RIGHT", "BRAKE")
MAX_VEHICLES = 3
TTC_HORIZON_S = 60.0
GAP_CAP_M = 200.0
SPEED_SCALE = 28.0  # simulation max_speed, used only to keep speed O(1)
SESSION_FILES = ("telemetry.jsonl", "events.jsonl", "episodes.jsonl")

# Columns read by feature_vector. Everything else is provenance, label or outcome.
FEATURE_SOURCE_COLUMNS = (
    "player_lane",
    "player_target_lane",
    "lane_change_in_progress",
    "lane_change_cooldown",
    "player_speed",
    "legal_actions",
    "nearest_same_lane_gap",
    "nearest_left_gap",
    "nearest_right_gap",
    "min_ttc",
) + tuple(f"v{i}_{name}" for i in range(1, MAX_VEHICLES + 1)
          for name in ("present", "lane", "distance", "rel_speed", "ttc", "in_player_lane"))

# Recorded for analysis and evaluation; never a model input.
OUTCOME_COLUMNS = (
    "episode_score",
    "episode_distance",
    "episode_survival_time",
    "episode_collisions",
    "episode_overtaken",
    "episode_termination",
    "episode_success",
)

ROW_COLUMNS = (
    "row_id",
    "example_id",
    "session_id",
    "episode_id",
    "scenario_seed",
    "split",
    "decision_seq",
    "tick",
    "sim_time",
    "player_lane",
    "player_target_lane",
    "player_speed",
    "lane_change_in_progress",
    "lane_change_cooldown",
    "legal_actions",
) + tuple(f"legal_{action}" for action in ACTIONS) + (
    "v1_present", "v1_lane", "v1_distance", "v1_rel_speed", "v1_ttc", "v1_in_player_lane",
    "v2_present", "v2_lane", "v2_distance", "v2_rel_speed", "v2_ttc", "v2_in_player_lane",
    "v3_present", "v3_lane", "v3_distance", "v3_rel_speed", "v3_ttc", "v3_in_player_lane",
    "nearest_same_lane_gap",
    "nearest_left_gap",
    "nearest_right_gap",
    "min_ttc",
    "action",
    "label_source",
) + tuple(f"prob_{action}" for action in ACTIONS) + (
    "accepted",
    "applied_action",
    "application_latency_ticks",
) + OUTCOME_COLUMNS


def feature_names():
    """The ordered model input names; matches feature_vector exactly."""
    names = [
        "player_lane_0",
        "player_lane_1",
        "player_lane_2",
        "target_lane_delta",
        "lane_change_in_progress",
        "lane_change_cooldown",
        "player_speed_scaled",
    ]
    names += [f"legal_{action}" for action in ACTIONS]
    for index in range(1, MAX_VEHICLES + 1):
        names += [
            f"v{index}_present",
            f"v{index}_lane",
            f"v{index}_distance",
            f"v{index}_rel_speed",
            f"v{index}_ttc",
            f"v{index}_in_player_lane",
        ]
    names += ["nearest_same_lane_gap", "nearest_left_gap", "nearest_right_gap", "min_ttc"]
    return tuple(names)


FEATURE_NAMES = feature_names()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _round(value, digits=4):
    number = _number(value)
    return None if number is None else round(number, digits)


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def time_to_collision(distance, closing_speed):
    """Seconds until a same-lane vehicle ahead is reached, or None if it is not closing."""
    if distance is None or closing_speed is None:
        return None
    if distance <= 0.0 or closing_speed <= 0.0:
        return None
    return min(distance / closing_speed, TTC_HORIZON_S)


def nearest_vehicles(traffic, limit=MAX_VEHICLES):
    """Order traffic the way kev_state.compact_kev_state does: nearest by rounded absolute distance.

    Python's sort is stable, so vehicles whose rounded distances tie keep their
    snapshot order, matching the Kev projection's tie behaviour.
    """
    ordered = []
    for vehicle in traffic:
        if not isinstance(vehicle, dict):
            continue
        distance = _number(vehicle.get("relative_distance_z"))
        rounded = _round(distance, 2)
        sort_key = abs(rounded) if rounded is not None else 1_000_000.0
        ordered.append((sort_key, distance, vehicle))
    ordered.sort(key=lambda item: item[0])
    return ordered[:limit]


def lane_gap(traffic, lane):
    """Smallest non-negative gap ahead in `lane`, or None when the lane is clear/absent."""
    if not isinstance(lane, int) or lane < 0 or lane > 2:
        return None
    best = None
    for vehicle in traffic:
        if not isinstance(vehicle, dict) or vehicle.get("lane") != lane:
            continue
        distance = _number(vehicle.get("relative_distance_z"))
        if distance is None or distance < 0.0:
            continue
        if best is None or distance < best:
            best = distance
    return best


def project_decision(candidate, session_id):
    """Build one flat decision row from a dataset_prep candidate."""
    state = candidate.get("state")
    player = state.get("player") if isinstance(state, dict) else None
    traffic = state.get("traffic") if isinstance(state, dict) else None
    legal_actions = state.get("legal_actions") if isinstance(state, dict) else None
    if not isinstance(player, dict) or not isinstance(traffic, list) or not isinstance(legal_actions, list):
        return None
    lane = player.get("lane")
    if lane not in (0, 1, 2):
        return None
    accepted = candidate.get("accepted") is True
    action = candidate.get("action")
    if not accepted or action not in ALLOWED_ACTIONS:
        return None

    episode_id = str(candidate.get("episode_id", ""))
    namespaced_episode = f"{session_id}:{episode_id}"
    metadata = candidate.get("decision_metadata") if isinstance(candidate.get("decision_metadata"), dict) else {}
    probabilities = metadata.get("probabilities") if isinstance(metadata.get("probabilities"), dict) else {}
    outcome = candidate.get("outcome") if isinstance(candidate.get("outcome"), dict) else {}

    row = {
        "row_id": f"{namespaced_episode}:{candidate.get('decision_seq')}",
        "example_id": candidate.get("example_id"),
        "session_id": session_id,
        "episode_id": namespaced_episode,
        "scenario_seed": candidate.get("scenario_seed"),
        "split": None,
        "decision_seq": candidate.get("decision_seq"),
        "tick": candidate.get("decision_tick"),
        "sim_time": _round(state.get("sim_time")),
        "player_lane": lane,
        "player_target_lane": player.get("target_lane"),
        "player_speed": _round(player.get("speed")),
        "lane_change_in_progress": 1 if player.get("lane_change_in_progress") else 0,
        "lane_change_cooldown": _round(player.get("lane_change_cooldown")),
        "legal_actions": "|".join(action_name for action_name in ACTIONS if action_name in legal_actions),
    }
    for action_name in ACTIONS:
        row[f"legal_{action_name}"] = 1 if action_name in legal_actions else 0

    for index, (_abs_distance, distance, vehicle) in enumerate(nearest_vehicles(traffic), 1):
        closing = _number(vehicle.get("relative_speed_z"))
        in_lane = vehicle.get("lane") == lane
        row[f"v{index}_present"] = 1
        row[f"v{index}_lane"] = vehicle.get("lane") if vehicle.get("lane") in (0, 1, 2) else None
        row[f"v{index}_distance"] = _round(distance)
        row[f"v{index}_rel_speed"] = _round(closing)
        row[f"v{index}_ttc"] = _round(time_to_collision(distance, closing) if in_lane else None)
        row[f"v{index}_in_player_lane"] = 1 if in_lane else 0
    present_count = sum(1 for index in range(1, MAX_VEHICLES + 1) if row.get(f"v{index}_present"))
    for index in range(present_count + 1, MAX_VEHICLES + 1):
        row[f"v{index}_present"] = 0
        row[f"v{index}_lane"] = None
        row[f"v{index}_distance"] = None
        row[f"v{index}_rel_speed"] = None
        row[f"v{index}_ttc"] = None
        row[f"v{index}_in_player_lane"] = 0

    row["nearest_same_lane_gap"] = _round(lane_gap(traffic, lane))
    row["nearest_left_gap"] = _round(lane_gap(traffic, lane - 1))
    row["nearest_right_gap"] = _round(lane_gap(traffic, lane + 1))
    same_lane_ttcs = []
    for vehicle in traffic:
        if not isinstance(vehicle, dict) or vehicle.get("lane") != lane:
            continue
        ttc = time_to_collision(
            _number(vehicle.get("relative_distance_z")), _number(vehicle.get("relative_speed_z"))
        )
        if ttc is not None:
            same_lane_ttcs.append(ttc)
    row["min_ttc"] = _round(min(same_lane_ttcs)) if same_lane_ttcs else None

    row["action"] = action
    row["label_source"] = candidate.get("label_source")
    for action_name in ACTIONS:
        row[f"prob_{action_name}"] = _round(probabilities.get(action_name), 4)
    row["accepted"] = 1
    row["applied_action"] = action
    row["application_latency_ticks"] = max(
        0, int(candidate.get("application_tick", 0)) - int(candidate.get("decision_tick", 0))
    )

    row["episode_score"] = _round(outcome.get("score"))
    row["episode_distance"] = _round(outcome.get("distance"))
    row["episode_survival_time"] = _round(outcome.get("survival_time"))
    row["episode_collisions"] = outcome.get("collision_count")
    row["episode_overtaken"] = outcome.get("overtaken_count")
    row["episode_termination"] = outcome.get("termination_reason")
    if outcome:
        row["episode_success"] = 1 if outcome.get("termination_reason") in {
            "timeout", "completed", "episode_complete"
        } else 0
    else:
        row["episode_success"] = None
    return row


def feature_vector(row):
    """The model input for one decision row, in FEATURE_NAMES order.

    Only decision-state columns are read. Episode outcomes, ids, timestamps and
    the chosen action are deliberately absent.
    """
    lane = row["player_lane"]
    values = [
        1.0 if lane == 0 else 0.0,
        1.0 if lane == 1 else 0.0,
        1.0 if lane == 2 else 0.0,
        float(row["player_target_lane"] - lane) if row.get("player_target_lane") is not None else 0.0,
        1.0 if row["lane_change_in_progress"] else 0.0,
        float(row["lane_change_cooldown"] or 0.0),
        _clamp(row["player_speed"] or 0.0, 0.0, 2.0 * SPEED_SCALE) / SPEED_SCALE,
    ]
    legal = row["legal_actions"].split("|") if row.get("legal_actions") else []
    values += [1.0 if action in legal else 0.0 for action in ACTIONS]
    for index in range(1, MAX_VEHICLES + 1):
        present = bool(row.get(f"v{index}_present"))
        distance = row.get(f"v{index}_distance")
        rel_speed = row.get(f"v{index}_rel_speed")
        ttc = row.get(f"v{index}_ttc")
        values += [
            1.0 if present else 0.0,
            float(row[f"v{index}_lane"]) if present and row.get(f"v{index}_lane") is not None else -1.0,
            _clamp(distance, -GAP_CAP_M, GAP_CAP_M) if present and distance is not None else 0.0,
            _clamp(rel_speed, -50.0, 50.0) if present and rel_speed is not None else 0.0,
            float(ttc) if ttc is not None else TTC_HORIZON_S,
            1.0 if present and row.get(f"v{index}_in_player_lane") else 0.0,
        ]
    for key in ("nearest_same_lane_gap", "nearest_left_gap", "nearest_right_gap"):
        gap = row.get(key)
        values.append(_clamp(gap, 0.0, GAP_CAP_M) if gap is not None else GAP_CAP_M)
    values.append(row["min_ttc"] if row.get("min_ttc") is not None else TTC_HORIZON_S)
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("feature vector does not match FEATURE_NAMES")
    return values


def discover_sessions(recordings_root):
    """Session directories that contain all three log files, oldest first."""
    root = Path(recordings_root)
    if not root.is_dir():
        raise SystemExit(f"recordings root does not exist: {root}")
    sessions = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and all((path / name).is_file() for name in SESSION_FILES)
    ]
    if not sessions:
        raise SystemExit(f"no session directories with {', '.join(SESSION_FILES)} under {root}")
    return sessions


def build_rows(sessions):
    """Join every session's logs into decision rows plus a provenance report."""
    rows = []
    report = {
        "sessions": [],
        "telemetry_records": 0,
        "action_events": 0,
        "candidate_count": 0,
        "rejected_actions": 0,
        "incomplete_actions": 0,
        "illegal_labels": 0,
        "duplicate_actions": 0,
        "episodes_without_summary": 0,
        "malformed_records": 0,
        "rows_without_episode_summary": 0,
    }
    for session in sessions:
        session_id = session.name
        telemetry, bad_telemetry = read_jsonl(session / "telemetry.jsonl")
        events, bad_events = read_jsonl(session / "events.jsonl")
        episodes, bad_episodes = read_jsonl(session / "episodes.jsonl")
        candidates, session_report = prepare_records(telemetry, events, episodes)
        session_rows = [row for row in (project_decision(c, session_id) for c in candidates) if row]
        rows.extend(session_rows)
        for key in (
            "telemetry_records", "action_events", "candidate_count", "rejected_actions",
            "incomplete_actions", "illegal_labels", "duplicate_actions", "episodes_without_summary",
        ):
            report[key] += int(session_report.get(key, 0))
        report["malformed_records"] += int(session_report.get("malformed_records", 0)) + sum(
            map(len, (bad_telemetry, bad_events, bad_episodes))
        )
        report["rows_without_episode_summary"] += sum(1 for row in session_rows if row.get("episode_success") is None)
        report["sessions"].append({
            "session_id": session_id,
            "path": str(session),
            "telemetry_records": len(telemetry),
            "event_records": len(events),
            "episode_records": len(episodes),
            "candidates": len(candidates),
            "rows": len(session_rows),
            "rejected_actions": int(session_report.get("rejected_actions", 0)),
            "malformed_lines": len(bad_telemetry) + len(bad_events) + len(bad_episodes),
        })
    return rows, report


def _assert_split_disjoint(rows_by_split, key, label):
    """Every pair of splits must share no value of `key` (seed or episode id)."""
    values = {
        name: {row.get(key) for row in split_rows if row.get(key) is not None}
        for name, split_rows in rows_by_split.items()
    }
    for left, right in itertools.combinations(sorted(rows_by_split), 2):
        overlap = values[left] & values[right]
        if overlap:
            sample = ", ".join(str(value) for value in sorted(map(str, overlap))[:5])
            raise SystemExit(f"{label} appear in both {left} and {right}: {sample}")


def assign_splits(rows, validation_fraction, test_fraction, split_seed):
    """Attach a split label from dataset_prep's seed-grouped three-way split.

    Refuses to continue unless every pair of splits is disjoint in both
    scenario seeds and episode ids, so one episode can never be trained on and
    scored at the same time.
    """
    splits = split_by_scenario(
        rows,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    for name, split_rows in splits.items():
        for row in split_rows:
            row["split"] = name
    summary = {}
    for name, split_rows in splits.items():
        seeds = sorted({row["scenario_seed"] for row in split_rows if row.get("scenario_seed") is not None})
        summary[name] = {
            "rows": len(split_rows),
            "episodes": len({row["episode_id"] for row in split_rows}),
            "scenario_seeds": seeds,
            "action_counts": dict(Counter(row["action"] for row in split_rows)),
        }
        if not split_rows:
            raise SystemExit(f"split {name!r} is empty; use more scenario seeds")
    _assert_split_disjoint(splits, "scenario_seed", "scenario seeds")
    _assert_split_disjoint(splits, "episode_id", "episodes")
    return summary


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit(repo_root=None):
    """Best-effort git identity, so artifacts can be tied to the code that made them."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    if commit.returncode != 0 or not commit.stdout.strip():
        return {"commit": None, "dirty": None}
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"commit": commit.stdout.strip(), "dirty": dirty}


def build_data_card(rows, report, split_summary, args):
    """The data card: what this dataset is, how it was split, and what it cannot show."""
    episode_ids = {row["episode_id"] for row in rows}
    label_sources = dict(Counter(str(row["label_source"]) for row in rows))
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode_id"], row)
    outcomes = [row["episode_termination"] for row in episodes.values()]
    mean_score = sum(row["episode_score"] or 0.0 for row in episodes.values()) / len(episodes) if episodes else 0.0
    mean_survival = (
        sum(row["episode_survival_time"] or 0.0 for row in episodes.values()) / len(episodes) if episodes else 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LaneShift simulation recordings joined by python/export_decisions.py",
        "code_commit": code_commit(),
        "conditions": {
            "policy": next(iter(label_sources)) if len(label_sources) == 1 else "mixed",
            "label_sources": sorted(label_sources),
            "decision_cadence_hz": 10,
            "telemetry_cadence_hz": 30,
            "episode_duration_s": args.duration,
            "traffic_density": args.traffic_density,
            "sim_speed_scale": args.sim_speed_scale,
        },
        "counts": {
            "rows": len(rows),
            "episodes": len(episode_ids),
            "scenario_seeds": len({row["scenario_seed"] for row in rows}),
            "sessions": len(report["sessions"]),
            "rows_without_episode_summary": report["rows_without_episode_summary"],
        },
        "episode_outcomes": {
            "termination_reasons": dict(Counter(str(reason) for reason in outcomes)),
            "mean_score": round(mean_score, 4),
            "mean_survival_time": round(mean_survival, 4),
            "episodes_with_collision": sum(1 for row in episodes.values() if (row["episode_collisions"] or 0) > 0),
        },
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "label_source_counts": label_sources,
        "probabilities_present": sum(1 for row in rows if row["prob_LEFT"] is not None),
        "rejected_decisions": report["rejected_actions"],
        "incomplete_decisions": report["incomplete_actions"],
        "malformed_records": report["malformed_records"],
        "sessions": report["sessions"],
        "split_policy": "whole scenario-seed groups; an episode and every adjacent decision stay in one split",
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "splits": split_summary,
        "feature_names": list(FEATURE_NAMES),
        "feature_sources": list(FEATURE_SOURCE_COLUMNS),
        "outcome_columns_excluded_from_features": list(OUTCOME_COLUMNS),
        "limits": [
            "Labels are the deterministic rule_based policy's actions, not human driving decisions.",
            "A model trained on this export can imitate the rule policy; it is not evidence of safer driving.",
            "rule_based decisions carry no action probabilities, so probability columns are null.",
            "Consecutive rows are 0.1 s apart and highly correlated; row counts overstate effective sample size.",
            "All state is simulated ground truth: there is no phone capture, perception pipeline or human play.",
            "Rows are produced from one headless session per runner invocation; split groups are scenario seeds.",
            "Episode outcome columns are for evaluation only and must never enter the model input.",
        ],
    }


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROW_COLUMNS), restval="", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in ROW_COLUMNS})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, help="recordings root containing session_* directories")
    parser.add_argument("--session", type=Path, action="append", default=[],
                        help="explicit session directory; repeatable, overrides --recordings")
    parser.add_argument("--output", type=Path, required=True, help="output stem; writes .jsonl, .csv and .card.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=30.0, help="recorded episode duration, for the data card")
    parser.add_argument("--traffic-density", type=float, default=1.5, help="recorded traffic density, for the data card")
    parser.add_argument("--sim-speed-scale", type=float, default=2.0, help="recorded simulation speed, for the data card")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sessions = list(args.session) if args.session else discover_sessions(args.recordings)
    for session in sessions:
        missing = [name for name in SESSION_FILES if not (session / name).is_file()]
        if missing:
            raise SystemExit(f"{session} is missing {', '.join(missing)}")
    rows, report = build_rows(sessions)
    if not rows:
        raise SystemExit("no accepted controller decisions found; check the recordings")
    split_summary = assign_splits(rows, args.validation_fraction, args.test_fraction, args.split_seed)

    base = args.output.with_suffix("")
    jsonl_path = base.with_name(base.name + ".jsonl")
    csv_path = base.with_name(base.name + ".csv")
    card_path = base.with_name(base.name + ".card.json")
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    card = build_data_card(rows, report, split_summary, args)
    # The dataset itself lives outside git; the hash lets a fresh checkout verify
    # any copy of it against this card.
    card["decisions_file"] = str(jsonl_path)
    card["decisions_sha256"] = file_sha256(jsonl_path)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    summary = {
        "rows": len(rows),
        "episodes": card["counts"]["episodes"],
        "action_counts": card["action_counts"],
        "decisions_sha256": card["decisions_sha256"],
        "splits": {name: {"rows": value["rows"], "episodes": value["episodes"], "scenario_seeds": value["scenario_seeds"]}
                   for name, value in split_summary.items()},
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path), "data_card": str(card_path)},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
