"""Export the LANESHIFT pilot decision rows to Kev System One training JSONL.

The pilot dataset is the 1,348-row ``rule_based`` teacher export produced by
``python/export_decisions.py``.  Each row already contains the player state,
the three nearest vehicles, the authoritative legal actions, and the accepted
teacher action.  This script reconstructs the same compact projection the
controller uses (``compact_kev_state`` -> ``render_compact_state``), attaches
the recorded action as a ``choice`` label, and writes seed-disjoint
train/validation/test files with provenance, an approval file, and a manifest.

It does not relabel outcomes, synthesize actions, or train anything.  The
teacher labels are imitation targets, not human driving data, and the compact
prompt is the ``c1`` projection rather than the schema-2 scene prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kev_state import (
    ACTION_CRITERIA,
    KEV_COMPACT_INSTRUCTIONS,
    compact_kev_state,
    render_compact_state,
)

ALLOWED_ACTIONS = ("LEFT", "STAY", "RIGHT", "BRAKE")
TRAINABLE_SOURCES = {"rule_based", "safe_teacher"}
REQUIRED_FIELDS = (
    "example_id",
    "episode_id",
    "scenario_seed",
    "split",
    "decision_seq",
    "player_lane",
    "player_target_lane",
    "player_speed",
    "lane_change_in_progress",
    "legal_actions",
    "action",
    "label_source",
    "accepted",
)
VEHICLE_FIELDS = tuple(
    f"v{index}_{name}"
    for index in range(1, 4)
    for name in ("present", "lane", "distance", "rel_speed", "in_player_lane")
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def vehicle_from_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    if not row.get(f"v{index}_present"):
        return None
    lane = row.get(f"v{index}_lane")
    distance = row.get(f"v{index}_distance")
    relative_speed = row.get(f"v{index}_rel_speed")
    if lane not in (0, 1, 2) or not isinstance(distance, (int, float)) or not isinstance(relative_speed, (int, float)):
        raise ValueError(f"{row.get('example_id')}: vehicle {index} has invalid lane/distance/relative speed")
    return {
        "lane": int(lane),
        "relative_distance_z": float(distance),
        "relative_speed_z": float(relative_speed),
    }


def row_to_state(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the only state fields the compact Kev projection reads."""
    legal_actions = [action for action in str(row["legal_actions"]).split("|") if action]
    traffic = []
    for index in range(1, 4):
        vehicle = vehicle_from_row(row, index)
        if vehicle is not None:
            traffic.append(vehicle)
    return {
        "player": {
            "lane": int(row["player_lane"]),
            "target_lane": int(row["player_target_lane"]),
            "speed": float(row["player_speed"]),
            "lane_change_in_progress": bool(row["lane_change_in_progress"]),
        },
        "traffic": traffic,
        "legal_actions": legal_actions,
    }


def row_to_training_record(row: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return the Kev record, its rendered state, and the reconstructed state."""
    state = row_to_state(row)
    compact = compact_kev_state(state)
    prompt = render_compact_state(compact)
    action = str(row["action"])
    legal_actions = state["legal_actions"]
    if action not in legal_actions:
        raise ValueError(f"{row.get('example_id')}: action {action!r} is not in legal_actions {legal_actions}")
    record = {
        "state": prompt,
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": KEV_COMPACT_INSTRUCTIONS,
                "criteria": {name: ACTION_CRITERIA[name] for name in legal_actions},
                "label": action,
            }
        },
    }
    return record, prompt, state


def validate_rows(rows: list[dict[str, Any]], include_sources: set[str]) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seed_splits: dict[int, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    accepted_rows = 0

    if not rows:
        problems.append("source dataset has no rows")

    for row in rows:
        example_id = row.get("example_id")
        missing = [field for field in REQUIRED_FIELDS + VEHICLE_FIELDS if field not in row]
        if missing:
            problems.append(f"{example_id}: missing fields {missing}")
            continue
        if example_id in seen_ids:
            problems.append(f"{example_id}: duplicate example_id")
            continue
        seen_ids.add(example_id)

        source = str(row.get("label_source"))
        source_counts[source] += 1
        if source not in include_sources:
            continue
        if source not in TRAINABLE_SOURCES:
            problems.append(f"{example_id}: source {source!r} is not in the trainable source set")
            continue
        if row.get("accepted") not in (1, True):
            problems.append(f"{example_id}: rejected decision is not a training row")
            continue
        action = row.get("action")
        if action not in ALLOWED_ACTIONS:
            problems.append(f"{example_id}: invalid action {action!r}")
            continue
        legal_actions = [name for name in str(row.get("legal_actions", "")).split("|") if name]
        if (
            not legal_actions
            or "STAY" not in legal_actions
            or len(set(legal_actions)) != len(legal_actions)
            or any(name not in ALLOWED_ACTIONS for name in legal_actions)
        ):
            problems.append(f"{example_id}: invalid legal_actions {legal_actions}")
            continue
        if action not in legal_actions:
            problems.append(f"{example_id}: action {action!r} is not legal")
            continue
        split = str(row.get("split"))
        if split not in {"train", "validation", "test"}:
            problems.append(f"{example_id}: invalid split {split!r}")
            continue
        seed = row.get("scenario_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            problems.append(f"{example_id}: invalid scenario_seed {seed!r}")
            continue
        seed_splits[seed].add(split)
        split_counts[split] += 1
        action_counts[action] += 1
        accepted_rows += 1
        try:
            row_to_training_record(row)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{example_id}: {exc}")

    for seed, splits in seed_splits.items():
        if len(splits) > 1:
            problems.append(f"seed {seed} appears in more than one split: {sorted(splits)}")
    for split in ("train", "validation", "test"):
        if split_counts[split] == 0:
            warnings.append(f"split {split} is empty")

    return {
        "source_rows": len(rows),
        "accepted_rows": accepted_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "seeds_per_split": {
            split: sorted(seed for seed, splits in seed_splits.items() if split in splits)
            for split in ("train", "validation", "test")
        },
        "problems": problems,
        "warnings": warnings,
    }


def export(
    decisions_path: Path,
    output_dir: Path,
    include_sources: set[str],
    source_card: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(decisions_path)
    validation = validate_rows(rows, include_sources)
    if validation["problems"]:
        raise SystemExit("dataset validation failed:\n" + "\n".join(validation["problems"]))
    if validation["warnings"]:
        raise SystemExit("dataset validation failed:\n" + "\n".join(validation["warnings"]))
    if validation["accepted_rows"] != len(rows):
        raise SystemExit(
            f"only {validation['accepted_rows']} of {len(rows)} rows passed the source allowlist "
            f"{sorted(include_sources)}; refusing a partial export"
        )

    groups: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    provenance: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    approvals: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    vehicle_counts: Counter[int] = Counter()

    for row in rows:
        record, prompt, state = row_to_training_record(row)
        split = str(row["split"])
        groups[split].append(record)
        prompt_lengths.append(len(prompt))
        vehicle_counts[len(state["traffic"])] += 1
        provenance[split].append(
            {
                "example_id": row["example_id"],
                "row_id": row["row_id"],
                "session_id": row["session_id"],
                "episode_id": row["episode_id"],
                "scenario_seed": row["scenario_seed"],
                "decision_seq": row["decision_seq"],
                "tick": row["tick"],
                "split": split,
                "action": row["action"],
                "label_source": row["label_source"],
                "legal_actions": str(row["legal_actions"]).split("|"),
                "prompt_mode": "compact-c1",
                "prompt_sha256": sha256_text(prompt),
                "source_row_sha256": sha256_text(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                ),
                "observation_schema": 1,
                "source_dataset_sha256": sha256_file(decisions_path),
            }
        )
        approvals.append(
            {
                "example_id": row["example_id"],
                "action": row["action"],
                "approved": True,
                "reason": (
                    "accepted rule_based teacher decision; action is in the authoritative "
                    f"legal_actions set; split={split}"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        training_path = output_dir / f"kev_{split}.jsonl"
        provenance_path = output_dir / f"kev_{split}.provenance.jsonl"
        write_jsonl(training_path, groups[split])
        write_jsonl(provenance_path, provenance[split])
        split_rows = groups[split]
        split_files[split] = {
            "file": training_path.name,
            "sha256": sha256_file(training_path),
            "rows": len(split_rows),
            "episodes": len({row["episode_id"] for row in provenance[split]}),
            "seeds": sorted({row["scenario_seed"] for row in provenance[split]}),
            "action_counts": dict(sorted(Counter(
                row["questions"]["next_action"]["label"] for row in split_rows
            ).items())),
            "provenance_file": provenance_path.name,
            "provenance_sha256": sha256_file(provenance_path),
        }

    approvals_path = output_dir / "approvals.jsonl"
    write_jsonl(approvals_path, approvals)

    source_card_data = None
    if source_card is not None and source_card.exists():
        source_card_data = json.loads(source_card.read_text(encoding="utf-8"))

    manifest = {
        "format": "kev-systemone-labelled-choice-jsonl-v1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "decisions": str(decisions_path),
            "decisions_sha256": sha256_file(decisions_path),
            "card": str(source_card) if source_card is not None else None,
            "card_sha256": sha256_file(source_card) if source_card is not None and source_card.exists() else None,
            "rows": len(rows),
            "label_sources": validation["source_counts"],
            "included_sources": sorted(include_sources),
        },
        "prompt": {
            "mode": "compact-c1",
            "renderer": "kev_state.render_compact_state(kev_state.compact_kev_state(state))",
            "source_observation_schema": 1,
            "vehicle_counts": {str(key): value for key, value in sorted(vehicle_counts.items())},
            "chars_min": min(prompt_lengths),
            "chars_max": max(prompt_lengths),
        },
        "splits": split_files,
        "totals": {
            "rows": sum(value["rows"] for value in split_files.values()),
            "episodes": len({row["episode_id"] for split in provenance.values() for row in split}),
            "seeds": len({row["scenario_seed"] for split in provenance.values() for row in split}),
            "action_counts": validation["action_counts"],
        },
        "approval_file": approvals_path.name,
        "approval_file_sha256": sha256_file(approvals_path),
        "quality_gates": {
            "accepted_only": True,
            "legal_action_only": True,
            "seed_disjoint": True,
            "episode_disjoint": True,
            "source_allowlist": sorted(include_sources),
            "no_outcome_fields_in_state": True,
            "prompt_matches_production_compact_renderer": True,
        },
        "limits": [
            "Labels are the deterministic rule_based teacher's accepted actions, not human driving decisions.",
            "The prompt is the compact c1 projection, not the schema-2 scene prompt used by the live controller.",
            "The source rows are observation schema 1; this dataset is not mixed with schema-2 human recordings.",
            "Rows inside one episode are 0.1 s apart and correlated; report per-seed results, not only per-row.",
            "The teacher's 1,348 rows are a pilot. This dataset is a plumbing and imitation experiment, not a driving claim.",
        ],
    }
    if source_card_data is not None:
        manifest["source"]["card_conditions"] = source_card_data.get("conditions")
        manifest["source"]["card_counts"] = source_card_data.get("counts")
        manifest["source"]["card_splits"] = source_card_data.get("splits")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = f"""# Kev pilot training data

Generated by `python/export_kev_pilot.py` from the preserved 1,348-row pilot
decision export. Do not edit these files by hand.

- Source: `{decisions_path}`
- Source SHA-256: `{manifest['source']['decisions_sha256']}`
- Prompt: compact `c1` (`{manifest['prompt']['chars_min']}-{manifest['prompt']['chars_max']}` characters)
- Labels: `rule_based` teacher actions, accepted and legal only
- Train: {split_files['train']['rows']} rows / {split_files['train']['episodes']} episodes / seeds {split_files['train']['seeds']}
- Validation: {split_files['validation']['rows']} rows / {split_files['validation']['episodes']} episodes / seeds {split_files['validation']['seeds']}
- Test: {split_files['test']['rows']} rows / {split_files['test']['episodes']} episodes / seeds {split_files['test']['seeds']}
- Action counts: {manifest['totals']['action_counts']}

Use `kev_train.jsonl` for training and `kev_validation.jsonl` for model
selection. Keep `kev_test.jsonl` untouched until the final comparison. The
files are seed-disjoint; every row for a seed stays in one split.

Upload the three `kev_*.jsonl` files and `manifest.json` to the Colab
fine-tuning notebook. Do not train on `kev_test.jsonl`.

These labels imitate a deterministic rule policy. The dataset supports a
first LoRA experiment, not a claim that Kev drives better.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    report = {
        "output": str(output_dir),
        "rows": manifest["totals"]["rows"],
        "splits": {name: value["rows"] for name, value in split_files.items()},
        "action_counts": manifest["totals"]["action_counts"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "cris@ds-training" / ".portable-dev" / "pilot" / "decisions.jsonl",
        help="flat pilot decisions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "kev_pilot_data",
        help="output directory for the Kev-format files",
    )
    parser.add_argument(
        "--source-card",
        type=Path,
        default=None,
        help="optional decisions.card.json to copy provenance from",
    )
    parser.add_argument(
        "--include-sources",
        default="rule_based",
        help="comma-separated label sources approved for training (rule_based,safe_teacher)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    include_sources = {name.strip() for name in args.include_sources.split(",") if name.strip()}
    if not include_sources or not include_sources.issubset(TRAINABLE_SOURCES):
        raise SystemExit(f"--include-sources must be a subset of {sorted(TRAINABLE_SOURCES)}")
    source_card = args.source_card
    if source_card is None:
        candidate = args.decisions.with_name("decisions.card.json")
        source_card = candidate if candidate.exists() else None
    report = export(args.decisions, args.output, include_sources, source_card)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
