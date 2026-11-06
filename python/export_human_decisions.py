"""Export genuine human decisions from LANESHIFT recordings to raw records and
Kev-format training examples.

Raw human records are preserved separately. The training row is a derived view:
one decision, the pre-action state, the human action, and no outcome fields in
the model-facing state. Episode outcomes stay in metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kev_state import (
    ACTION_CRITERIA,
    ACTION_CRITERIA_LANE,
    ACTION_CRITERIA_SCENE,
    KEV_COMPACT_INSTRUCTIONS,
    KEV_LANE_INSTRUCTIONS,
    KEV_SCENE_INSTRUCTIONS,
    OBSERVATION_SCHEMA_VERSION,
    compact_kev_state,
    lane_observation,
    render_compact_state,
    render_lane_state,
    render_scene_state,
    scene_observation,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _action_name(event: dict[str, Any]) -> str | None:
    action = event.get("human_action")
    if isinstance(action, str):
        return action
    legacy = event.get("requested_action")
    return legacy if isinstance(legacy, str) else None


def _seed_from_state(state: dict[str, Any]) -> int | None:
    for key in ("scenario_seed", "seed"):
        value = state.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _kev_state(state: dict[str, Any], prompt_mode: str) -> str:
    if prompt_mode == "scene":
        return render_scene_state(scene_observation(state))
    if prompt_mode == "lane":
        return render_lane_state(lane_observation(state))
    return render_compact_state(compact_kev_state(state))


def _criteria(prompt_mode: str, legal_actions: list[str]) -> dict[str, str]:
    """Criteria for exactly the actions the model is offered at inference."""
    if prompt_mode == "scene":
        base = ACTION_CRITERIA_SCENE
    elif prompt_mode == "lane":
        base = ACTION_CRITERIA_LANE
    else:
        base = ACTION_CRITERIA
    return {name: base[name] for name in legal_actions if name in base}


def _prompt_token_count(
    tokenizer: Any,
    state_text: str,
    legal_actions: list[str],
    instructions: str,
    criteria: dict[str, str],
    prompt_length: int,
) -> tuple[int, int, int]:
    """Mirror OnnxKevPolicy._encode: (total, state+delimiter, branch) tokens.

    The inference encoder is the authority. Keeping the same delimiter and
    span construction here makes an exporter gate match the L256 graph exactly.
    """
    from policies.onnx_kev import SPECIAL, _user_tokens

    q_id = tokenizer.token_to_id(SPECIAL[1])
    o_id = tokenizer.token_to_id(SPECIAL[2])
    c_id = tokenizer.token_to_id(SPECIAL[3])
    d_id = tokenizer.token_to_id(SPECIAL[4])
    instr = [q_id] + _user_tokens(tokenizer, instructions)
    spans = [
        [o_id] + _user_tokens(tokenizer, f"{action}: {criteria[action]}") + [c_id]
        for action in legal_actions
    ]
    branch = instr + [token for span in spans for token in span] + [d_id]
    state_tokens = _user_tokens(tokenizer, state_text)
    state_with_delimiter = 1 + len(state_tokens)
    total = state_with_delimiter + len(branch)
    if total > prompt_length:
        raise ValueError(
            f"{prompt_length}-token prompt budget exceeded: {total} tokens "
            f"({state_with_delimiter} state, {len(branch)} branch)"
        )
    return total, state_with_delimiter, len(branch)


def _instructions(prompt_mode: str) -> str:
    if prompt_mode == "scene":
        return KEV_SCENE_INSTRUCTIONS
    if prompt_mode == "lane":
        return KEV_LANE_INSTRUCTIONS
    return KEV_COMPACT_INSTRUCTIONS


def _split_groups(groups: list[str]) -> dict[str, str]:
    """Assign whole seed/session groups to train/validation/test."""
    ordered = sorted(groups)
    if len(ordered) < 3:
        return {key: "train" for key in ordered}
    test_count = max(1, round(len(ordered) * 0.2))
    validation_count = max(1, round(len(ordered) * 0.2))
    while test_count + validation_count >= len(ordered):
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    split = {}
    for key in ordered[:test_count]:
        split[key] = "test"
    for key in ordered[test_count : test_count + validation_count]:
        split[key] = "validation"
    for key in ordered[test_count + validation_count :]:
        split[key] = "train"
    return split


def export(
    recordings: Path,
    output: Path,
    prompt_mode: str,
    session_filter: set[str] | None,
    allow_legacy: bool = False,
    tokenizer_path: Path | None = None,
    prompt_length: int = 256,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    nonhuman = Counter()
    malformed = 0
    duplicates = 0
    illegal = 0
    prompt_over_budget = 0
    prompt_token_counts: list[int] = []
    legacy_schema = 0
    schema_versions = Counter()
    sessions = 0
    episodes: set[tuple[str, str]] = set()
    seen_event_ids: set[str] = set()
    seen_decisions: set[tuple[str, str, int]] = set()
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_names: dict[str, str] = {}

    for session_dir in sorted(recordings.glob("session_*")):
        if session_filter and session_dir.name not in session_filter:
            continue
        events_path = session_dir / "events.jsonl"
        if not events_path.exists():
            continue
        sessions += 1
        events = _read_jsonl(events_path)
        episode_outcomes = {
            event.get("episode_id"): event
            for event in _read_jsonl(session_dir / "episodes.jsonl")
            if event.get("type") in (None, "episode_end") and event.get("episode_id")
        }
        for event in events:
            if event.get("type") != "action_decision":
                continue
            label = event.get("label_source")
            if label != "human":
                nonhuman[str(label)] += 1
                continue
            if not event.get("accepted", False):
                malformed += 1
                continue
            action = _action_name(event)
            state = event.get("state_before")
            if not isinstance(action, str) or not isinstance(state, dict):
                malformed += 1
                continue
            episode_id = str(event.get("episode_id") or state.get("episode_id") or "")
            seed = _seed_from_state(state)
            decision_seq = event.get("decision_seq")
            if not episode_id or seed is None or not isinstance(decision_seq, int):
                malformed += 1
                continue
            event_id = str(event.get("event_id") or "")
            decision_key = (session_dir.name, episode_id, decision_seq)
            if (event_id and event_id in seen_event_ids) or decision_key in seen_decisions:
                duplicates += 1
                continue
            if event_id:
                seen_event_ids.add(event_id)
            seen_decisions.add(decision_key)
            legal = [name for name in state.get("legal_actions", []) if isinstance(name, str)]
            is_legal = action in legal
            if not is_legal:
                illegal += 1
            legal_criteria = _criteria(prompt_mode, legal)
            instructions = _instructions(prompt_mode)
            state_text = _kev_state(state, prompt_mode)
            human_adjudicated = event.get("human_adjudicated_action")
            schema_version = int(state.get("observation_schema", 1) or 1)
            schema_versions[str(schema_version)] += 1
            if schema_version != OBSERVATION_SCHEMA_VERSION and not allow_legacy:
                legacy_schema += 1
                # Preserve the raw record, but do not mix old information into
                # training rows with the new observation schema.
                raw.append({
                    "session_id": session_dir.name,
                    "episode_id": episode_id,
                    "scenario_seed": seed,
                    "decision_seq": decision_seq,
                    "physics_tick": event.get("physics_tick"),
                    "sim_time": event.get("decision_sim_time"),
                    "human_action": action,
                    "human_adjudicated_action": human_adjudicated,
                    "accepted": True,
                    "legal_action": is_legal,
                    "recent_actions": event.get("recent_actions", []),
                    "state_before": state,
                    "episode_outcome": episode_outcomes.get(episode_id),
                    "raw_event": event,
                    "schema_version": schema_version,
                })
                continue
            episodes.add((session_dir.name, episode_id))
            human_adjudicated = event.get("human_adjudicated_action")
            if human_adjudicated is not None and human_adjudicated != action:
                # Keep the labels separate. The training row uses the action the
                # human actually took; adjudication is stored in metadata only.
                pass
            raw_record = {
                "session_id": session_dir.name,
                "episode_id": episode_id,
                "scenario_seed": seed,
                "decision_seq": decision_seq,
                "physics_tick": event.get("physics_tick"),
                "sim_time": event.get("decision_sim_time"),
                "human_action": action,
                "human_adjudicated_action": human_adjudicated,
                "accepted": True,
                "legal_action": is_legal,
                "recent_actions": event.get("recent_actions", []),
                "state_before": state,
                "episode_outcome": episode_outcomes.get(episode_id),
                "raw_event": event,
                "schema_version": schema_version,
            }
            raw.append(raw_record)
            if not is_legal:
                raw_record["training_eligible"] = False
                raw_record["exclusion_reason"] = "illegal_action"
                continue
            if tokenizer is not None:
                if not legal_criteria:
                    malformed += 1
                    raw_record["training_eligible"] = False
                    raw_record["exclusion_reason"] = "no_legal_actions"
                    continue
                try:
                    total_tokens, state_tokens, branch_tokens = _prompt_token_count(
                        tokenizer, state_text, legal, instructions, legal_criteria, prompt_length
                    )
                except ValueError as exc:
                    prompt_over_budget += 1
                    raw_record["training_eligible"] = False
                    raw_record["exclusion_reason"] = "prompt_over_budget"
                    raw_record["prompt_error"] = str(exc)
                    continue
                raw_record["prompt_tokens"] = total_tokens
                raw_record["state_tokens"] = state_tokens
                raw_record["branch_tokens"] = branch_tokens
                raw_record["training_eligible"] = True
                prompt_token_counts.append(total_tokens)
            group_key = f"seed:{seed}"
            group_names[group_key] = group_key
            group_rows[group_key].append(raw_record)

    split_map = _split_groups(list(group_rows))
    training = {"train": [], "validation": [], "test": []}
    meta = {"train": [], "validation": [], "test": []}
    for group_key, rows in group_rows.items():
        split = split_map.get(group_key, "train")
        for row in rows:
            state = row["state_before"]
            legal_actions = [name for name in state.get("legal_actions", []) if isinstance(name, str)]
            example = {
                "state": _kev_state(state, prompt_mode),
                "questions": {
                    "next_action": {
                        "type": "choice",
                        "instructions": _instructions(prompt_mode),
                        "criteria": _criteria(prompt_mode, legal_actions),
                        "label": row["human_action"],
                    }
                },
            }
            training[split].append(example)
            meta[split].append({
                "session_id": row["session_id"],
                "episode_id": row["episode_id"],
                "scenario_seed": row["scenario_seed"],
                "decision_seq": row["decision_seq"],
                "physics_tick": row["physics_tick"],
                "human_action": row["human_action"],
                "human_adjudicated_action": row["human_adjudicated_action"],
                "label_source": "human",
                "legal_action": row["legal_action"],
                "legal_actions": legal_actions,
                "episode_outcome": row["episode_outcome"],
                "prompt_mode": prompt_mode,
                "prompt_tokens": row.get("prompt_tokens"),
                "state_tokens": row.get("state_tokens"),
                "branch_tokens": row.get("branch_tokens"),
                "observation_schema": OBSERVATION_SCHEMA_VERSION,
            })

    raw_path = output / "raw_human.jsonl"
    raw_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in raw) + ("\n" if raw else ""), encoding="utf-8")
    for split, rows in training.items():
        (output / f"kev_{split}.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        (output / f"kev_{split}.meta.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in meta[split]) + ("\n" if meta[split] else ""),
            encoding="utf-8",
        )

    action_counts = Counter(row["human_action"] for row in raw)
    split_counts = {split: len(rows) for split, rows in training.items()}
    seed_count = len(group_rows)
    card = {
        "schema_version": 1,
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "source": "human action_decision events with accepted=true and label_source=human",
        "prompt_mode": prompt_mode,
        "sessions": sessions,
        "episodes": len(episodes),
        "decisions": len(raw),
        "seeds": seed_count,
        "action_counts": dict(action_counts),
        "nonhuman_action_counts": dict(nonhuman),
        "malformed": malformed,
        "duplicates": duplicates,
        "illegal_human_actions": illegal,
        "prompt_length": prompt_length,
        "prompt_over_budget": prompt_over_budget,
        "prompt_tokens": {
            "min": min(prompt_token_counts) if prompt_token_counts else None,
            "max": max(prompt_token_counts) if prompt_token_counts else None,
            "rows": len(prompt_token_counts),
        },
        "schema_versions": dict(schema_versions),
        "legacy_schema_excluded": legacy_schema,
        "splits": split_counts,
        "raw_file": str(raw_path),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "limits": [
            "Human actions are recorded from keyboard input in the simulation, not phone gameplay.",
            "Training rows use the pre-action state_before snapshot; outcomes are metadata only.",
            "human_adjudicated_action is kept separate and is not used as the training label.",
            "Sessions with fewer than three seed groups cannot produce a held-out split.",
        ],
    }
    (output / "human-data.card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Human decision dataset",
        "",
        f"- sessions: {sessions}",
        f"- episodes: {len(episodes)}",
        f"- decisions: {len(raw)}",
        f"- seed groups: {seed_count}",
        f"- action counts: {dict(action_counts)}",
        f"- non-human action counts: {dict(nonhuman)}",
        f"- malformed: {malformed}",
        f"- duplicates: {duplicates}",
        f"- illegal human actions: {illegal}",
        f"- prompt budget: {prompt_length} tokens; over-budget rows: {prompt_over_budget}",
        f"- prompt tokens: {card['prompt_tokens']}",
        f"- observation schema required for training: {OBSERVATION_SCHEMA_VERSION}",
        f"- schema versions: {dict(schema_versions)}",
        f"- legacy schema excluded from training: {legacy_schema}",
        f"- splits: {split_counts}",
        f"- raw file: `{raw_path}`",
        "",
        "The training rows are derived from the exact `state_before` snapshot and the",
        "accepted `human_action`. Episode outcomes are metadata, never model input.",
    ]
    (output / "human-data.report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--output", type=Path, default=Path("human-data"))
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--prompt-mode", choices=("compact", "lane", "scene"), default="scene")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "kev-original" / "tokenizer.json",
        help="tokenizer.json used by the inference graph; required for the exact L256 prompt gate",
    )
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="include schema-v1 states in training rows; not recommended")
    args = parser.parse_args()
    session_filter = set(args.session) if args.session else None
    tokenizer_path = args.tokenizer if args.tokenizer and args.tokenizer.exists() else None
    if tokenizer_path is None:
        print("WARNING: tokenizer not found; prompt token budget is not enforced", flush=True)
    card = export(
        args.recordings,
        args.output,
        args.prompt_mode,
        session_filter,
        args.allow_legacy,
        tokenizer_path,
        args.prompt_length,
    )
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
