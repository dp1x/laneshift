"""Validate the written kev-v2-scene dataset from its files.

This is a post-write check: it reads the JSONL and provenance files exactly as
a training job would, checks the Kev System One shape, prompt hashes, token
budget, legal-action criteria, and seed/episode split isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def validate(root: Path) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    card_path = root / "dataset.card.json"
    manifest_path = root / "manifest.json"
    if not card_path.exists():
        problems.append("dataset.card.json is missing")
        card = {}
    else:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    if not manifest_path.exists():
        problems.append("manifest.json is missing")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_rows = []
    seed_splits: dict[int, set[str]] = defaultdict(set)
    episode_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    actions = Counter()
    sources = Counter()
    token_max = 0
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        data_path = root / f"kev_{split}.jsonl"
        meta_path = root / f"kev_{split}.provenance.jsonl"
        if not data_path.exists() or not meta_path.exists():
            problems.append(f"{split}: data or provenance file is missing")
            continue
        rows = read_jsonl(data_path)
        meta = read_jsonl(meta_path)
        if len(rows) != len(meta):
            problems.append(f"{split}: {len(rows)} rows and {len(meta)} provenance rows")
        split_actions = Counter()
        split_sources = Counter()
        for index, (row, provenance) in enumerate(zip(rows, meta)):
            all_rows.append(row)
            example_id = provenance.get("example_id")
            if example_id in seen_ids:
                problems.append(f"{split}:{index}: duplicate example_id {example_id}")
            seen_ids.add(example_id)
            state = row.get("state")
            question = (row.get("questions") or {}).get("next_action")
            if not isinstance(state, str) or not state.startswith("v2\n"):
                problems.append(f"{split}:{index}: state is not a v2 scene prompt")
            if not isinstance(question, dict):
                problems.append(f"{split}:{index}: next_action question is missing")
                continue
            if question.get("type") != "choice":
                problems.append(f"{split}:{index}: question type is not choice")
            criteria = question.get("criteria")
            label = question.get("label")
            if not isinstance(criteria, dict) or not criteria:
                problems.append(f"{split}:{index}: criteria is empty")
                continue
            legal = provenance.get("legal_actions")
            if list(criteria) != legal:
                problems.append(f"{split}:{index}: criteria keys differ from legal_actions")
            if label not in criteria:
                problems.append(f"{split}:{index}: label {label!r} is not in criteria")
            if provenance.get("prompt_sha256") != sha256_text(state):
                problems.append(f"{split}:{index}: prompt hash mismatch")
            total_tokens = int(provenance.get("total_tokens", 0))
            if total_tokens > int(card.get("prompt_length", 256)):
                problems.append(f"{split}:{index}: {total_tokens} tokens over budget")
            token_max = max(token_max, total_tokens)
            if provenance.get("observation_schema") != 2:
                problems.append(f"{split}:{index}: observation schema is not 2")
            if provenance.get("label_source") not in {"human", "safe_teacher"}:
                problems.append(f"{split}:{index}: invalid label_source")
            if provenance.get("split") != split:
                problems.append(f"{split}:{index}: provenance split mismatch")
            seed = provenance.get("scenario_seed")
            seed_splits[seed].add(split)
            episode_splits[provenance.get("episode_id")].add(split)
            actions[label] += 1
            sources[provenance.get("label_source")] += 1
            split_actions[label] += 1
            split_sources[provenance.get("label_source")] += 1
        split_summary[split] = {
            "rows": len(rows),
            "episodes": len({row.get("episode_id") for row in meta}),
            "seeds": len({row.get("scenario_seed") for row in meta}),
            "actions": dict(sorted(split_actions.items())),
            "sources": dict(sorted(split_sources.items())),
        }
    for seed, splits in seed_splits.items():
        if len(splits) > 1:
            problems.append(f"seed {seed} appears in splits {sorted(splits)}")
    for episode, splits in episode_splits.items():
        if len(splits) > 1:
            problems.append(f"episode {episode} appears in splits {sorted(splits)}")
    human_splits = {split for split in SPLITS if split_summary.get(split, {}).get("sources", {}).get("human")}
    if human_splits - {"train"}:
        problems.append(f"human rows appear outside train: {sorted(human_splits)}")
    if not all(split_summary.get(split, {}).get("rows") for split in SPLITS):
        problems.append("one or more splits are empty")
    if token_max > int(card.get("prompt_length", 256)):
        problems.append(f"max token count {token_max} exceeds the card budget")
    if card.get("tokens", {}).get("max") != token_max:
        warnings.append(f"card token max {card.get('tokens', {}).get('max')} != observed {token_max}")
    if card.get("action_counts") != dict(sorted(actions.items())):
        warnings.append("card action counts differ from the written rows")
    if manifest.get("totals", {}).get("rows") != len(all_rows):
        warnings.append("manifest row total differs from the written rows")
    return {
        "root": str(root),
        "rows": len(all_rows),
        "splits": split_summary,
        "actions": dict(sorted(actions.items())),
        "sources": dict(sorted(sources.items())),
        "token_max": token_max,
        "problems": problems,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
