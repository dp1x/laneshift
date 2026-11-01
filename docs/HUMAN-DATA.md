# Human decision collection

This is the Phase 2 workflow. It records human keyboard decisions in the
simulation. The model is not controlling the car in this mode.

## Launch

From the repository (development build):

```powershell
RECORD-HUMAN.bat
```

From the portable package:

```powershell
RECORD-HUMAN.bat
```

The game opens the normal playable mode with `--record-on-start --no-controller`.
The title screen appears; click **Start a run**, then drive with A/D and Space.

## What is recorded

Every accepted human action produces an `action_decision` event in
`recordings/session_*/events.jsonl` with:

- `human_action`: the action the player actually selected.
- `state_before`: the exact pre-action simulation snapshot.
- `episode_id`, `scenario_seed`, `decision_seq`, `physics_tick`.
- `recent_actions`: the preceding action history.
- `label_source: "human"`, `accepted: true`.
- `human_adjudicated_action`: reserved for post-game review; it is kept
  separate from the training label.
- `episode_outcome` is available by joining `episodes.jsonl`; it is not part of
  the model-facing state.

Controller and fallback actions are never labelled human. A controller timeout
is recorded as `controller_fallback`, not as a human STAY.

## Export

Raw records and Kev training rows are separate:

```powershell
cd python
python export_human_decisions.py `
  --recordings "..\recordings" `
  --output "..\human-data" `
  --prompt-mode scene
```

For a specific pilot session:

```powershell
python export_human_decisions.py `
  --recordings "..\recordings" `
  --session session_YYYYMMDD_HHMMSS_mmm `
  --output "..\human-data\pilot"
```

Outputs:

| File | Meaning |
|---|---|
| `raw_human.jsonl` | Full raw event and exact `state_before`; no derived labels |
| `kev_train.jsonl` / `kev_validation.jsonl` / `kev_test.jsonl` | Kev `state` + `questions.next_action` rows |
| `kev_*.meta.jsonl` | Episode, seed, decision, action and outcome provenance |
| `human-data.card.json` | Counts, hashes, split membership and limits |
| `human-data.report.md` | Human-readable quality report |

The exporter uses the same schema-2 scene observation and prompt as inference
(`prompt_mode=scene`). It can also export the compact baseline with
`--prompt-mode compact` or the older lane summary with `--prompt-mode lane`.

Only `state_before` snapshots with `observation_schema: 2` become training
rows. Schema-1 records are written to `raw_human.jsonl` unchanged and counted
in `legacy_schema_excluded`; they are not mixed into v2 training. Passing
`--allow-legacy` is an explicit override and is not recommended.

## Splits

Rows are grouped by scenario seed. A seed appears in exactly one split. With
fewer than three seeds, everything is train and the card records that no
held-out split is possible. Never split adjacent decisions from one episode.

## Quality rules

- Keep `human_action` separate from any later `human_adjudicated_action`.
- Do not infer intent from the outcome.
- Exclude `controller_fallback`, `rule_based`, `external_ai` and rejected action
  events from human training data.
- Flag illegal or malformed human actions instead of silently repairing them.
- Do not duplicate or relabel decisions to manufacture class balance.
- Collect more LEFT/RIGHT/BRAKE examples deliberately; most human play will be
  STAY.
