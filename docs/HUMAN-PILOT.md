# Human pilot — 2026-11-01 21:51

This was a genuine keyboard-controlled session, not a controller or test input.

- Session: `session_20261101_215100_688`
- Scenario seed: `2995404006`
- Accepted human actions: 17
- Action counts: LEFT 9, RIGHT 8, STAY 0, BRAKE 0
- Human/non-human: 17 human, 0 non-human
- Malformed: 0
- Duplicates: 0
- Illegal human actions: 0
- Distinct seeds: 1
- Splits: train 17, validation 0, test 0
- Raw SHA-256: `2df5705290fad783669fc57b680ecb2b9c88e9d765658730fec437462eecccbd`

The session was still recording when the pilot export was taken, so the export
has no terminal episode outcome. The decisions and exact `state_before`
snapshots are valid.

## What this pilot proves

- Human keyboard actions are recorded with `label_source=human` and
  `accepted=true`.
- `state_before` is the exact pre-action snapshot.
- Actions are linked to episode, seed, decision sequence and physics tick.
- The exporter produces raw records, Kev `state` + `questions.next_action`
  rows, metadata and a data card.
- No controller, fallback or synthetic action was labelled human.

## Gaps

- One seed, so no held-out split is possible.
- STAY and BRAKE are absent; the human used only lane changes.
- The session was not finalised with an episode outcome.
- The package executable used for this pilot predates the `human_action` and
  `recent_actions` event fields. The exporter used the accepted
  `requested_action` as the legacy human action. Future builds record both.
- This session is observation schema 1. Its raw records are preserved, but v2
  training export excludes it; a v2 human session must be recorded with a
  current build before it can join the v2 dataset.

Do not treat 17 lane-change decisions as sufficient training data. This is a
schema and workflow pilot.
