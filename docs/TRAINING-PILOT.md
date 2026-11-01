# Training pilot: decision-level dataset and a small supervised baseline

This is the first step after the matched policy pilot in
[`RESULTS.md`](RESULTS.md): recorded decisions are turned into a decision-level
dataset, and a small supervised model is trained on held-out seeds with
calibration reported.

**The labels are rule-based.** They are the actions of the deterministic
`rule_based` policy, not human choices. Nothing here shows better driving, and
no fine-tuning has happened. What this pilot establishes is the measurement
path: recording → decision export → seed-grouped evaluation → calibration.

## 1. Conditions

| Field | Value |
| --- | --- |
| Policy | `rule_based` (Python controller, 10 Hz decisions) |
| Seeds | 3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67 |
| Episodes | 1 per seed (15 total) |
| Episode duration | 30 s simulated |
| Traffic density | 1.5 |
| Simulation speed | 2.0x (runtime bound only) |
| Decision / telemetry cadence | 10 Hz / 30 Hz |
| Runner | `python/experiment_runner.py` |
| Session | `session_20261101_144044_173` |
| Machine | Snapdragon X Elite, Windows on ARM64, CPU |

Commands:

```powershell
python python\experiment_runner.py --project . --policies rule_based `
  --seeds 3,7,11,19,23,29,31,37,41,43,47,53,59,61,67 `
  --duration 30 --traffic-density 1.5 --sim-speed-scale 2.0 `
  --output .portable-dev\pilot\pilot-15x30.csv

python python\export_decisions.py `
  --session .portable-dev\recordings\session_20261101_144044_173 `
  --output .portable-dev\pilot\decisions.jsonl `
  --duration 30 --traffic-density 1.5 --sim-speed-scale 2.0 --split-seed 7

python python\train_baseline.py --decisions .portable-dev\pilot\decisions.jsonl `
  --output-dir experiments\20261101-144858-baseline-rule_based\none

python python\train_baseline.py --decisions .portable-dev\pilot\decisions.jsonl `
  --output-dir experiments\20261101-144858-baseline-rule_based\balanced --class-weights balanced
```

`run_tests.bat` also writes its own recording sessions into
`.portable-dev/recordings`, so the export is pointed at the pilot session with
`--session` rather than at the whole recordings root.

## 2. Dataset

Recording session: 4,756 telemetry snapshots, 4,281 events, 15 episode
summaries, status `complete`. The join produced **1,348 decision rows**, one per
accepted controller decision, with 0 rejected, 0 incomplete, 0 illegal and 0
malformed records.

Rows per episode vary because 8 of 15 episodes ended in a crash before 30 s:

| Seed | Score | Distance (m) | Survival (s) | Ending | Final lane | Actions |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 3 | 248.5 | 347 | 12.8 | crash | 1 | RIGHT:1 STAY:53 |
| 7 | 330.6 | 496 | 21.2 | crash | 2 | BRAKE:10 LEFT:3 RIGHT:3 STAY:74 |
| 11 | 467.0 | 670 | 30.0 | timeout | 1 | BRAKE:14 LEFT:1 RIGHT:1 STAY:112 |
| 19 | 442.7 | 677 | 30.0 | timeout | 1 | BRAKE:15 LEFT:4 RIGHT:4 STAY:105 |
| 23 | 707.1 | 821 | 30.0 | timeout | 1 | BRAKE:4 LEFT:2 RIGHT:2 STAY:120 |
| 29 | 549.9 | 749 | 30.0 | timeout | 2 | BRAKE:10 LEFT:1 RIGHT:2 STAY:115 |
| 31 | 72.4 | 162 | 6.2 | crash | 0 | BRAKE:1 LEFT:1 RIGHT:1 STAY:23 |
| 37 | 513.6 | 665 | 24.4 | crash | 1 | BRAKE:4 LEFT:3 RIGHT:2 STAY:95 |
| 41 | 251.6 | 516 | 30.0 | timeout | 0 | BRAKE:28 LEFT:2 RIGHT:1 STAY:97 |
| 43 | 632.7 | 827 | 30.0 | timeout | 2 | BRAKE:1 RIGHT:1 STAY:126 |
| 47 | 474.1 | 526 | 19.3 | crash | 1 | BRAKE:2 LEFT:2 RIGHT:1 STAY:77 |
| 53 | 72.6 | 164 | 6.2 | crash | 1 | LEFT:1 STAY:25 |
| 59 | 251.7 | 355 | 13.2 | crash | 2 | BRAKE:3 LEFT:1 RIGHT:1 STAY:51 |
| 61 | 310.1 | 601 | 30.0 | timeout | 0 | BRAKE:19 LEFT:2 RIGHT:1 STAY:106 |
| 67 | 25.7 | 86 | 3.4 | crash | 1 | LEFT:1 STAY:13 |

Aggregate: mean score 356.68, mean distance 510.68 m, mean survival 21.12 s,
8 crashes / 7 timeouts, controller mean decision latency 0.53 ms, 0 rejected
controller commands, 0 stale decisions.

Action distribution over 1,348 decisions:

| Action | Rows | Share |
| --- | ---: | ---: |
| STAY | 1,192 | 88.4% |
| BRAKE | 111 | 8.2% |
| LEFT | 24 | 1.8% |
| RIGHT | 21 | 1.6% |

This is a STAY-dominated dataset. Density 1.5 (up from 0.7 in the earlier
pilot) is what made the minority classes non-trivial at all; at 0.7 the policy
performs almost no manoeuvres.

## 3. Row schema

One row per controller decision, written twice with the same columns:
`decisions.jsonl` (canonical) and `decisions.csv` (compact, 57 columns).

**Identity and provenance.** `row_id`, `example_id`, `session_id`,
`episode_id` (namespaced as `<session_id>:<episode_id>` because episode names
repeat across processes), `scenario_seed`, `split`, `decision_seq`, `tick`,
`sim_time`.

**Decision state.**

- Player: `player_lane`, `player_target_lane`, `player_speed`,
  `lane_change_in_progress`, `lane_change_cooldown`.
- Legality: `legal_actions` (`LEFT|STAY|RIGHT|BRAKE`) plus one `legal_*` flag
  per action.
- Nearest three traffic vehicles, ordered by absolute relative distance rounded
  to 2 decimals, with snapshot order as the tie-break — the same ordering
  `kev_state.compact_kev_state` uses: `v{n}_present`, `v{n}_lane`,
  `v{n}_distance`, `v{n}_rel_speed`, `v{n}_ttc`, `v{n}_in_player_lane`.
- Lane aggregates over the full traffic list: `nearest_same_lane_gap`,
  `nearest_left_gap`, `nearest_right_gap`, `min_ttc`.

`ttc` is `distance / closing_speed` for a vehicle ahead in the player's lane
with `closing_speed > 0`, capped at 60 s and null otherwise. Gaps are metres
along +Z; `rel_speed` is player speed minus vehicle speed, so positive means
closing.

**Label and action outcome.** `action`, `label_source` (`rule_based` for all
rows), `prob_LEFT`/`prob_STAY`/`prob_RIGHT`/`prob_BRAKE` (null for rule-based
decisions; the policy records no distribution), `accepted` (1; rejected
decisions are not exported as labels), `applied_action`,
`application_latency_ticks`.

**Episode outcome (evaluation only).** `episode_score`, `episode_distance`,
`episode_survival_time`, `episode_collisions`, `episode_overtaken`,
`episode_termination`, `episode_success`.

**Model features.** `feature_vector` maps a row to 33 floats:
lane one-hot (3), target-lane delta, lane-change flag and cooldown, speed/28,
four legality flags, six values per vehicle (present, lane, distance,
relative speed, TTC, in-lane), and the four lane aggregates. Missing vehicles
and clear lanes use explicit sentinels (lane `-1`, gap 200 m, TTC 60 s) so the
network sees "free road" as a value rather than as a hole.

**Deliberately excluded from the features:** every episode-outcome column, all
ids and timestamps, the split label, the chosen action, `label_source`, and the
cumulative `score`/`distance`/`survival_time` counters that appear in the raw
snapshot (they are episode progress, not decision state). Tests assert that
mutating every outcome column leaves `feature_vector` unchanged and that the
feature names intersect neither the outcome columns nor the identity columns.

## 4. Splits

`dataset_prep.split_by_scenario` groups whole scenario seeds, so an episode and
all of its 0.1 s-apart decisions stay in one split. Validation and test are
20% of the seeds each, split seed 7.

| Split | Seeds | Rows | Episodes | Actions |
| --- | --- | ---: | ---: | --- |
| train | 11, 19, 23, 31, 37, 43, 47, 53, 67 | 764 | 9 | LEFT 15, STAY 696, RIGHT 12, BRAKE 41 |
| validation | 3, 41, 59 | 238 | 3 | LEFT 3, STAY 201, RIGHT 3, BRAKE 31 |
| test | 7, 29, 61 | 346 | 3 | LEFT 6, STAY 295, RIGHT 6, BRAKE 39 |

The export refuses to write unless **all three pairs** of splits are disjoint in
both scenario seeds and episode ids, and unless every split is non-empty. The
trainer re-checks the same two properties when it loads the decision file.

Row counts overstate evidence: decisions inside one episode are 0.1 s apart, so
the effective sample size is closer to the 15 episodes than to 1,348 rows. The
report therefore also breaks validation and test down per seed.

## 5. Baseline

Small MLP, one `tanh` hidden layer and a 4-way softmax, trained with Adam and
L2 in NumPy (no new dependency). Features are standardised with train-split
statistics only, and balanced class weights are computed from train labels
only. Hyperparameters are chosen from a six-point grid
(hidden 8/16/32 × L2 1e-4/1e-3) on **validation log loss**; the test split is
scored once, after selection. Seed 1234 reproduces the saved weights and
metrics exactly on the same Python/NumPy/BLAS build and thread settings;
another BLAS build can change floating-point summation order. The exact
environment is recorded in `metrics.json` and `model.json`.

Two variants are reported. `none` trains on the natural class distribution;
`balanced` reweights the loss by inverse class frequency. Both use the same
selection criterion.

### Metrics

| Split | Model | Rows | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | MLP (none) | 764 | 0.9804 | 0.8406 | 0.0598 | 0.0315 | 0.0095 |
| train | MLP (balanced) | 764 | 0.9581 | 0.8168 | 0.0871 | 0.0537 | 0.0125 |
| train | majority (STAY) | 764 | 0.9110 | 0.2384 | 2.4593 | 0.1780 | 0.0890 |
| validation | MLP (none) | 238 | 0.9286 | 0.6761 | 0.2521 | 0.1212 | 0.0368 |
| validation | MLP (balanced) | 238 | 0.8950 | 0.6355 | 0.3560 | 0.1844 | 0.0690 |
| validation | majority (STAY) | 238 | 0.8445 | 0.2289 | 4.2956 | 0.3109 | 0.1555 |
| **test** | **MLP (none)** | 346 | **0.8844** | **0.4426** | **0.3587** | **0.1763** | **0.0671** |
| test | MLP (balanced) | 346 | 0.8728 | 0.5574 | 0.3831 | 0.1887 | 0.0617 |
| test | majority (STAY) | 346 | 0.8526 | 0.2301 | 4.0728 | 0.2948 | 0.1474 |

The majority class in the training split is STAY, so the majority baseline is
also the STAY baseline.

Per-seed held-out results (unweighted model):

| Split | Seed | Rows | Accuracy | Macro F1 | Log loss | Episode ending |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| validation | 3 | 54 | 0.9815 | 0.4953 | 0.0251 | crash |
| validation | 41 | 128 | 0.8906 | 0.6678 | 0.4076 | timeout |
| validation | 59 | 56 | 0.9643 | 0.6952 | 0.1153 | crash |
| test | 7 | 90 | 0.8778 | 0.4992 | 0.5533 | crash |
| test | 29 | 128 | 0.9375 | 0.4290 | 0.1625 | timeout |
| test | 61 | 128 | 0.8359 | 0.3616 | 0.4181 | timeout |

Test confusion matrix (unweighted model):

| true \ predicted | LEFT | STAY | RIGHT | BRAKE |
| --- | ---: | ---: | ---: | ---: |
| LEFT | 1 | 4 | 0 | 1 |
| STAY | 1 | 283 | 1 | 10 |
| RIGHT | 0 | 6 | 0 | 0 |
| BRAKE | 1 | 16 | 0 | 22 |

Test confusion matrix (balanced model):

| true \ predicted | LEFT | STAY | RIGHT | BRAKE |
| --- | ---: | ---: | ---: | ---: |
| LEFT | 1 | 5 | 0 | 0 |
| STAY | 5 | 273 | 6 | 11 |
| RIGHT | 0 | 2 | 4 | 0 |
| BRAKE | 1 | 14 | 0 | 24 |

Calibration on the held-out test split: ECE 0.067 and Brier 0.176 for the
unweighted model, against 0.147 and 0.295 for the always-STAY baseline. The
model is over-confident in the top bin and its confidence is not yet useful as
a safety signal, but it is better calibrated than a constant predictor.

## 6. What worked, what failed

Worked:

- The recording → export → training loop is reproducible end to end, including
  namespacing episodes across sessions and excluding outcome columns by test.
- The dataset is non-degenerate: density 1.5 produced 156 non-STAY decisions
  (11.6%), so all four classes appear — but the turn classes are thin: 45 turn
  decisions overall and only 6 LEFT / 6 RIGHT in the held-out test split.
- Seed-grouped held-out evaluation works, and per-seed results expose how much
  of the headline number comes from a single seed.
- Both variants beat the STAY baseline on every metric, including calibration,
  without ever seeing an outcome column.

Failed or weak:

- The unweighted model never predicts RIGHT correctly (0 of 6) and mislabels
  4 of 6 LEFT as STAY. The balanced variant recovers most RIGHT decisions but
  loses accuracy and still misses 5 of 6 LEFT.
- Per-class numbers for LEFT and RIGHT rest on 6 examples each in test; they are
  indicative, not measured.
- The unweighted model is close to the majority baseline on accuracy
  (+3.2 points); most of its advantage is in macro F1 and in log loss. BRAKE is
  the only class it learns at a useful rate — recall 22/39 = 0.5641, precision
  0.6667, F1 0.6111 — while the turn classes mostly collapse to STAY (LEFT 1/6
  correct, RIGHT 0/6). That is better than a constant predictor and much worse
  than the rule it is imitating.
- Rule-based decisions carry no probability distribution, so the labelled
  policy's own calibration cannot be measured from this dataset.
- 8 of 15 episodes crashed, several within 7 s (seeds 31, 53, 67). Density 1.5
  is harsh for this rule policy, and the rows are concentrated in the first
  seconds of those episodes.

## 7. Failure modes and limits

- **Rule-based labels.** The target is a deterministic function of the state,
  so the ceiling is imitation, not driving quality. A rule-based dataset cannot
  tell us what a human would do at a merge, and cannot be used to claim a
  driving improvement.
- **Simulation only.** Kinematic lane runner, three lanes, discrete actions, no
  perception, no phone capture.
- **Correlated rows.** 1,348 rows come from 15 episodes; treating them as
  independent overstates confidence. Report per seed, not per row.
- **Tiny minority classes.** 45 turn decisions in the whole dataset, 12 in test.
- **Class-imbalance trade-off is a choice.** Unweighted maximises accuracy;
  balanced maximises macro F1. Neither is "the" baseline, and both are reported.
- **No in-game evaluation.** The MLP is scored offline on logged states. It has
  never driven the car, so nothing here measures closed-loop behaviour or
  safety.
- **Feature set is the decision state only.** Good for leakage, but it also
  means the model cannot use longer history (no sequence window yet).

## 8. Exact next step: human data gate, then Kev fine-tune

The pilot's MLP is a pipeline check. The next milestone is the one that matters
and it is gated on human data:

1. **Record human play.** Run the game with `--record-on-start` (or F9) for at
   least 20 sessions × 3 episodes at density 1.0-1.5, mixed seeds. Human input
   already produces `action_decision` events with `label_source=human`, the
   same schema this export consumes.
2. **Export with session grouping.** Human sessions must be the grouping unit,
   not just seeds: add a `--group-by session` mode to `export_decisions.py` so
   all decisions from one person's session land in exactly one split.
3. **Handle human provenance end to end.** The export is not yet safe for human
   data as-is: the data card's `limits` still say there is no human play, the
   training report only distinguishes human when the label source says so, and
   rule-based and human rows would currently be mixed without an approval or
   source-allowlist gate. Human export needs (a) a data-card profile for human
   sessions, (b) a source allowlist / `--include-sources human` gate shared with
   the Kev export in `dataset_prep.py`, and (c) a check that refuses a mixed
   rule-based + human dataset unless the caller asks for it explicitly.
4. **Review before training.** Use the existing `dataset_prep.py --format kev`
   approval file (`example_id`, `action`, `approved=true`) and
   `--include-sources human`; rule-based rows must never be mixed in silently.
5. **Retrain this baseline on human labels.** Same features, same metrics, same
   held-out seeds. Compare its macro F1 and ECE on human data with the numbers
   above; that difference measures how much of the rule the model actually
   captured versus how much was rule determinism.
6. **Fine-tune Kev** on the approved human export, label the checkpoint
   `fine_tuned_model`, and record its SHA-256. Then run the matched comparison
   (`python/experiment_runner.py`) of original Kev, fine-tuned Kev, the
   rule-based policy and this MLP as a controller on held-out seeds, reporting
   accuracy, macro F1, log loss, Brier, ECE, action distribution, score,
   survival and collisions.
7. **Only then** state whether fine-tuning helped, on which seeds, and at what
   calibration cost.

Until steps 1-4 exist, the honest claim is: the pipeline produces a leakage-free
decision dataset and a small model that imitates the rule-based policy
partially, on held-out seeds.

## 9. Artifacts

- `python/export_decisions.py` — decision-level export (JSONL + CSV + data card).
- `python/train_baseline.py` — MLP baseline, metrics and report.
- `python/tests/test_export_decisions.py`, `python/tests/test_train_baseline.py`.
- `experiments/20261101-144858-baseline-rule_based/decisions.card.json` — data
  card for the pilot dataset.
- `experiments/20261101-144858-baseline-rule_based/{none,balanced}/` — saved
  model, metrics and report for both variants.

The decision file itself lives under the gitignored `.portable-dev/`, so the
data card, `metrics.json` and `model.json` record its SHA-256
(`decisions_sha256`) and the git HEAD plus dirty flag used to produce it. A
fresh clone can verify a regenerated dataset against those hashes. The
committed artifacts were generated before the commit that contains them, so
their `code_commit.dirty` flag is `true` and the recorded commit is that
commit's parent; the dataset hash is the stable identifier.

Verification: `run_tests.bat` passes (73 Python tests and the six Godot test
scripts).
