# Independent data and synthetic-demonstration audit — `kev-v2-scene` + 4,096-row fine-tune subset

**Date:** 2026-11-05
**Auditor role:** independent data / synthetic-demonstration audit (read-only)
**Status:** complete; all figures below were recomputed from the files, not copied from any prior report.

Nothing outside `LANESHIFT/final-audit-data-20261105/` was created, modified, moved, renamed,
relabelled or regenerated. The dataset, public cards, handoff copy, recordings, models and all prior
reports were opened read-only. Helper scripts and their raw output logs live in this directory
(`audit_core.py`, `audit_tokens.py`, `audit_questions.py`, `audit_dupes_turns.py`, `audit_teacher.py`,
`audit_distribution.py`, `audit_subset.py`, `audit_episodes.py`, `audit_budget.py`, `audit_tokenize.py`,
`audit_label_fidelity.py`, plus `*_out.txt`).

---

## 0. Verdict summary

| Question | Verdict |
|---|---|
| Do the published hashes verify? | **Yes.** 16/16 files in `kev-v2-scene/SHA256SUMS`; 7/7 public-card files; 15/15 in the GPU-handoff `dataset/SHA256SUMS` (including `kev_train.jsonl` at `ca2d34f3…` and `kev_test.jsonl` at `d16f65ab…`). |
| Are the claimed counts real? | **Yes.** 25,427 rows = 16,795 / 4,315 / 4,317; 25,368 `safe_teacher` + 59 `human`; 195 episodes = 195 seeds. |
| Is every row inside the L256 graph? | **Yes.** All 25,427 prompts re-tokenised independently: max = 256, over-budget = 0; 10 rows sit exactly at 256. |
| Are splits isolated? | **Seeds/episodes: yes** (0 overlap). **One prompt text is byte-identical between train and test**, and that train row is inside the 4,096-row fine-tune subset. |
| Is the human data real and correctly labelled? | **Yes.** 59 rows / 1 session / 3 episodes; every label reproduces the recorded `human_action`; one unsafe row excluded. |
| Is the teacher safe? | **No, not fully.** 7 crashes in 192 teacher episodes (3.65 %); 518 pre-crash teacher rows are included as `label_quality: ok` in train/val/test. |
| Is the 4,096-row subset a faithful sample of train? | **Mostly.** 100 % of its rows are byte-identical train rows, 124/131 seeds, but turn rows fall from 465 to 105 (22.6 %) and 52 of 125 turn-bearing seeds lose every turn. |
| Does the test measure unseen *scenarios*? | **Unseen seeds, re-weighted grid.** Same six scenario cells, but the cell mix differs by 20.2 TV points (density 1.5 = 47.6 % of test rows vs 27.4 % of train). |
| Is the test set untouched? | **Yes** for training: hash and mtime unchanged since generation, no test row in any training file, subset built only from `kev_train.jsonl`. |

---

## 1. Reconstruction from files

### 1.1 Hash verification (16/16)

`LANESHIFT/cris@ds/.portable-dev/kev-v2-scene/SHA256SUMS`, recomputed byte-for-byte:

| File | Bytes | SHA-256 | Result |
|---|---:|---|---|
| `kev_train.jsonl` | 12,189,117 | `ca2d34f365e074803b99421f83579017a8b7970e83be1f7c2aa74e3382b63ff3` | OK |
| `kev_validation.jsonl` | 3,142,053 | `3098550e7e1e7f79996df3739e59293c06761e5c87a51e6f1968229faea09894` | OK |
| `kev_test.jsonl` | 3,146,096 | `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4` | OK |
| `kev_train.provenance.jsonl` | 19,674,248 | `9bf6025fa247b645490b52645ca2f30eec678e9e30dda96c8470a9c6f92a9a1a` | OK |
| `kev_validation.provenance.jsonl` | 5,052,425 | `f930bdc00ef6cfba4ebf2fc741fc94d2c8ef50d00d3eddc3b18e13b7da97e7ca` | OK |
| `kev_test.provenance.jsonl` | 5,031,022 | `5427c5968e5ef4adfd630331f4c55ef607037f2e117c56d4e6d97463af1a851b` | OK |
| `candidates/human.candidates.jsonl` | 202,997 | `39d559483f7d3ac8ee3e7489ac3d01dcb946f8af9959b773d2fdcd7ac6a67c51` | OK |
| `candidates/safe_teacher.candidates.jsonl` | 48,517,887 | `c6ec31504aef2703fe23270e5b22a164044daccb72ab0ddd1dfa94291b665df9` | OK |
| `approvals.jsonl` | 5,102,090 | `8b8d8684c219cd2ebd6878dede12512400a00ef1879bf192345a0411536e5370` | OK |
| `episode_index.jsonl` | 85,809 | `784b42f85cf9d8f0e93f6f2bb3c27a30d7c909290c6a30d59fa4b14eff27479d` | OK |
| `dataset.card.json` | 13,690 | `a13df32f95f440254b1258b7e2fb4702d2fb344f67bb9c7aad20b466daca9d9d` | OK |
| `human.card.json` | 5,937 | `35500654014d5f0774cb1390f009634d1cec0eefb7300ea6550b0a365a5212bd` | OK |
| `safe_teacher.card.json` | 962 | `0a2831386fa1ce3c4bf1055ebbe81565ede3647f2ae72dee29821dae8f14a960` | OK |
| `manifest.json` | 14,172 | `fef79c5cee92c1a5a498655d987567a9eb5b3cb834892496c70293ae1d408110` | OK |
| `seed_plan.json` | 39,074 | `f31cd518a90edabb7b54f07ccaf72d8e1b4e7be6ae8812be1b94dfb11ef96757` | OK |
| `README.md` | 1,216 | `692d7ad009a26bc99da4a5f5b731b7328d894461a070f362e45d2ebfef85b76a` | OK |

- **Test file hash re-verified against every independent record:** `SHA256SUMS`, `manifest.json`
  (`splits.test.sha256`), `dataset.card.json`, and `kev-deliverables-20261105/results/checkpoint_manifest.json`
  (`test_sha256`) all carry `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4`.
- **Public cards:** `LANESHIFT/cris@ds/datasets/kev-v2-scene/SHA256SUMS` (7 files) — all 7 recomputed
  hashes match, and each card file is byte-identical to its generated copy in `.portable-dev/kev-v2-scene/`.
  `DATA-LOCATION.txt` = `22958ee742741909fa451db2f83e26c4eab22801f9de1cc11c45b970a02ee65b`.
- **Handoff copy:** `LANESHIFT/kev-gpu-handoff-20261105/dataset/SHA256SUMS` — 15/15 hashes verify,
  and each file is byte-identical to its dataset-root counterpart (including `kev_train.jsonl`
  `ca2d34f3…` and `kev_test.jsonl` `d16f65ab…`).

### 1.2 Counts: rows, episodes, seeds, sources

| Split | Rows | Provenance rows | Episodes | Seeds | `human` rows | `safe_teacher` rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 16,795 | 16,795 | 131 | 131 | 59 | 16,736 |
| validation | 4,315 | 4,315 | 32 | 32 | 0 | 4,315 |
| test | 4,317 | 4,317 | 32 | 32 | 0 | 4,317 |
| **total** | **25,427** | **25,427** | **195** | **195** | **59** | **25,368** |

Episodes map 1:1 to seeds (195 distinct `episode_id`, 195 distinct `scenario_seed`, no episode with two
seeds, no seed with two episodes). Every seed carries exactly one `label_source`; every episode likewise.
`episode_index.jsonl` independently reports 195 episodes: 131 train / 32 validation / 32 test and
192 `safe_teacher` + 3 `human`.

### 1.3 Actions

| Split | STAY | BRAKE | LEFT | RIGHT | Turn (L+R) |
|---|---:|---:|---:|---:|---:|
| train | 13,921 (82.89 %) | 2,350 (13.99 %) | 262 (1.56 %) | 262 (1.56 %) | 524 (3.12 %) |
| validation | 3,625 (84.01 %) | 572 (13.26 %) | 61 (1.41 %) | 57 (1.32 %) | 118 (2.73 %) |
| test | 3,633 (84.16 %) | 562 (13.02 %) | 55 (1.27 %) | 67 (1.55 %) | 122 (2.83 %) |
| **total** | **21,179 (83.29 %)** | **3,484 (13.70 %)** | **378 (1.49 %)** | **386 (1.52 %)** | **764 (3.00 %)** |

Class index is consistent everywhere: LEFT→0 (378), STAY→1 (21,179), RIGHT→2 (386), BRAKE→3 (3,484).
`criteria_keys` equals the legal set on 25,427/25,427 rows; the label is always inside `legal_actions`
(0 violations). `row_id` = SHA-256 of `example_id` on 25,427/25,427 rows; `prompt_sha256` =
SHA-256 of the stored `state` text on 25,427/25,427 rows; every row's `split` field matches the file it
lives in.

### 1.4 Densities and cells (rows)

| Cell | Density | Duration | train | validation | test | total |
|---|---|---:|---:|---:|---:|---:|
| A | 0.7 | 60 s | 1,031 | 257 | 257 | 1,545 |
| B | 1.0 | 60 s | 2,578 | 515 | 515 | 3,608 |
| C | 1.5 | 60 s | 2,837 | 1,229 | 1,289 | 5,355 |
| D | 1.5 | 90 s | 1,761 | 771 | 766 | 3,298 |
| E | 2.5 | 60 s | 4,747 | 771 | 718 | 6,236 |
| F | 3.5 | 60 s | 3,782 | 772 | 772 | 5,326 |
| H | 0.7 | human | 59 | 0 | 0 | 59 |

Density totals by split — train: 0.7 = 1,090; 1.0 = 2,578; 1.5 = 4,598; 2.5 = 4,747; 3.5 = 3,782.
Validation: 257 / 515 / 2,000 / 771 / 772. Test: 257 / 515 / 2,055 / 718 / 772.
`duration_s` is 60 s on 22,070 rows, 90 s on 3,298 rows, null on the 59 human rows.

### 1.5 Terminal reasons

Row-level `episode_termination` (every row in a crashed episode carries the episode outcome):

| Split | crash | timeout | window_close |
|---|---:|---:|---:|
| train | 244 | 16,547 | 4 |
| validation | 69 | 4,246 | 0 |
| test | 260 | 4,057 | 0 |

Episode-level (`episode_index.jsonl`): 185 timeout, 9 crash, 1 window_close.
- train `safe_teacher`: 128 episodes → 124 timeout, 4 crash.
- train `human`: 3 episodes → 2 crash, 1 window_close.
- validation: 32 → 31 timeout, 1 crash.
- test: 32 → 30 timeout, 2 crash.

Rows per episode: train 1–193 (mean 128.21), validation 69–193 (mean 134.84), test 73–193 (mean 134.91).

---

## 2. What is human and what is synthetic

### 2.1 Scale

| Component | Rows | Share of corpus | Episodes | Seeds | Sim time |
|---|---:|---:|---:|---:|---:|
| `safe_teacher` (synthetic) | 25,368 | **99.768 %** | 192 | 192 | 11,828.63 s |
| `human` | 59 | **0.232 %** | 3 | 3 | 395.67 s |

The 59 human rows are the **only** non-synthetic labels in the corpus. Everything else — all 25,368
teacher rows, in all three splits — is behaviour-cloned from a deterministic script that drives the
same Godot simulator headless. There is no real vehicle, no camera, no perception pipeline and no human
demonstration anywhere in `kev-v2-scene` other than those 59 rows.

### 2.2 Human block (all in train)

| Episode | Rows | LEFT | RIGHT | STAY | BRAKE | Termination | Survival |
|---|---:|---:|---:|---:|---:|---|---:|
| `session_20261103_192529_340:ep_644099890_1` | 1 | 0 | 1 | 0 | 0 | crash | 3.57 s |
| `session_20261103_192529_340:ep_1501968680_2` | 54 | 27 | 27 | 0 | 0 | crash | 382.50 s |
| `session_20261103_192529_340:ep_4135897261_3` | 4 | 2 | 2 | 0 | 0 | window_close | 9.60 s |
| **total** | **59** | **29** | **30** | **0** | **0** | 2 crash / 1 close | **395.67 s** |

- **Action counts: LEFT 29, RIGHT 30, STAY 0, BRAKE 0.** The human block is 100 % lane changes.
  The recording contains 61 `action_decision` events: 60 accepted (30 LEFT, 30 RIGHT) and 1 rejected
  keypress. Human decisions are event-driven, not a continuous trace: the median gap between sampled
  human decisions is 5.15 s of sim time (range 0.37–24.6 s).
- **Sim time:** 395.67 s. **Wall time inside episodes:** 395.64 s (`episode_start`→`episode_end`
  timestamps: 3.55 + 382.49 + 9.60 s). **Session wall span:** 45,066.6 s, of which 44,667.7 s
  (12 h 24 m) is a single idle gap between the second and third episode. Real driving wall time is
  therefore ~6.6 minutes, not 12.5 hours.
- **Unsafe / excluded human row:** `human:session_20261103_192529_340:ep_644099890_1:2`
  (row_id `e131ca469b3a87fdd73f28cb7e022bd2745621883843c69f20a9057157999d8a`), action LEFT at
  physics tick 204, excluded with `label_quality: "unsafe"`, `excluded_reason: "target_lane_front_gap_2.87m"`.
  The rule (`build_kev_v2_scene.py:80-117`) rejects a lane change when the nearest forward vehicle in
  the destination lane is closer than 10.0 m (or TTC < 1.0 s). In this state the target lane held a
  truck 2.87 m ahead. The row survives in `candidates/human.candidates.jsonl` with `included: false`.
- **Verified:** all 59 labels equal the recorded `human_action`, and all 59 `state_sha256` values equal
  the hash of the recorded `state_before` (100 % match). The 59 rows re-render byte-identically from
  `state_before` through `kev_state.render_scene_state(kev_state.scene_observation(...))`.

### 2.3 Teacher labels and provenance

- `label_source: "safe_teacher"`, `policy_id: "safe_teacher@1"`, `policy_version: "1"`,
  `generator_version: "kev-v2-scene-builder@1"`, `observation_schema: 2`, `prompt_mode: "scene"`,
  `prompt_length: 256` — identical on all 25,368 teacher rows.
- The teacher is `LANESHIFT/cris@ds/python/policies/safe_teacher.py`, a deterministic hand-written
  heuristic: it stays if the current lane is safe; otherwise it changes lane if an adjacent lane has
  ≥20 m front gap or ≥3 s front TTC (hard floor 10 m) and ≥20 m rear clearance or ≥2 s rear TTC
  (hard floor 8 m); otherwise it brakes. **These labels are a scripted policy's choices, not safety
  ground truth and not an expert driver.**
- Sampling: every second accepted teacher decision (stride 2). Verified from the data: 25,157 of the
  inter-row decision-seq gaps are exactly 2 and the median sim-time delta is 0.4667 s (matching
  `seed_plan.json` `stride.accepted_decisions: 2`). Gaps of 4 (18 cases) and 6 (1 case) correspond to
  the over-budget candidates that were dropped.
- **Label fidelity spot check from the recordings:** 1,545 teacher rows across three whole sessions —
  test `session_20261104_212325_311` (257 rows), validation `session_20261104_223349_382` (257) and
  train `session_20261104_211223_296` (1,031). For every row: `state_sha256` matched the recorded
  `state_before` (1,545/1,545), the recorded `requested_action` matched the dataset label
  (1,545/1,545), and `SafeTeacherPolicy.choose_action(state_before)` reproduced the label
  (1,545/1,545).
- **Provenance limitation:** `state_before` is stored **only for the 59 human rows**. Teacher provenance
  keeps `state_sha256` (hash of the raw state) but not the raw state itself, so for 25,368 rows the
  full simulator state (absolute positions, off-screen vehicles, `on_screen` flags) cannot be
  reconstructed from the dataset — only the rendered prompt text, its hash, and the state hash.

---

## 3. Teacher quality

### 3.1 Crash inventory (7 teacher crashes, 192 episodes = 3.65 %)

| Split | Seed | Cell | Density | Rows in episode | Survival (s) | Last sampled sim (s) | Actions in episode |
|---|---|---:|---:|---:|---:|---:|---|
| test | 920017 | D | 1.5 | 187 | 87.60 | 87.27 | STAY 155, BRAKE 21, RIGHT 6, LEFT 5 |
| test | 920021 | E | 2.5 | 73 | 34.53 | 34.07 | STAY 60, BRAKE 11, RIGHT 1, LEFT 1 |
| validation | 910008 | C | 1.5 | 69 | 32.37 | 31.97 | STAY 57, BRAKE 8, LEFT 2, RIGHT 2 |
| train | 900047 | E | 2.5 | 81 | 37.77 | 37.57 | STAY 59, BRAKE 18, RIGHT 3, LEFT 1 |
| train | 900086 | F | 3.5 | 58 | 26.93 | 26.83 | STAY 45, BRAKE 11, LEFT 2 |
| train | 900107 | D | 1.5 | 25 | 11.70 | 11.43 | STAY 18, BRAKE 7 |
| train | 900113 | E | 2.5 | 25 | 11.57 | 11.43 | STAY 19, BRAKE 5, LEFT 1 |

(For completeness the two human crashes are `644099890` — 1 row, 3.57 s — and `1501968680` — 54 rows,
382.50 s; both train/cell H/0.7.)

- Crash rates by split: train teacher 4/128 (3.12 %), validation 1/32 (3.12 %), test 2/32 (6.25 %).
- Mean teacher survival across 192 episodes: **61.6075 s** (card claims 61.607 — matches).
- **Pre-crash rows retained as training data:** 518 teacher rows (train 189, validation 69, test 260)
  plus 55 human rows = 573 crash-episode rows in total. Every one of them is a **pre-crash** row:
  in all nine crash episodes the largest sampled `sim_time` is strictly below the episode's
  `survival_time` (e.g. 900107: 11.43 s sampled vs 11.70 s survival), so no post-collision frame was
  captured.
- **Post-crash rows: none. Unsafe teacher labels: none.** All 573 crash-episode rows carry
  `label_quality: "ok"` and `included: true`. The dataset does not mark any teacher frame as unsafe,
  even when the episode ended in a collision. Every included row in all three splits is
  `label_quality: "ok"`, `included: true`, `excluded_reason: null`.
- **Every one of the 7 teacher crashes ends on a BRAKE sample.** 900086 ends
  LEFT → STAY → STAY → BRAKE → BRAKE → BRAKE, and 920017 ends … LEFT (86.33 s) → STAY → BRAKE
  (87.27 s). The teacher therefore detects the hazard but not soon enough to avoid the collision.
  Two crashes (900107 in `session_20261104_215525_265` at density 1.5, 900113 in
  `session_20261104_220820_331` at density 2.5) share an identical last-four tail
  (BRAKE, BRAKE, BRAKE, BRAKE at 10.03/10.50/10.97/11.43 s, decision_seq 43/45/47/49) and near-identical
  survival, despite different seeds and sessions; this is flagged as unexplained in §9.3.

### 3.2 The 20 over-budget teacher exclusions

| Property | Value |
|---|---|
| Candidate file | `candidates/safe_teacher.candidates.jsonl` — 25,388 rows, 25,368 `included: true`, 20 `included: false` |
| Label quality | `"invalid"` ×20 (never `"unsafe"`) |
| Action | STAY ×18, BRAKE ×2 |
| Legal action count | 4 on all 20 (LEFT, STAY, RIGHT, BRAKE) |
| Cells | F ×17, E ×3 |
| Densities | 3.5 ×17, 2.5 ×3 |
| Splits | train 16, validation 3, test 1 |
| Seeds | 900039, 900057, 900060 (4), 900062 (5), 900087, 900121 (2), 900126, 900127, 910024, 910025, 910031, 920030 |
| Reason strings | “needs 155 state tokens; only 154 fit” ×11, “156 … only 154 fit” ×8, “157 … only 154 fit” ×1 |

The 154-token figure is correct only for a 4-option prompt: the option branch costs 102 tokens
(74 for 2 options, 88 for 3, 102 for 4 — verified exactly on all 25,427 rows), and 102 + 154 = 256.
So these 20 candidates needed 257–259 total tokens and were rejected by the `_encode` guard before
any encoding was stored — which is why their recorded `state_tokens`/`total_tokens` are `0` and their
content cannot be inspected beyond the reason string.

---

## 4. Full train split vs the 4,096-row fine-tune subset

### 4.1 Subset integrity

`LANESHIFT/kev-deliverables-20261105/subset/` (identical copy in `LANESHIFT/kev-subset-20261105/`):

- 4,096 rows / 4,096 provenance rows; 4,096 unique `row_id`, `example_id`, `prompt_sha256` and
  `state_sha256`.
- **Every subset row exists in `kev_train.jsonl`** (4,096/4,096 row_ids found), its prompt text is
  byte-identical, its `prompt_sha256` matches, and its provenance record is **dict-identical** to the
  train record.
- 100 % `safe_teacher`; 0 human rows; 0 crash-episode rows (`episode_termination: timeout` ×4,096,
  `episode_collisions: 0` ×4,096); 0 duplicate prompts (the builder drops duplicate prompt hashes, but
  the full train contains none to drop).
- 124 seeds / 124 episodes. The 7 excluded train seeds are exactly the 4 teacher crash seeds
  (900047, 900086, 900107, 900113) and the 3 human seeds (644099890, 1501968680, 4135897261).
- **Seed balance is skewed by the top-up:** 122 seeds have 32 rows, seed 900001 has **128**, seed
  900002 has **64**. `make_subset.py` selects 32 rows per seed (3,904), then tops up the missing 192
  rows from the lowest-sorted seeds rather than spreading them.
- One row of the subset is the train half of the train↔test duplicate prompt (row_id
  `b990f7b034e4e8d974153b1114203114333262a04fd32ca7d6cadc58113636cf`, seed 900004).

### 4.2 Side-by-side

| Metric | Full train | Subset | Subset / train |
|---|---:|---:|---:|
| Rows | 16,795 | 4,096 | 24.38 % |
| Episodes | 131 | 124 | 94.7 % |
| Seeds | 131 | 124 | 94.7 % |
| STAY | 13,921 (82.89 %) | 3,439 (83.96 %) | 24.7 % |
| BRAKE | 2,350 (13.99 %) | 552 (13.48 %) | 23.5 % |
| LEFT | 262 (1.56 %) | 55 (1.34 %) | 21.0 % |
| RIGHT | 262 (1.56 %) | 50 (1.22 %) | 19.1 % |
| Turn rows (L+R) | 524 (3.12 %) | 105 (2.56 %) | 20.0 % |
| **Teacher-only turn rows** | **465** (233 L / 232 R; 2.78 % of 16,736) | **105** (55 L / 50 R; 2.56 %) | **22.6 %** |
| Seeds with ≥1 teacher turn | 125 | 73 | 58.4 % |
| Max teacher turns in one seed | 12 (seed 900031) | 4 | — |
| Seeds with ≥1 turn (any source) | 128 (max 54, human seed 1501968680) | 73 (max 4) | 57.0 % |
| Cells A/B/C/D/E/F | 1,031 / 2,578 / 2,837 / 1,761 / 4,747 / 3,782 | 384 / 640 / 704 / 288 / 1,152 / 928 | — |
| Cell H (human) | 59 | 0 | — |
| Densities 0.7/1.0/1.5/2.5/3.5 | 1,090 / 2,578 / 4,598 / 4,747 / 3,782 | 384 / 640 / 992 / 1,152 / 928 | — |
| `total_tokens` min/mean/median/max | 165 / 225.65 / 229 / 256 | 165 / 224.66 / — / 256 | — |
| Rows at 256 tokens | 6 | 1 | — |

The subset preserves the **cell and density proportions** closely (A 6.1 %→9.4 %, B 15.4 %→15.6 %,
C 16.9 %→17.2 %, D 10.5 %→7.0 %, E 28.3 %→28.1 %, F 22.5 %→22.7 %; every density share within 3.2
points) and
preserves the **turn rate** almost exactly (3.12 %→2.56 % of all rows; teacher 2.78 %→2.56 %). What it
does not preserve is the **absolute number** of turn examples (465→105) or the **number of seeds that
contain a turn** (125→73).

### 4.3 Turn retention detail

| Observation | Value |
|---|---:|
| Teacher turns in full train | 465 |
| Teacher turns kept in subset | 105 (22.6 %) |
| Seeds that lose every turn | 52 of 125 |
| Seeds with exactly 1 turn in subset | 51 of 73 |
| Seeds with 2 / 3 / 4 turns | 14 / 6 / 2 |
| Mean turns per turn-bearing seed | 1.44 |
| Turn rows with another turn within 4 decisions in the same seed | **0 of 105** |
| Turn rows whose preceding sampled row in the same seed is STAY / BRAKE / LEFT | 90 / 10 / 5 |

Every retained turn is a **single isolated frame**: no seed contributes two turns close enough in
decision order to show the manoeuvre's beginning and end.

---

## 5. Duplicates and leakage

### 5.1 Within-split uniqueness (0 duplicates anywhere)

| Check | train | validation | test |
|---|---:|---:|---:|
| Unique `row_id` / rows | 16,795 / 16,795 | 4,315 / 4,315 | 4,317 / 4,317 |
| Unique `example_id` / rows | 16,795 / 16,795 | 4,315 / 4,315 | 4,317 / 4,317 |
| Unique `prompt_sha256` / rows | 16,795 / 16,795 | 4,315 / 4,315 | 4,317 / 4,317 |
| Unique `state_sha256` / rows | 16,795 / 16,795 | 4,315 / 4,315 | 4,317 / 4,317 |
| Duplicate `(seed, physics_tick)` pairs | 0 | 0 | 0 |

### 5.2 The single train↔test prompt duplicate

| Field | Train row | Test row |
|---|---|---|
| `example_id` | `safe_teacher:session_20261104_214022_286:ep_900004_4:199` | `safe_teacher:session_20261104_212325_311:ep_920001_1:115` |
| Seed / split | 900004 / train | 920001 / test |
| Cell / density | A / 0.7 | A / 0.7 |
| Tick / sim time | 1393 / 46.433 s | 812 / 27.067 s |
| Label | STAY | STAY |
| Legal actions / tokens | LEFT, STAY, RIGHT, BRAKE / 181 | LEFT, STAY, RIGHT, BRAKE / 181 |
| `state_sha256` | `661d98465d41a0c8ec69c5decf37f80deb64f00ec6701c9eb82d1e20c923719a` | `bad5fe1bf2c09992a69cd4168a6222946a825f65345c8641b111e433b22f3645` |
| `row_id` | `b990f7b034e4e8d974153b1114203114333262a04fd32ca7d6cadc58113636cf` | `f6510867e9507057a5f854bd4c04830eab939dbe47738d4103ab392f76a8552a` |

The rendered prompt is byte-identical (`P C C 28 0 0 0 100 0 0 S` with a single truck at `C+32 +10 18 t`);
the underlying raw states differ, so this is a renderer discretisation collision, not a copied row.
- **Cross-split `state_sha256` duplicates: 0.**
- **The train half is inside the 4,096-row subset**, i.e. the fine-tuned model was trained on one
  prompt that is byte-identical to one test prompt. Both carry the same label (STAY), and the leak is
  1 of 4,317 test rows (0.023 %), so it cannot explain the model's STAY-heavy output.
- There are **no fully empty-road prompts** (`ON\nnone`) in any split, so the dataset-card bullet
  “Repeated empty-road STAY prompts are natural scenario repeats, not duplicated rows” does not
  describe anything actually present in the data.

### 5.3 Seed / episode disjointness and source isolation

- Seeds in more than one split: **0**. Episodes in more than one split: **0**.
- 195 distinct seeds = 195 distinct episodes; each seed and each episode has exactly one `label_source`.
- Human rows: 59 in train, 0 in validation, 0 in test. Validation and test are `safe_teacher`-only.
- The unsafe human row never entered any split (`included: false` in the candidate file).

### 5.4 Overlap with the pilot

| Pilot | Rows | Seeds | Session | Labels | Overlap with `kev-v2-scene` |
|---|---:|---|---|---|---|
| `LANESHIFT/cris@ds/datasets/pilot-v1` | 1,348 (764 train / 238 val / 346 test) | 3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67 | `session_20261101_144044_173` | `rule_based` ×1,348 | **0 seeds, 0 sessions, 0 episodes.** Different seed space, different session, feature-based schema (no prompt text to compare), 452 crash rows of its own. |
| `LANESHIFT/cris@ds-training/docs/pilot-5x20` | 10 episodes | 1–5 | (not recorded) | `rule_based` 5, `onnx_kev` 5 | **0 seeds.** |

The v2 builder also explicitly skipped the 2026-11-02 controller sessions that fed the pilot-era work:
`dataset.card.json` `extra_scan_counts` lists 7 `ignored_sessions` and 38 `ignored_source:original_model`
events, plus `rejected:human` ×1 and `stride_skipped:safe_teacher` ×25,218.

### 5.5 Is the test untouched?

Yes, in the training sense:

- `kev_test.jsonl` mtime `2026-11-04T18:54:18.947Z` and `kev_test.provenance.jsonl` mtime
  `2026-11-04T18:54:19.027Z` — the generation write, with no later modification.
- Hash `d16f65ab…` matches `SHA256SUMS`, `manifest.json`, `dataset.card.json` and the fine-tune
  `checkpoint_manifest.json` (`test_sha256`, `test_rows: 4317`).
- The subset builder reads only `SRC / "kev_train.jsonl"` (`make_subset.py`, `SRC = …/kev-v2-scene`,
  `rows = load(SRC / "kev_train.jsonl")`); its card records `test_split_touched: false`, and 0 test
  row_ids appear in the subset.
- The training log records `records_seen: 4096`, `requested_records: 4096`, `optimizer_steps: 512`,
  `data: /workspace/data/subset_train.jsonl`, `epochs: 1`.
- The only use of the test file is the evaluation that produced
  `results/predictions_{fine_tuned,original}_test.jsonl` — 4,317 rows each, which is the intended use.

---

## 6. Prompt lengths and L256 compatibility

### 6.1 Distribution (`total_tokens`, all 25,427 rows)

| Statistic | Value |
|---|---:|
| min | 165 (train; 166 in validation and test) |
| p01 / p05 | 178 / 190 |
| p25 / p50 / p75 | 222 / 228 / 237 |
| p95 / p99 | 246 / 250 |
| max | 256 |
| rows ≤ 175 tokens | 58 |
| rows ≥ 250 tokens | 343 |
| rows exactly 256 | **10** (6 train, 3 validation, 1 test) |
| rows over 256 | **0** |

### 6.2 The 10 rows at exactly 256

All ten have `state_tokens = 154` and all four legal actions: 154 + 102 (4-option branch) = 256. They
have **zero headroom**. Ids: train `…ep_900039_7:227`, `…ep_900045_13:249`, `…ep_900060_10:131`,
`…ep_900060_10:157`, `…ep_900060_10:255`, `…ep_900121_15:219`; validation `…ep_910024_4:225`,
`…ep_910024_4:231`, `…ep_910031_5:225`; test `…ep_920030_4:245`. All ten are STAY in cells E/F.

### 6.3 Why some rows have more than 154 state tokens

The graph budget is on `total_tokens`, not on state tokens. The option branch costs 74 tokens for two
legal actions, 88 for three and 102 for four (verified on all 25,427 rows:

| legal actions | rows | total − state |
|---:|---:|---:|
| 2 | 682 | 74 |
| 3 | 15,566 | 88 |
| 4 | 9,179 | 102 |

). Consequently 48 rows carry 155–159 state tokens (38 train, 10 test) — they all have two or three
legal actions. The largest is `…ep_900082_2:175` with 159 state tokens and 247 total.

### 6.4 Independent re-tokenisation (the L256 check)

Every prompt was re-encoded with `models/kev-original/tokenizer.json` (vocab 248,066) using the exact
`OnnxKevPolicy._encode` arithmetic (`SPECIAL` markers, `_user_tokens`, 1 + state + branch):

| Split | Rows | `prompt_sha256` mismatch | `state_tokens` mismatch | `total_tokens` mismatch | over 256 |
|---|---:|---:|---:|---:|---:|
| train | 16,795 | 0 | 0 | 0 | 0 |
| validation | 4,315 | 0 | 0 | 0 | 0 |
| test | 4,317 | 0 | 0 | 0 | 0 |

The 59 human rows were additionally re-rendered from `state_before` and produce identical text
(59/59), so the stored prompt equals what the live inference path would build.

### 6.5 Over-budget teacher candidates

20 teacher candidates were rejected for budget (see §3.2). All had four legal actions, which fixes the
branch at 102 tokens and the state budget at 154; they needed 155–157 state tokens (257–259 total).
The recorded `state_tokens` and `total_tokens` for these rejected candidates are `0`, so their exact
content is not recoverable.

---

## 7. Contextual diversity and what the test actually measures

### 7.1 The scenario grid (from `seed_plan.json` and `make_seed_plan.py`)

| Cell | Density | Duration | Train seeds | Validation seeds | Test seeds |
|---|---:|---:|---:|---:|---:|
| A | 0.7 | 60 s | 8 | 2 | 2 |
| B | 1.0 | 60 s | 20 | 4 | 4 |
| C | 1.5 | 60 s | 22 | 10 | 10 |
| D | 1.5 | 90 s | 10 | 4 | 4 |
| E | 2.5 | 60 s | 38 | 6 | 6 |
| F | 3.5 | 60 s | 30 | 6 | 6 |
| H (human) | 0.7 | — | 3 | 0 | 0 |
| **total** | | | **131** | **32** | **32** |

### 7.2 Action mix per cell (row shares)

| Cell | train STAY / BRAKE / turn % | test STAY / BRAKE / turn % | per-cell action TV |
|---|---|---|---|
| A | 92.3 / 5.5 / 2.1 | 96.9 / 0.8 / 2.3 | 0.050 |
| B | 90.9 / 6.2 / 2.9 | 89.3 / 8.0 / 2.7 | 0.019 |
| C | 82.3 / 14.1 / 3.6 | 83.5 / 13.7 / 2.8 | 0.012 |
| D | 82.3 / 14.3 / 3.4 | 85.6 / 10.8 / 3.5 | 0.037 |
| E | 80.2 / 17.4 / 2.4 | 80.5 / 16.9 / 2.6 | 0.007 |
| F | 80.2 / 17.4 / 2.4 | 79.5 / 17.9 / 2.6 | 0.008 |
| H | 0.0 / 0.0 / 100.0 | — | — |

Within a cell the conditional behaviour is almost identical (TV 0.7–5.0 percentage points).

### 7.3 Train vs test composition

| Dimension | train | test | Total-variation distance |
|---|---:|---:|---:|
| Cell A share | 6.14 % | 5.95 % | |
| Cell B | 15.35 % | 11.93 % | |
| Cell C | 16.89 % | 29.86 % | |
| Cell D | 10.49 % | 17.74 % | |
| Cell E | 28.26 % | 16.63 % | |
| Cell F | 22.52 % | 17.88 % | |
| **cell mix** | | | **0.2023** |
| Density 0.7 | 6.49 % | 5.95 % | |
| Density 1.0 | 15.35 % | 11.93 % | |
| Density 1.5 | 27.38 % | 47.60 % | |
| Density 2.5 | 28.26 % | 16.63 % | |
| Density 3.5 | 22.52 % | 17.88 % | |
| **density mix** | | | **0.2023** |

### 7.4 Is the test “unseen scenarios” or “unseen seeds”?

**Unseen seeds under the same scenario family, with a materially re-weighted mix.**

Evidence for “same family”: identical six-cell grid with identical densities and durations, identical
seed-plan generator, identical teacher policy and prompt format, same action space, and near-identical
per-cell action distributions (§7.2). No test seed or episode appears in train or validation
(32 seeds, all 92xxxx).

Evidence against “same distribution”: validation and test were drawn with the same allocation
(2/4/10/4/6/6 seeds) while train accumulated 8/20/22/10/38/30 — the test set is 47.6 % density-1.5
rows vs 27.4 % in train, and 16.6 % density-2.5 vs 28.3 %. One fifth of the test's probability mass
sits in cells the model saw at a very different rate. Test row counts per cell therefore differ from
train by 20.2 TV points even though the conditional policy per cell is the same.

Practical consequence: a test score measures generalisation to new traffic seeds *and* a shifted
scenario mix. The shift over-weights the mid-density 1.5 cells (C and D: 47.6 % of test rows vs 27.4 %
of train) and under-weights the highest-density cells (E at 2.5 and F at 3.5), so a test score is not
directly comparable to a uniform draw from the train mix.

---

## 8. Why 105 turn examples across 73 seeds were probably insufficient

Everything below is computed from the delivered subset, the full train split and the delivered
predictions; the causal claim is flagged as interpretation in §9.

### 8.1 Absolute count

| Quantity | Value |
|---|---:|
| Turn rows in subset | **105** of 4,096 = **2.56 %** |
| LEFT / RIGHT | 55 / 50 |
| Turn-bearing seeds | 73 of 124 |
| Seeds with exactly one turn | 51 |
| Mean turns per turn-bearing seed | 1.44 |
| Max turns in one seed | 4 (full train: 12) |
| Teacher turn rows dropped relative to full train | 465 → 105 (**−77.4 %**) |
| Seeds that lost all turns | 52 of 125 |
| Test turn rows (for comparison) | 122 of 4,317 = 2.83 % |

### 8.2 Exposure per example

The trainer ran `epochs: 1`, `batch: 1`, `accum: 8`, 512 optimizer steps on 4,096 records
(`training_config.json`, `logs/train_subset.log`). Because each record is used exactly once:

- Each turn example contributes to **exactly one** forward/backward pass in the whole run.
- Expected number of the 512 steps that contain at least one turn:
  512 × (1 − (1 − 0.02564)^8) ≈ **96 steps** (≈19 %). In those steps the turn is 1 of 8 records,
  i.e. 12.5 % of that step's loss.
- Aggregate weight of turn examples in the loss ≈ 105 / 4,096 = **2.56 %**, versus STAY 83.96 % and
  BRAKE 13.48 % (STAY + BRAKE = 97.44 %).

### 8.3 Repeated contexts: there are none

- **0 of 105** turn rows have another turn within 4 decisions in the same seed: every turn is a
  single isolated frame.
- **90 of 105** are immediately preceded (in sampled decision order) by a **STAY** row in the same
  seed; 10 by BRAKE and 5 by LEFT. With 32 sampled frames per seed and one turn in most of them, 31 of
  that seed's 32 targets are non-turns, so the single turn is a ~1-in-32 exception inside its own
  context.
- Zero duplicate `prompt_sha256` anywhere in the subset or the full corpus means the model never sees
  the same pre-turn state twice in the epoch. There is no repetition to reinforce the turn signal.
- The full-train corpus also has no repeated prompts (0 duplicate hashes in 16,795 rows), so this is a
  property of the data, not of the subset.

### 8.4 The fine-tune dropped the richest turn source

The full train split contained 524 turn rows, including the 54-turn human episode
`ep_1501968680_2` — 10.3 % of all turns in the corpus in a single episode. The subset excludes all
human rows (correctly, to avoid teaching weaving), so the teacher-only ceiling is 465; the subset then
keeps only 105 (22.6 %). Neither the full corpus nor the subset ever contains a dense, contiguous
sequence of turn decisions that could teach the manoeuvre as a sequence.

### 8.5 What the delivered model actually did

From `results/predictions_fine_tuned_test.jsonl` (4,317 rows), independently recomputed:

| Class | Support | Recall | Predicted count |
|---|---:|---:|---:|
| STAY | 3,633 | 0.9983 | 4,198 |
| BRAKE | 562 | 0.2011 | 119 |
| LEFT | 55 | 0.0000 | 0 |
| RIGHT | 67 | 0.0000 | 0 |

- Fine-tuned accuracy 0.8663; the **always-STAY majority baseline is 0.8416** (3,633/4,317), i.e. the
  fine-tune is **+2.47 points** over a trivial constant policy. Its macro-F1 0.3145 vs 0.2285 for
  always-STAY. The original model's 0.7160 is **12.6 points below** the majority baseline.

### 8.6 Context fragmentation, not merely class imbalance

Data supporting “fragmentation” over “imbalance”:

1. **The turn *rate* barely changed** (3.12 % → 2.56 % of rows; teacher 2.78 % → 2.56 %), so the
   subset did not dilute turns relative to the corpus; the absolute count collapsed because the corpus
   itself is 76 % smaller. A pure imbalance explanation would predict the same per-example signal per
   turn, only fewer of them.
2. **Whole contexts disappeared:** 52 of 125 turn-bearing seeds have no turn at all in the subset, and
   the 73 that remain have 1.44 turns each. The model therefore learns “STAY” for 52 entire scenarios
   that in the full corpus contained a lane change.
3. **Every turn is an orphan frame** (0/105 with a neighbouring turn within 4 decisions; 90/105
   preceded by STAY). The decision to change lane is a state-conditioned event, and the state
   immediately before and after the change is never labelled as part of the same manoeuvre.
4. **One epoch, one exposure.** 105 examples × 1 exposure ≈ 96 of 512 steps touched a turn, and each
   turn is 12.5 % of the loss in those steps.
5. **The test asks for turns at a slightly higher rate (2.83 %) than the subset supplied (2.56 %)**,
   and the model answered zero of 122.

---

## 9. Verified facts, interpretations, unanswered questions

### 9.1 VERIFIED FACTS (recomputed from files)

1. 16/16 dataset hashes in `SHA256SUMS` verify; the public 7-file card set verifies; 15/15 handoff
   hashes verify and the handoff JSONL files are byte-identical to the dataset. Test hash
   `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4` agrees with `SHA256SUMS`,
   `manifest.json`, `dataset.card.json` and `checkpoint_manifest.json`.
2. 25,427 rows = 16,795 train + 4,315 validation + 4,317 test; 25,368 `safe_teacher` + 59 `human`;
   195 episodes = 195 seeds; 131/32/32.
3. Actions: STAY 21,179; BRAKE 3,484; LEFT 378; RIGHT 386. Per-split counts as in §1.3.
4. Rows at 256 total tokens: 10. Rows over 256 after independent re-tokenisation of all 25,427 prompts:
   0. 20 teacher candidates were rejected for budget (18 STAY, 2 BRAKE, all 4-option, cells F/E,
   densities 3.5/2.5).
5. Cell/density/duration grid exactly as in §7.1; test and validation share the 2/4/10/4/6/6 seed
   allocation; train is 8/20/22/10/38/30 (+3 human).
6. Teacher: 192 episodes / 21 sessions / 11,828.63 s sim / mean survival 61.6075 s; 7 crashes
   (train 4, validation 1, test 2); 518 teacher crash-episode rows, all pre-crash
   (`max sim_time < survival_time` in all nine crash episodes), all `label_quality: "ok"`.
   No post-crash row and no unsafe teacher label exists in the corpus.
7. Teacher label fidelity on a 1,545-row sample (three whole sessions): state hash, recorded action
   and `SafeTeacherPolicy` reproduction all 100 %.
8. Human: 59 rows / 1 session / 3 episodes / 29 LEFT + 30 RIGHT / 0 STAY / 0 BRAKE; 395.67 s sim,
   395.64 s inside-episode wall, 45,066.6 s session span with a 44,667.7 s idle gap; 1 excluded unsafe
   row (`target_lane_front_gap_2.87m`); 59/59 labels match `human_action`.
9. Duplicates: 0 `row_id`, `example_id`, `prompt_sha256` or `state_sha256` duplicates within any split;
   0 duplicate `(seed, tick)`; 1 cross-split prompt hash (train `…ep_900004_4:199` ↔ test
   `…ep_920001_1:115`, both STAY, 181 tokens, different state hashes); 0 cross-split state hashes.
10. Disjointness: 0 shared seeds, 0 shared episodes across splits; human rows only in train;
    0 overlap with `pilot-v1` (seeds/sessions/episodes) or `pilot-5x20`.
11. The 4,096-row subset is entirely contained in `kev_train.jsonl` (dict-identical provenance), is
    100 % teacher, contains 0 crash rows, 124 seeds (7 excluded: 4 crash + 3 human), 105 turn rows
    (55 L / 50 R) across 73 seeds, max 4 turns in a seed, 51 seeds with exactly 1 turn.
12. Turn share: full train 524/16,795 (3.12 %); teacher-only 465/16,736 (2.78 %); subset 105/4,096
    (2.56 %); test 122/4,317 (2.83 %).
13. Training: 4,096 records seen, 1 epoch, batch 1, accum 8, 512 steps, `max_state 384`
    (max observed state tokens = 159, so no truncation).
14. Delivered fine-tuned predictions: STAY 4,198; BRAKE 119; LEFT 0; RIGHT 0. Recall STAY 0.9983,
    BRAKE 0.2011, LEFT 0, RIGHT 0. Accuracy 0.8663 vs always-STAY 0.8416.

### 9.2 INTERPRETATIONS (data-supported, not proved)

1. The teacher is a conservative heuristic whose braking arrives too late in dense traffic; the seven
   crashes are consistent with the fixed 20 m / 3 s thresholds being unable to resolve a closing
   vehicle in time. The dataset nonetheless treats all pre-crash frames as valid imitation targets,
   which teaches “brake” right up to impact.
2. The 4,096-row subset is a *proportional* sample, not a turn-enriched one. Its turn rate (2.56 %) is
   essentially the teacher's own rate (2.78 %), so the fine-tune saw nearly the same turn base rate as
   the full corpus but 78 % fewer absolute examples, in 42 % fewer seeds, with no repeated context.
   I read the zero LEFT/RIGHT predictions as the expected outcome of that exposure budget rather than
   as evidence that the model cannot represent lane changes at all.
3. Because per-cell action distributions in test match train within 0.7–5.0 points while the cell mix
   differs by 20.2 points, the test is best described as “new seeds, re-weighted scenario grid”, not
   as a fully out-of-distribution scenario set. Scores on it should not be read as generalisation to
   unseen scenario *types*.
4. The single train↔test prompt identity is a renderer discretisation collision (identical text,
   different raw state). At 1/4,317 rows with the same label it is immaterial to the reported metrics.
5. The human block cannot support imitation of normal driving: it contains only turn decisions, is
   0.23 % of the corpus, and comes from one 6.6-minute session. Its train-only placement and its
   exclusion from the fine-tune are consistent with the stated intent to avoid teaching weaving.
6. The fine-tune is mostly learning the prior: 97.4 % of its training rows are STAY/BRAKE and its
   accuracy gain over always-STAY is 2.47 points, with BRAKE recall 20 % and zero turns.

### 9.3 UNANSWERED QUESTIONS

1. **Whether 105 turns is *the* cause of zero turn predictions.** A controlled run at the same row
   budget with 10–15 % turns would be needed; the delivered evidence is consistent with the hypothesis
   but does not isolate it from one-epoch training, LoRA capacity, or the loss weighting.
2. **Whether the 20 rejected over-budget candidates mattered.** Their content is not stored
   (`state_tokens`/`total_tokens` = 0), and all were STAY/BRAKE in cells E/F; their labels cannot be
   recovered from the dataset.
3. **Why the teacher crashed in 900107 and 900113 with identical action/sim tails at different
   densities.** Possibly deterministic policy behaviour on similar scenes; not investigated.
4. **Whether the human session's 44,667.7 s idle gap hides any unrecorded play.** The recording has
   no events in that interval; the telemetry file was not re-scanned event-by-event here.
5. **Whether the validation split's exact mirroring of the test allocation was intentional.** The
   seed plan shows it (`make_seed_plan.py:47-58`), but no design note states the rationale.
6. **Whether any metric beyond accuracy/F1 is stable** on the test set: per-cell and per-density
   scores were not part of the delivered evaluation and were not recomputed here beyond the confusion
   matrix.

### 9.4 Documentation defects found (minor)

1. `seed_plan.json.description` says “128 train + 32 validation + 32 test scenarios, plus **2** human
   train seeds”; the plan actually contains **3** human seeds (644099890, 1501968680, 4135897261),
   and the dataset has 3 human episodes.
2. `dataset.card.json.limits` says “Repeated empty-road STAY prompts are natural scenario repeats, not
   duplicated rows.” There are **0** empty-road prompts (`ON\nnone`) and **0** duplicate prompt hashes
   within any split; the only repeated prompt text is the single train↔test pair in §5.2.
3. The rejection string “scene observation needs 155 state tokens; only 154 fit” is accurate only for
   four-option prompts; rows with 2–3 legal actions legitimately carry up to 159 state tokens
   (§6.3). The gate itself is correct (it bounds `total_tokens`), but the message is misleading.

---

## Appendix A — File citations

| Artefact | Path |
|---|---|
| Dataset root | `LANESHIFT/cris@ds/.portable-dev/kev-v2-scene/` |
| Splits | `kev_train.jsonl`, `kev_validation.jsonl`, `kev_test.jsonl` + `.provenance.jsonl` |
| Candidates | `candidates/human.candidates.jsonl` (60), `candidates/safe_teacher.candidates.jsonl` (25,388) |
| Episode index | `episode_index.jsonl` (195) |
| Cards | `dataset.card.json`, `human.card.json`, `safe_teacher.card.json`, `manifest.json`, `seed_plan.json`, `README.md`, `SHA256SUMS` |
| Public cards | `LANESHIFT/cris@ds/datasets/kev-v2-scene/` (7 files + `SHA256SUMS`) |
| Handoff | `LANESHIFT/kev-gpu-handoff-20261105/dataset/` |
| Seed plan generator | `LANESHIFT/kev-v2-dataset-20261104/make_seed_plan.py`, `seed_plan.json` |
| Builder | `LANESHIFT/cris@ds/python/build_kev_v2_scene.py` (turn safety `:80-117`; encode gate `:231-249`; inclusion `:275-289`) |
| Teacher policy | `LANESHIFT/cris@ds/python/policies/safe_teacher.py` |
| Prompt encoding | `LANESHIFT/cris@ds/python/policies/onnx_kev.py:41-53`, `:213-283` |
| Renderer | `LANESHIFT/cris@ds/python/kev_state.py:160-175`, `:180-…`, `:441-…` |
| Tokenizer | `LANESHIFT/cris@ds/models/kev-original/tokenizer.json` (`fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`) |
| Subset | `LANESHIFT/kev-deliverables-20261105/subset/{subset_train.jsonl,subset_train.provenance.jsonl,subset.card.json}` |
| Subset builder | `LANESHIFT/kev-subset-20261105/make_subset.py` |
| Training config/log | `LANESHIFT/kev-deliverables-20261105/run/training_config.json`, `logs/train_subset.log` |
| Predictions / metrics | `LANESHIFT/kev-deliverables-20261105/results/{predictions_fine_tuned_test.jsonl,predictions_original_test.jsonl,eval_metrics.json,checkpoint_manifest.json}` |
| Human recording | `LANESHIFT/cris@ds/.portable-dev/recordings/session_20261103_192529_340/{manifest.json,events.jsonl,episodes.jsonl}` |
| Pilot | `LANESHIFT/cris@ds/datasets/pilot-v1/` (1,348 rows), `LANESHIFT/cris@ds-training/docs/pilot-5x20.*` |

## Appendix B — Reproduction

```pwsh
$py = "C:\Users\Dhane\dev\cris@ds\LANESHIFT\.venv-host\Scripts\python.exe"
cd C:\Users\Dhane\dev\cris@ds\LANESHIFT\final-audit-data-20261105
& $py audit_core.py          # hashes, counts, actions, cells, densities, terminations
& $py audit_tokens.py        # token distribution, 256-token rows, question overhead
& $py audit_questions.py     # 2/3/4-option branch cost
& $py audit_dupes_turns.py   # duplicates, leakage, turns, human block
& $py audit_teacher.py       # crashes, pre/post-crash, 20+1 exclusions
& $py audit_distribution.py  # cell x density, per-cell action mix, duplicate prompt text
& $py audit_subset.py        # 4,096-row subset vs full train
& $py audit_episodes.py      # episode stats, crash tails, pilot inventory
& $py audit_budget.py        # turn arithmetic, seed skew, majority baseline
& $py audit_tokenize.py      # independent re-tokenisation of all 25,427 prompts
& $py audit_label_fidelity.py # teacher labels re-derived from recordings (1,545 rows)
```

Runtime: ~15 s for the whole set except `audit_label_fidelity.py` (~60 s, parses 23 MB of events).
