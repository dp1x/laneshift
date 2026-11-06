# LANESHIFT — Independent Statistics and Gameplay Audit

**Date:** 2026-11-05
**Auditor role:** independent statistics and gameplay audit (read-only)
**Workspace:** `C:\Users\Dhane\dev\cris@ds\LANESHIFT`

This audit recomputed every offline metric from raw test labels and prediction
probabilities, rebuilt both 12-seed gameplay sessions from the raw session
recordings, and re-derived every paired statistic. Nothing was trained,
modified, deleted or relabelled. All writes are confined to
`LANESHIFT\final-audit-stats-20261105\`.

---

## 0. Bottom line

| Question | Answer |
|---|---|
| Do the offline numbers in `eval_metrics.json` reproduce? | **Yes, exactly.** Every reported quantity matches to the published 4 decimals from raw labels + probabilities. |
| Is the offline improvement real? | **Yes, and it is large and consistent.** Accuracy 0.7160 → 0.8663 (+0.1503 pooled, 95% seed-cluster CI [+0.1080, +0.1962]); all 32 seeds improve. |
| Is the offline improvement substantive? | **Only partly.** The fine model never predicts LEFT or RIGHT (0 of 4,317 rows) and recovers 20% of BRAKE. It beats a constant-STAY predictor by only **+2.48 pp** accuracy (95% CI [+1.98, +3.13]) and +0.086 macro F1 (pooled). |
| Does any of it carry into gameplay? | **No convincing evidence.** Mean survival +1.07 s (95% CI [−1.38, +3.71]), 5 wins / 4 losses / 3 ties, sign test p = 1.00, exact sign-flip p = 0.46. Median difference 0.000 s. All 12 episodes crash in both arms. |
| Is the gameplay comparison confounded? | **Yes, in one direction only.** The fine arm's latency is 37% higher (1.581 s vs 1.170 s), which removed 3 of its 12 episodes from control entirely — but every fine action was STAY, which is a physical no-op, so its outcomes are exactly the do-nothing trajectories. The original arm's lane changes, in contrast, were executed on 2.4 s-stale observations. |
| Is the earlier "rejected actions 5 vs 0" claim correct? | **No.** Raw recordings contain **3** rejection events, of which **1** is a policy-action rejection. The CSV's 5 is a derived double count. |

---

## 1. Inputs, provenance and integrity

### 1.1 Data actually used

| Input | Rows / size | Role |
|---|---|---|
| `cris@ds\.portable-dev\kev-v2-scene\kev_test.jsonl` | 4,317 | **raw labels** (`questions.next_action.label`) |
| `…\kev_test.provenance.jsonl` | 4,317 | seed, cell, density, episode metadata |
| `kev-deliverables-20261105\results\predictions_original_test.jsonl` | 4,317 | probabilities |
| `kev-deliverables-20261105\results\predictions_fine_tuned_test.jsonl` | 4,317 | probabilities |
| `…\results\eval_metrics.json` | — | **cross-check only** |
| `cris@ds\.portable-dev\recordings\session_20261105_201308_176` | 166 events, 1,241 telemetry rows, 12 episodes | original arm, raw |
| `…\session_20261105_201408_325` | 160 events, 1,434 telemetry rows, 12 episodes | fine arm, raw |
| `gameplay-eval-20261105\original-12.*`, `fine-12.*` | — | **cross-check only** |

Test split: 32 seeds (920001–920032), 73–193 rows per seed, cells A–F,
densities 0.7/1.0/1.5/2.5/3.5, all labels from `safe_teacher`.
Gameplay seeds 930001–930012 are **disjoint** from every dataset split
(train 900001+, validation 910001–910032, test 920001–920032), so the gameplay
run is a genuine held-out evaluation.

### 1.2 Integrity checks that passed

* Prediction files are row-aligned with `kev_test.jsonl`: identical order, identical
  `example_id`, identical labels (`label` fields agree 4,317/4,317; the provenance
  `action` agrees with the prediction label for every row).
* Every probability vector sums to 1 (max deviation 1e-6) and contains only the
  offered actions. Offered sets are dynamic: 2–4 options
  (`LEFT,STAY,RIGHT,BRAKE` 1,813; `LEFT,STAY,BRAKE` 1,233; `STAY,RIGHT,BRAKE` 1,153;
  `STAY,BRAKE` 118). LEFT is offered in 3,046 rows and RIGHT in 2,966 rows.
* Both gameplay sessions use **identical initial conditions** for all 12 seeds
  (same `config`, same `initial_traffic` list, same `next_spawn_z`), verified by
  exact dictionary equality on the `scenario_initialized` event.
* Game config in both arms: `base_speed 20, max_speed 28, episode_duration 60,
  traffic_density 1.5, spawn_interval 55–95, traffic_speed 13–21`.
* Simulated time base is exactly **2.0×** wall clock in both sessions
  (`sim_time / physics_tick = 1/30 s`), consistent with the reported
  `sim_speed_scale 2.0`.
* Recorded `model_sha256` in every decision matches the model files on disk:
  original `031a03bd…3129`, fine `b952923c…3a28`. Data blobs, adapter and head
  hashes also match the claimed values (Table J).
* The simulation is deterministic given seed and actions: on the three seeds
  where **both** arms only ever issued STAY (930004, 930006, 930007), survival,
  distance, score and overtake count are identical to the last decimal despite
  different decision counts (3 vs 2, 2 vs 2, 2 vs 0).
* Score identity holds exactly in every episode:
  `score = 0.1·distance + 5·survival_time + 25·overtaken_count` (max abs error 3.6e-13).
* Every requested action was legal in the decision state; every logged
  probability argmax equals the requested action.
* Both arms used `prompt_mode = scene`, `state_truncated = false`,
  total tokens 212–251 (within the 256 limit).

### 1.3 Things that are *not* clean

* The two model directories ship **different tokenizer files**
  (`kev-original/tokenizer.json` 12.8 MB, vocab 248,066;
  `kev-finetuned-subset-e1/tokenizer.json` 20.0 MB, vocab 248,077). Recorded
  `state_tokens` match both tokenizers for all 41 logged states, so no
  tokenisation divergence is observable in this run — but the two arms were not
  demonstrably run with the same tokenizer file.
* The two smoke sessions are **not comparable**: `session_20261105_201150_256`
  is seed 930001, `session_20261105_201248_185` is seed 930002, both at
  density 0.7 / 15 s. They must not be used as a model comparison.
* 58 of 16,795 training rows carry `scenario_seed` values far outside the
  900xxx scheme (1,501,968,680; 4,135,897,261; and a maximum of 644,099,890).
  The test split is clean; this is a training-data seed-recording anomaly noted
  out of scope.

---

## 2. Offline metrics — independent recalculation

All metrics below were computed from `kev_test.jsonl` labels and the probability
vectors in the prediction files. Class order is `[LEFT, STAY, RIGHT, BRAKE]`
(the order used by `eval_metrics.json`, recovered from its confusion matrix).
Non-offered actions are assigned probability 0. Log loss is natural log.
Brier is the multi-class Brier score `mean_rows Σ_c (p_c − y_c)²`.
ECE uses 10 equal-width bins on the top-1 probability.

#### Table A. Pooled offline metrics (4,317 rows, 32 test seeds)

| Metric | Original | Fine-tuned | Fine − original |
|---|---|---|---|
| Accuracy | 0.7160 | 0.8663 | 0.1503 |
| Macro F1 (4-class) | 0.2143 | 0.3145 | 0.1002 |
| Weighted F1 | 0.7093 | 0.8228 | 0.1135 |
| Log loss | 0.9268 | 0.4093 | -0.5175 |
| Brier (sum over classes) | 0.5417 | 0.2090 | -0.3327 |
| ECE (10 bins, max-prob) | 0.2876 | 0.0395 | -0.2481 |
| MCE | 0.6930 | 0.1394 | -0.5536 |
| Mean confidence | 0.4647 | 0.9059 | 0.4412 |
| Top-2 accuracy | 0.9166 | 0.9717 | 0.0551 |
| Mean true-action rank | 1.3753 | 1.1696 | -0.2057 |
| Median true-action rank | 1 | 1 | — |
| MRR | 0.8434 | 0.9278 | 0.0844 |
| Mean log-odds of true action | -0.3744 | 2.1435 | 2.5179 |
| Mean log p(true) | -0.9268 | -0.4093 | 0.5175 |

Definitions: *true-action rank* is the position of the true action in the
probability ordering (1 = top-1); *MRR* is the mean reciprocal rank; *top-2* means
the true action is among the two highest-probability actions; *mean log-odds of
the true action* is the row-mean of `log(p_true / (1 − p_true))`; *MCE* is the
largest per-bin calibration gap.

#### Table B. Per-action precision / recall / F1 (support from raw labels)

| Action | Support | Orig P | Orig R | Orig F1 | Fine P | Fine R | Fine F1 |
|---|---|---|---|---|---|---|---|
| LEFT | 55 | 0.0080 | 0.0909 | 0.0147 | 0.0000 | 0.0000 | 0.0000 |
| STAY | 3633 | 0.8359 | 0.8494 | 0.8426 | 0.8640 | 0.9983 | 0.9263 |
| RIGHT | 67 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BRAKE | 562 | 0.0000 | 0.0000 | 0.0000 | 0.9496 | 0.2011 | 0.3319 |

#### Table C. Confusion matrices (rows = true, cols = LEFT, STAY, RIGHT, BRAKE)

|  | O:L | O:S | O:R | O:B | F:L | F:S | F:R | F:B |
|---|---|---|---|---|---|---|---|---|
| true LEFT | 5 | 50 | 0 | 0 | 0 | 55 | 0 | 0 |
| true STAY | 547 | 3086 | 0 | 0 | 0 | 3627 | 0 | 6 |
| true RIGHT | 8 | 59 | 0 | 0 | 0 | 67 | 0 | 0 |
| true BRAKE | 65 | 497 | 0 | 0 | 0 | 449 | 0 | 113 |

### 2.1 Agreement with the shipped `eval_metrics.json`

Every field reproduces **exactly** at the published precision:

| Field | Original (mine / reported) | Fine (mine / reported) |
|---|---|---|
| accuracy | 0.7160 / 0.7160 | 0.8663 / 0.8663 |
| macro_f1 | 0.2143 / 0.2143 | 0.3145 / 0.3145 |
| log_loss | 0.9268 / 0.9268 | 0.4093 / 0.4093 |
| brier | 0.5417 / 0.5417 | 0.2090 / 0.2090 |
| ece | 0.2876 / 0.2876 | 0.0395 / 0.0395 |
| mean_confidence | 0.4647 / 0.4647 | 0.9059 / 0.9059 |
| confusion_matrix | identical | identical |
| per_class | identical | identical |
| actual_counts | identical | identical |
| calibration_bins | identical (5 non-empty bins each) | identical (5 non-empty bins each) |
| predicted_counts | identical values | identical values |

The only difference is representational: the reported `predicted_counts`
dictionaries include zero-count classes (`"LEFT": 0, "RIGHT": 0` for the fine
model); this recomputation omits zero keys. No value differs.

### 2.2 What the offline gain actually consists of

| | Original | Fine-tuned | Constant STAY |
|---|---|---|---|
| Accuracy | 0.7160 | 0.8663 | **0.8416** |
| Macro F1 (4-class) | 0.2143 | 0.3145 | **0.2285** |
| Balanced accuracy (mean recall) | 0.2351 | 0.2999 | 0.2500 |
| LEFT recall | 0.0909 (5/55) | **0.0000 (0/55)** | 0 |
| RIGHT recall | 0.0000 (0/67) | **0.0000 (0/67)** | 0 |
| BRAKE recall | 0.0000 (0/562) | 0.2011 (113/562) | 0 |
| STAY recall | 0.8494 | 0.9983 | 1.0000 |

Verified facts:

* The fine model **never emits LEFT or RIGHT in 4,317 rows**, although LEFT was
  legal in 3,046 and RIGHT in 2,966 of them. Its entire action vocabulary is
  STAY (4,198 predictions) and BRAKE (119).
* Against a constant-STAY predictor the fine model gains **+0.0248 accuracy**
  (95% seed-bootstrap CI [+0.0198, +0.0313]; 30 wins, 0 losses, 2 ties) and
  **+0.086 macro F1** pooled (+0.101 mean over seeds, CI [+0.079, +0.126]).
* The original is **worse than constant STAY** on accuracy (0.716 vs 0.842): it
  predicts LEFT 625 times and is right 5 times (precision 0.008).
* The fine model's gain over the original is therefore mostly *stopping the
  original's pathological LEFT prediction* and secondarily *partial BRAKE
  detection*. It is not evidence of lane-change competence; it is evidence of a
  STAY prior.

---

## 3. Offline per-seed statistics (32 test seeds)

#### Table D. Per-seed offline metrics (32 held-out test seeds)

| Seed | Rows | Orig acc | Fine acc | Orig F1 | Fine F1 | Orig LL | Fine LL | Orig ECE | Fine ECE |
|---|---|---|---|---|---|---|---|---|---|
| 920001 | 128 | 0.8906 | 0.9609 | 0.3140 | 0.3267 | 0.8750 | 0.2233 | 0.4592 | 0.0223 |
| 920002 | 129 | 0.8295 | 0.9922 | 0.3023 | 0.6653 | 0.8292 | 0.0683 | 0.4003 | 0.0235 |
| 920003 | 128 | 0.5859 | 0.8438 | 0.1904 | 0.3081 | 0.9906 | 0.4801 | 0.1835 | 0.0642 |
| 920004 | 129 | 0.7907 | 0.9147 | 0.2217 | 0.3387 | 0.8348 | 0.3800 | 0.3331 | 0.0616 |
| 920005 | 129 | 0.6899 | 0.9535 | 0.2051 | 0.4257 | 0.9352 | 0.2097 | 0.3071 | 0.0202 |
| 920006 | 129 | 0.7442 | 0.9767 | 0.2133 | 0.4969 | 0.8816 | 0.1408 | 0.3537 | 0.0086 |
| 920007 | 128 | 0.7734 | 0.8750 | 0.2181 | 0.3669 | 0.8953 | 0.4368 | 0.3733 | 0.0553 |
| 920008 | 129 | 0.7519 | 0.8295 | 0.2913 | 0.3755 | 0.9279 | 0.4620 | 0.3144 | 0.0502 |
| 920009 | 129 | 0.4264 | 0.9147 | 0.1597 | 0.3636 | 0.9616 | 0.3775 | 0.0992 | 0.0418 |
| 920010 | 129 | 0.6589 | 0.7829 | 0.2063 | 0.2928 | 1.0330 | 0.5307 | 0.2294 | 0.1407 |
| 920011 | 129 | 0.5271 | 0.9225 | 0.1726 | 0.3824 | 0.8854 | 0.3264 | 0.1570 | 0.0226 |
| 920012 | 129 | 0.7287 | 0.8682 | 0.2361 | 0.3152 | 0.8871 | 0.4863 | 0.2814 | 0.0761 |
| 920013 | 129 | 0.8295 | 0.8837 | 0.2652 | 0.3112 | 0.8626 | 0.4430 | 0.3759 | 0.0494 |
| 920014 | 129 | 0.7287 | 0.8605 | 0.2108 | 0.3549 | 0.9225 | 0.3720 | 0.3457 | 0.0730 |
| 920015 | 129 | 0.7054 | 0.7829 | 0.2068 | 0.2547 | 0.9495 | 0.5497 | 0.2757 | 0.1258 |
| 920016 | 129 | 0.6279 | 0.8605 | 0.1938 | 0.2623 | 0.9397 | 0.4566 | 0.1827 | 0.0752 |
| 920017 | 187 | 0.7326 | 0.8663 | 0.2311 | 0.3563 | 0.9148 | 0.5029 | 0.3067 | 0.0691 |
| 920018 | 193 | 0.3782 | 0.8860 | 0.1409 | 0.3593 | 1.0012 | 0.4375 | 0.1433 | 0.0625 |
| 920019 | 193 | 0.6528 | 0.9585 | 0.2000 | 0.4112 | 0.9144 | 0.1640 | 0.2489 | 0.0303 |
| 920020 | 193 | 0.7202 | 0.8238 | 0.2310 | 0.2559 | 1.0006 | 0.5025 | 0.2944 | 0.0672 |
| 920021 | 73 | 0.7945 | 0.8904 | 0.2266 | 0.3906 | 0.9045 | 0.3536 | 0.4104 | 0.0369 |
| 920022 | 129 | 0.8140 | 0.8372 | 0.2253 | 0.2955 | 0.9188 | 0.5106 | 0.3750 | 0.0525 |
| 920023 | 129 | 0.7674 | 0.7984 | 0.2895 | 0.3953 | 0.9132 | 0.5156 | 0.3112 | 0.0824 |
| 920024 | 129 | 0.8140 | 0.8450 | 0.2244 | 0.2763 | 0.7983 | 0.4218 | 0.3159 | 0.0501 |
| 920025 | 129 | 0.7984 | 0.8295 | 0.2229 | 0.2698 | 0.9877 | 0.4807 | 0.3718 | 0.0419 |
| 920026 | 129 | 0.7674 | 0.8372 | 0.2210 | 0.3267 | 0.9779 | 0.4149 | 0.3877 | 0.0547 |
| 920027 | 128 | 0.7969 | 0.8438 | 0.2247 | 0.3681 | 0.9502 | 0.3809 | 0.3519 | 0.0518 |
| 920028 | 129 | 0.7829 | 0.8062 | 0.2928 | 0.3656 | 0.8346 | 0.4319 | 0.3040 | 0.1027 |
| 920029 | 129 | 0.6977 | 0.8062 | 0.2083 | 0.3100 | 1.0198 | 0.5165 | 0.3046 | 0.0617 |
| 920030 | 128 | 0.8047 | 0.8594 | 0.2269 | 0.3020 | 0.9885 | 0.4038 | 0.4048 | 0.0452 |
| 920031 | 129 | 0.7752 | 0.8062 | 0.2193 | 0.2780 | 0.9158 | 0.5112 | 0.3190 | 0.0473 |
| 920032 | 129 | 0.7519 | 0.7829 | 0.2156 | 0.2561 | 0.9345 | 0.5993 | 0.2937 | 0.0863 |

#### Table E. Offline paired differences over the 32 seeds (fine − original)

| Metric | Orig mean | Fine mean | Mean diff | 95% seed-bootstrap CI | Median diff | W/L/T | Sign test p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.7231 | 0.8656 | +0.1426 | [+0.1015, +0.1901] | +0.1026 | 32/0/0 | 4.66e-10 | 8.28e-07 |
| macro_f1 | 0.2252 | 0.3456 | +0.1203 | [+0.0947, +0.1489] | +0.1037 | 32/0/0 | 4.66e-10 | 4.66e-10 |
| weighted_f1 | 0.7072 | 0.8200 | +0.1128 | [+0.0858, +0.1448] | +0.0891 | 32/0/0 | 4.66e-10 | 4.66e-10 |
| log_loss | 0.9246 | 0.4091 | -0.5155 | [-0.5560, -0.4771] | -0.5028 | 0/32/0 | 4.66e-10 | 4.66e-10 |
| brier | 0.5405 | 0.2094 | -0.3311 | [-0.3572, -0.3059] | -0.3105 | 0/32/0 | 4.66e-10 | 4.66e-10 |
| ece | 0.3067 | 0.0579 | -0.2488 | [-0.2803, -0.2161] | -0.2650 | 0/32/0 | 4.66e-10 | 4.66e-10 |
| mean_confidence | 0.4656 | 0.9053 | +0.4396 | [+0.4221, +0.4580] | +0.4265 | 32/0/0 | 4.66e-10 | 4.66e-10 |
| predicted_stay_rate | 0.8637 | 0.9719 | +0.1081 | [+0.0645, +0.1574] | +0.0620 | 25/5/2 | 0.000325 | 5.73e-05 |

Table E averages the 32 per-seed values, so each seed has equal weight; the
pooled figures below weight every row equally. Both give the same conclusion.

### 3.1 Treating rows within a seed as correlated

Rows inside a seed come from one episode and are strongly autocorrelated in time.
The seed is the resampling unit throughout.

| Diagnostic | Original | Fine |
|---|---|---|
| Intra-cluster correlation of the correctness indicator | 0.0552 | 0.0218 |
| Design effect `1 + (n̄−1)·ICC` | **8.38** | **3.91** |
| Effective sample size (4,317 rows) | ≈ 515 | ≈ 1,104 |
| Naive row-level SE of accuracy | 0.0069 | 0.0052 |

Pooled cluster bootstrap (20,000 resamples of the 32 seeds, both arms on the same
resampled seeds):

| Quantity | Observed | 95% CI | Bootstrap SE |
|---|---|---|---|
| Accuracy difference | +0.1503 | [+0.1080, +0.1962] | 0.0227 |
| Macro-F1 difference | +0.1002 | [+0.0805, +0.1211] | 0.0104 |
| Log-loss difference | −0.5175 | [−0.5566, −0.4806] | 0.0195 |
| Brier difference | −0.3327 | [−0.3581, −0.3086] | 0.0127 |

The naive row-level SE of the accuracy difference would be 0.0086 — a factor of
**2.6 too small** relative to the cluster bootstrap SE. Ignoring clustering would
have overstated precision; the conclusion survives regardless because the effect
is large.

Sign test and Wilcoxon signed-rank on the 32 paired per-seed values:

| Metric | Wins/Losses/Ties | Sign test p | Wilcoxon p |
|---|---|---|---|
| Accuracy | 32/0/0 | 4.7e-10 | 8.3e-07 (normal approx; ties present) |
| Macro F1 | 32/0/0 | 4.7e-10 | 4.7e-10 (exact) |
| Log loss | 0/32/0 | 4.7e-10 | 4.7e-10 (exact) |
| Brier | 0/32/0 | 4.7e-10 | 4.7e-10 (exact) |
| ECE | 0/32/0 | 4.7e-10 | 4.7e-10 (exact) |
| Mean confidence | 32/0/0 | 4.7e-10 | 4.7e-10 (exact) |

**Interpretation.** The offline improvement is unambiguous and robust to seed
clustering. It is also *not* uniform in nature: every seed improves on every
proper scoring rule, but the class-level structure shows a degenerate policy.

---

## 4. Gameplay reconstruction from raw recordings

Both sessions were rebuilt directly from `events.jsonl` (decisions, applications,
rejections, collisions, overtakes) and `telemetry.jsonl` (player state per
physics tick). The summary files were used only afterwards, for cross-checking.

#### Table F. Gameplay per-seed reconstruction (raw session logs)

| Seed | Orig surv s | Fine surv s | Δ surv | Orig dist | Fine dist | Orig score | Fine score | Orig ovt | Fine ovt | Orig LC | Fine LC | Orig dec | Fine dec | Orig lane | Fine lane | Orig lat ms | Fine lat ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 930001 | 4.067 | 4.533 | +0.467 | 103.3 | 116.4 | 30.7 | 34.3 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1058 | 1672 |
| 930002 | 5.267 | 5.100 | -0.167 | 136.9 | 132.3 | 65.0 | 63.7 | 1 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 1085 | — |
| 930003 | 4.333 | 4.733 | +0.400 | 110.8 | 122.0 | 57.7 | 60.9 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1083 | 1638 |
| 930004 | 10.800 | 10.800 | +0.000 | 291.9 | 291.9 | 258.2 | 258.2 | 7 | 7 | 0 | 0 | 3 | 2 | 1 | 1 | 1101 | 1579 |
| 930005 | 5.733 | 13.200 | +7.467 | 150.0 | 359.1 | 93.7 | 251.9 | 2 | 6 | 1 | 0 | 1 | 3 | 0 | 1 | 1126 | 1591 |
| 930006 | 7.467 | 7.467 | +0.000 | 198.5 | 198.5 | 82.2 | 82.2 | 1 | 1 | 0 | 0 | 2 | 2 | 1 | 1 | 1102 | 1589 |
| 930007 | 5.200 | 5.200 | +0.000 | 135.1 | 135.1 | 64.5 | 64.5 | 1 | 1 | 0 | 0 | 2 | 0 | 1 | 1 | 1080 | — |
| 930008 | 12.100 | 5.500 | -6.600 | 328.3 | 143.5 | 243.3 | 41.8 | 6 | 0 | 1 | 0 | 4 | 1 | 0 | 1 | 1102 | 1586 |
| 930009 | 5.800 | 12.100 | +6.300 | 151.9 | 328.3 | 94.2 | 218.3 | 2 | 5 | 1 | 0 | 1 | 3 | 0 | 1 | 1141 | 1583 |
| 930010 | 8.533 | 4.767 | -3.767 | 228.4 | 122.9 | 190.5 | 61.1 | 5 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 1193 | — |
| 930011 | 6.567 | 5.133 | -1.433 | 173.3 | 133.2 | 75.2 | 64.0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1500 | 1586 |
| 930012 | 6.767 | 16.933 | +10.167 | 178.9 | 463.6 | 126.7 | 331.0 | 3 | 8 | 1 | 0 | 1 | 5 | 1 | 1 | 1577 | 1509 |

### 4.1 What the raw logs show

| Quantity | Original | Fine-tuned |
|---|---|---|
| Episodes / seeds | 12 / 930001–930012 | 12 / 930001–930012 |
| Terminations | 12 crash | 12 crash |
| Decisions sent by controller | 24 | 19 |
| Decisions delivered for the live episode | 22 | 19 |
| Decisions applied | 21 | 19 |
| Declined at application (illegal at application time) | 1 (LEFT, `not_legal_at_application`) | 0 |
| Rejection events in the log | 3 (2 `wrong_episode` + 1 `not_legal_at_application`) | 0 |
| Applied action mix | LEFT 9, STAY 12, RIGHT 0, BRAKE 0 | STAY 19 |
| Telemetry lane changes | 7 | 0 |
| Braking telemetry ticks | 0 | 0 |
| Final lanes | lane 0 in 7, lane 1 in 5 | lane 1 in 12 |
| Episodes with zero applied action | 0 | **3** (930002, 930007, 930010) |
| Mean survival | 6.886 s | 7.956 s |
| Median survival | 6.183 s | 5.350 s |
| Mean distance | 182.28 | 212.22 |
| Mean score | 115.16 | 127.67 |
| Mean overtaken | 2.50 | 2.67 |
| Total overtake events | 30 | 32 |

All applied actions were legal in the state they were decided on. Every logged
probability vector has its argmax equal to the requested action.

### 4.2 Cross-check against the summary files

#### Table K. Cross-check of summary files against raw recordings

| Item | Reported vs raw | Explanation |
|---|---|---|
| Summary JSONL vs raw `episodes.jsonl` | 0 differing fields | 12 fields × 12 episodes × 2 arms compared; all equal |
| CSV `action_decisions_requested` | 24 vs 22 | 24 = server-side decisions sent; raw events contain 22 `action_decision` records + 2 `external_action_rejected` (`wrong_episode`) |
| CSV `action_decisions_rejected` | 5 vs 3 | CSV uses max(rejected_actions + protocol_rejections, requested − applied) = max(3+2, 3) = 5; raw has 3 rejection events (2 wrong_episode STAY + 1 not_legal_at_application LEFT) |
| CSV `decision_latency_ms_mean` | 1170.42 vs 1148.35 | CSV averages all 24 delivered decisions; 1148.35 is the mean over the 21 applied model latencies |
| CSV `actions_taken` / action_counts | 21; LEFT 9 / STAY 12 | matches raw applied actions exactly |
| Final lanes / crashes / scores / distances | match | verified per episode |
| Smoke sessions | seeds 930001 vs 930002, density 0.7, 15 s | the two smoke runs are not seed-matched and use a different density; they cannot be compared |
| Tokenizer files | different bytes | kev-original/tokenizer.json 12.8 MB vs kev-finetuned-subset-e1/tokenizer.json 20.0 MB; recorded token counts match both for every logged state, so no observed tokenisation divergence |

The summary `*.episodes.jsonl` files agree with the raw `episodes.jsonl` on all
12 fields × 12 episodes × 2 arms. The CSVs disagree on three derived fields and
one claim in `GAMEPLAY-COMPARISON.md`:

1. `action_decisions_requested` 24 vs 22 `action_decision` records. Both are
   correct under different definitions: 24 decisions were sent by the controller,
   22 were delivered while their episode was still live, and 2 arrived after the
   episode had changed (`wrong_episode`) and were never applied. Both late
   decisions were STAY.
2. `action_decisions_rejected` 5 vs 3. The CSV formula is
   `max(rejected_actions + protocol_rejections, requested − applied)` =
   `max(3 + 2, 3) = 5`; the two `protocol_rejections` are the *same two*
   `wrong_episode` events already counted in `rejected_actions`. **The claim
   "five of nine lane changes rejected" is not supported by the recordings.**
   The correct statement is: 10 LEFT decisions, 9 applied, **1** rejected because
   it was no longer legal at application time, plus 2 late STAY decisions.
3. `decision_latency_ms_mean` 1170.42 vs 1148.35: the CSV averages all 24 sent
   decisions; 1148.35 is the mean model latency of the 21 applied decisions.
4. `GAMEPLAY-COMPARISON.md` reports "Rejected actions 5" for the original; the
   raw count is 3 (1 policy-level).

Everything else in the summary (survival, distance, score, overtakes, final
lanes, crashes, action counts, model hashes) is confirmed exactly.

### 4.3 Crash forensics

#### Table M. Final-tick state of every episode (what the vehicle was doing when it crashed)

| Arm | Seed | Lane | Lateral x | Target lane | Change in progress | Nearest same-lane gap m | Closing speed m/s |
|---|---|---|---|---|---|---|---|
| original | 930001 | 0 | 4.00 | 0 | False | 2.94 | 13.76 |
| original | 930002 | 0 | 4.00 | 0 | False | 2.91 | 11.07 |
| original | 930003 | 1 | 3.00 | 0 | True | 8.56 | 10.67 |
| original | 930004 | 1 | 0.00 | 1 | False | 4.19 | 12.70 |
| original | 930005 | 0 | 4.00 | 0 | False | 4.20 | 14.39 |
| original | 930006 | 1 | 0.00 | 1 | False | 3.04 | 7.11 |
| original | 930007 | 1 | 0.00 | 1 | False | 4.08 | 10.64 |
| original | 930008 | 0 | 4.00 | 0 | False | 2.92 | 7.42 |
| original | 930009 | 0 | 4.00 | 0 | False | 4.25 | 12.72 |
| original | 930010 | 0 | 4.00 | 0 | False | 3.91 | 12.39 |
| original | 930011 | 0 | 4.00 | 0 | False | 2.72 | 11.81 |
| original | 930012 | 1 | 3.50 | 0 | True | 74.18 | 7.02 |
| fine | 930001 | 1 | 0.00 | 1 | False | 2.75 | 11.21 |
| fine | 930002 | 1 | 0.00 | 1 | False | 2.89 | 14.77 |
| fine | 930003 | 1 | 0.00 | 1 | False | 4.30 | 10.67 |
| fine | 930004 | 1 | 0.00 | 1 | False | 4.19 | 12.70 |
| fine | 930005 | 1 | 0.00 | 1 | False | 2.68 | 13.44 |
| fine | 930006 | 1 | 0.00 | 1 | False | 3.04 | 7.11 |
| fine | 930007 | 1 | 0.00 | 1 | False | 4.08 | 10.64 |
| fine | 930008 | 1 | 0.00 | 1 | False | 4.13 | 9.68 |
| fine | 930009 | 1 | 0.00 | 1 | False | 4.03 | 9.35 |
| fine | 930010 | 1 | 0.00 | 1 | False | 2.92 | 11.64 |
| fine | 930011 | 1 | 0.00 | 1 | False | 4.28 | 9.72 |
| fine | 930012 | 1 | 0.00 | 1 | False | 2.84 | 7.02 |

Verified facts:

* **Every** episode in both arms ends in a rear-end or lane-change collision at
  full speed (28 m/s). Neither arm ever braked (0 brake actions, 0 braking
  telemetry ticks).
* All 12 fine crashes are plain rear-end collisions into a slower same-lane
  vehicle, with 2.7–4.3 m of gap and 7.0–14.8 m/s closing speed at the final tick.
* 7 of the original's 12 crashes are rear-ends in **lane 0**, the lane it moved
  into. 2 are mid-lane-change collisions (930003 at lateral x = 3.00 of 4.00;
  930012 at x = 3.50) with a lane-0 vehicle 0.4 m alongside and 2.9 m ahead
  respectively. 3 are rear-ends in lane 1 on the seeds where it never changed lane.
* The fine model's 12 crashes are the *same* trajectories the original would have
  had without lane changes: on the three seeds where the original also stayed
  (930004, 930006, 930007) the two arms produce bit-identical outcomes.

**Consequence.** This is not a comparison of two driving policies in the ordinary
sense. STAY is a physical no-op in this simulation (it only clears braking and
sets an action timer). The fine model's measured behaviour is exactly the
"do nothing, accelerate to 28 m/s" trajectory. The entire treatment contrast is
therefore **"occasionally change lane" vs "never change lane"**.

---

## 5. Paired gameplay statistics (12 seeds)

#### Table G. Gameplay paired tests (12 seeds, fine − original)

| Metric | Orig mean | Fine mean | Mean diff | 95% bootstrap CI | Median diff | W/L/T | Sign p | Wilcoxon p | Sign-flip p |
|---|---|---|---|---|---|---|---|---|---|
| survival_time | 6.8861 | 7.9556 | +1.0694 | [-1.3750, +3.7084] | +0.0000 | 5/4/3 | 1.0000 | 0.5703 | 0.4609 |
| distance | 182.2778 | 212.2222 | +29.9444 | [-38.5000, +103.8353] | +0.0000 | 5/4/3 | 1.0000 | 0.5703 | 0.4609 |
| score | 115.1583 | 127.6667 | +12.5083 | [-48.4054, +72.8278] | +0.0000 | 5/4/3 | 1.0000 | 0.7344 | 0.7344 |
| overtaken_count | 2.5000 | 2.6667 | +0.1667 | [-1.5000, +1.7500] | +0.0000 | 3/2/7 | 1.0000 | 1.0000 | 0.9375 |
| lane_changes | 0.7500 | 0.0000 | -0.7500 | [-1.0000, -0.5000] | -1.0000 | 0/9/3 | 0.0039 | 0.0034 | 0.0039 |
| brake_actions | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0/0/12 | 1.0000 | 1.0000 | 1.0000 |
| decisions_received | 1.8333 | 1.5833 | -0.2500 | [-1.2500, +0.9167] | +0.0000 | 3/5/4 | 0.7266 | 0.6193 | 0.7734 |
| decisions_applied | 1.7500 | 1.5833 | -0.1667 | [-1.1687, +1.0000] | +0.0000 | 3/5/4 | 0.7266 | 0.8295 | 0.8828 |
| decisions_rejected | 0.2500 | 0.0000 | -0.2500 | [-0.5000, +0.0000] | +0.0000 | 0/3/9 | 0.2500 | 0.1489 | 0.2500 |
| first_decision_sim | 4.0593 | 4.4074 | +0.3481 | [-0.3704, +1.0222] | +0.1000 | 7/2/0 | 0.1797 | 0.2500 | 0.4062 |
| mean_model_latency_ms | 1198.9117 | 1592.6251 | +393.7134 | [+240.1982, +516.8515] | +478.0378 | 8/1/0 | 0.0391 | 0.0078 | 0.0078 |
| decision_gap_sim_mean | 2.1917 | 3.1833 | +0.9917 | [+0.9833, +1.0000] | +0.9917 | 2/0/0 | 0.5000 | 0.5000 | 0.5000 |
| control_coverage | 0.3697 | 0.4120 | +0.0423 | [-0.1588, +0.2711] | -0.0536 | 4/5/0 | 1.0000 | 1.0000 | 0.7422 |

### 5.1 The outcome variables are not independent

* Distance is a **deterministic function of survival time** in this simulator:
  with no braking, `v(t) = min(20 + 3t, 28)`, so `distance(t)` is fixed.
  Measured Pearson r between survival and distance across the 24 episodes is
  **1.0000** (Spearman 1.0000). The "+29.94 m distance" is not independent
  evidence; it is the same 1.07 s measured in metres.
* Score is `0.1·distance + 5·survival + 25·overtakes`, so it is also dominated by
  survival (Pearson r = 0.959).
* Overtake count is the only partially independent outcome, and its difference is
  +0.17 (CI [−1.5, +1.75]).

So the table contains one real outcome (time to crash), one rescaled copy, one
weighted copy, and one weakly independent count.

### 5.2 All-crash censoring and zero variance

Both arms crash in **12/12** episodes. The crash indicator has zero variance in
both arms: McNemar's test has **no discordant pairs** (b = 0, c = 0) and is
degenerate (p = 1.0 by convention). There is no "completion" outcome to compare,
no survival curve beyond ~17 s, and no episode that reaches the 60 s cap. The
survival times are complete (uncensored) durations for the event "crash", but the
outcome space is truncated by the fact that *every* run fails early.

### 5.3 What the paired tests say

| Metric | Mean diff | Median diff | W/L/T | Sign p | Wilcoxon p | Exact sign-flip p | Paired t p |
|---|---|---|---|---|---|---|---|
| Survival (s) | +1.069 | 0.000 | 5/4/3 | 1.000 | 0.570 | 0.461 | 0.449 |
| Distance | +29.94 | 0.000 | 5/4/3 | 1.000 | 0.570 | 0.461 | 0.449 |
| Score | +12.51 | 0.000 | 5/4/3 | 1.000 | 0.734 | 0.734 | 0.706 |
| Overtaken | +0.167 | 0.000 | 3/2/7 | 1.000 | 1.000 | 0.938 | 0.853 |
| Lane changes | −0.750 | −1.000 | 0/9/3 | **0.0039** | **0.0034** | **0.0039** | **0.00013** |
| BRAKE actions | 0.000 | 0.000 | 0/0/12 | — | — | — | — (zero variance) |
| Decisions received | −0.250 | 0.000 | 3/5/4 | 0.727 | 0.619 | 0.773 | 0.667 |
| Decisions applied | −0.167 | 0.000 | 3/5/4 | 0.727 | 0.830 | 0.883 | 0.782 |
| Mean model latency (ms) | +393.7 | +478.0 | 8/1/0 | 0.039 | 0.0078 | 0.0078 | 0.00083 |

The **only gameplay outcome** with a statistically detectable difference is the
one that is true by construction: the fine model changes lanes 0 times, the
original 9 times. BRAKE is 0 in both arms (zero variance, no test possible).
Latency also differs (p = 0.0078), but that is the cause of the different control
opportunity, not an outcome. `decision_gap_sim_mean` is only defined for the two
seeds where both arms had two or more applied decisions (930004, 930006); the
session-level cadence in Table H is the reliable figure.

### 5.4 The mean is carried by two seeds

| Seed | Δ survival (s) |
|---|---|
| 930012 | **+10.167** |
| 930005 | **+7.467** |
| 930009 | +6.300 |
| 930001 | +0.467 |
| 930003 | +0.400 |
| 930004 / 930006 / 930007 | 0.000 (both arms stayed) |
| 930002 | −0.167 |
| 930011 | −1.433 |
| 930010 | −3.767 |
| 930008 | **−6.600** |

* Total sum of differences = +12.83 s.
* The top two seeds account for **137%** of the total.
* Mean without the top two = **−0.48 s**; mean without the top three = **−1.23 s**.
* Mean of the 9 lane-change seeds = +1.43 s (SD 5.4, SE 1.8, 95% CI ≈ [−2.3, +5.1]).

Decomposed by whether the fine arm ever received control:

| Subset | n | Original mean | Fine mean |
|---|---|---|---|
| Fine had ≥ 1 applied action | 9 | 7.07 s | 8.93 s |
| Fine had **zero** applied actions | 3 | 6.33 s | 5.02 s |

The fine arm's headline mean mixes two regimes: 9 episodes with (no-op) control
and 3 episodes with no control at all.

### 5.5 Power of the design

#### Table I. Power of the 12-seed paired design (α=0.05, 80% power)

| Metric | Observed diff | SD of diffs | MDE (n=12) | Observed / MDE | n needed for observed effect | P(true diff > 0) | P(true diff > 2) |
|---|---|---|---|---|---|---|---|
| survival_time | +1.0694 | 4.7221 | 3.819 | 0.280 | 154 | 0.788 | 0.235 |
| distance | +29.9444 | 132.2201 | 106.933 | 0.280 | 154 | 0.788 | 0.771 |
| score | +12.5083 | 111.9231 | 90.518 | 0.138 | 629 | 0.657 | 0.632 |
| overtaken_count | +0.1667 | 3.0401 | 2.459 | 0.068 | 2612 | 0.566 | 0.011 |
| lane_changes | -0.7500 | 0.4523 | 0.366 | -2.050 | 3 | 0.000 | 0.000 |
| decisions_applied | -0.1667 | 2.0375 | 1.648 | -0.101 | 1174 | 0.343 | 0.000 |
| mean_model_latency_ms | +393.7134 | 227.4965 | 212.450 | 1.853 | 3 | 1.000 | 1.000 |

With n = 12 paired seeds and an SD of paired survival differences of 4.72 s, the
minimum detectable effect at 80% power (α = 0.05, two-sided) is **3.82 s**. The
observed +1.07 s is **28% of the MDE**. Reaching 80% power for the observed
effect would need **≈154 seeds**; for score, ≈629; for overtakes, ≈2,612.

Bootstrap posterior mass for the true mean survival difference (flat prior over
the observed paired differences): P(> 0) = 0.79, P(> 1 s) = 0.51, P(> 2 s) = 0.24,
P(> 5 s) = 0.002. The bootstrap median is +1.03 s with a 90% interval of
[−0.58, +2.77]. Even under a flat prior the data do not support a claim of a
practically meaningful survival gain.

---

## 6. Inference latency, cadence and control opportunity

#### Table H. Latency and decision-cadence comparison

| Quantity | Original | Fine-tuned |
|---|---|---|
| Applied decisions (game-side) | 21 | 19 |
| Delivered decisions (server) | 24 | 19 |
| States sent by game | 346 | 401 |
| Superseded states | 309 | 369 |
| Stale decisions dropped | 11 | 11 |
| Inference runs (delivered + stale) | 35 | 30 |
| Episodes with zero applied action | 0 | 3 |
| Model latency mean (all delivered) ms | 1165.1 | 1572.9 |
| Controller decision latency mean ms | 1170.4 | 1580.8 |
| Model latency mean (applied) ms | 1148.3 | 1572.9 |
| Model latency p05 / p95 (applied) ms | 1079 / 1500 | 1493 / 1642 |
| Decision→application delay (sim s) | 2.400 | 3.263 |
| Inter-decision gap (sim s) | 2.207 | 3.120 |
| Mean confidence of delivered decisions | 0.336 | 0.969 |
| Mean top1−top2 margin | 0.026 | 0.955 |

### 6.1 The 1.170 vs 1.581 s figure is correct — and it is a controller-level number

* Controller-reported mean decision latency: **1,170.4 ms** vs **1,580.8 ms**
  (+35.1%). These are the CSV values and they are correct for what they measure
  (all delivered decisions, server-side wall clock).
* Model-only latency: 1,165.1 ms vs 1,572.9 ms over delivered decisions;
  1,148.3 ms vs 1,572.9 ms over **applied** decisions (n = 21 vs 19, +37.0%).
  The per-episode means used in Table G are episode-weighted (1,198.9 vs
  1,592.6 ms); both weightings give the same conclusion.
* Because the simulator runs at 2× wall clock, a 1.17 s wall-clock decision
  arrives **2.40 s of simulated time** later; the fine model's arrives
  **3.26 s** later (63–98 vs 91–105 physics ticks). At 28 m/s that is 67 m vs
  91 m of travel on a stale observation.

### 6.2 Effect on applied decisions and cadence

| | Original | Fine |
|---|---|---|
| Delivered decisions | 24 | 19 (−21%) |
| Applied decisions | 21 | 19 (−10%) |
| Inference runs (delivered + stale-dropped) | 35 | 30 |
| Stale decisions discarded (episode changed) | 11 (31%) | 11 (37%) |
| Inter-decision gap (sim time) | 2.21 s | 3.12 s |
| Applied decisions per simulated second | 0.254 | 0.199 |
| Episodes with zero applied action | 0 | 3 |

The controller is a single-slot queue driven as fast as inference allows: it
superseded 309 vs 369 state updates, and it discarded 11 results in each arm
(`stale_decisions`). The experiment runner's default `--max-state-age` is 10 s
and the observed decision-to-latest-state ages are 2.1–3.5 s, so these drops are
**episode changes** rather than age expiries — the same mechanism that produced
the two `wrong_episode` rejections. Three fine episodes (930002, 930007, 930010)
ended before any decision returned.

### 6.3 The confound is real but asymmetric

#### Table L. Original arm: LEFT decisions and the gap in the destination lane (lane 0)

| Episode | Seq | Decision sim s | Application sim s | Delay sim s | Gap ahead at decision m | Gap ahead at application m | Gap behind at application m |
|---|---|---|---|---|---|---|---|
| ep_930001_1 | 1 | 0.47 | 2.57 | 2.10 | 45.33 | — | — |
| ep_930002_2 | 2 | 0.70 | 2.93 | 2.23 | 47.76 | 28.74 | — |
| ep_930003_3 | 4 | 1.87 | 4.13 | 2.27 | 12.03 | 2.37 | — |
| ep_930005_5 | 9 | 2.33 | 4.77 | 2.43 | 52.98 | — | — |
| ep_930008_8 | 14 | 1.63 | 3.93 | 2.30 | 78.98 | 63.48 | — |
| ep_930009_9 | 18 | 0.70 | 3.00 | 2.30 | 63.41 | 39.86 | — |
| ep_930009_9 | 19 | 2.80 | 5.30 | 2.50 | 42.41 | — | — |
| ep_930010_10 | 20 | 1.87 | 4.27 | 2.40 | 85.61 | 56.79 | — |
| ep_930011_11 | 22 | 0.70 | 3.80 | 3.10 | 25.13 | 35.39 | -4.16 |
| ep_930012_12 | 24 | 3.27 | 6.53 | 3.27 | 11.64 | 6.30 | — |

* **The original arm is handicapped by latency.** Its LEFT decisions were taken
  on observations 2.1–3.3 s of simulated time old. In 930003 the gap ahead in
  lane 0 was 12.03 m at decision and 2.37 m at application; in 930012 it fell
  from 11.64 m to 6.30 m and the vehicle crashed 0.23 s after the lane change
  was applied; in 930011 a vehicle appeared **4.16 m behind** in lane 0 during
  the delay; in 930009 the requested LEFT had become illegal by application and
  was refused. At least three of the original's lane changes were executed into
  space that had closed during the latency window.
* **The fine arm is not handicapped in physical outcome.** Every one of its 19
  applied actions was STAY, which changes nothing in the simulator. Whether it
  received 19 or 24 decisions, and whether 3 episodes received none, its
  trajectory is identical to the do-nothing trajectory — demonstrated exactly by
  seeds 930004 and 930007, where the original issued 3 and 2 STAY actions and the
  fine model issued 2 and 0, and all four runs produced identical outcomes.
* **But the fine policy was barely exercised.** 19 decisions across 12 episodes,
  3 episodes with none. The run therefore does not measure "what the fine model
  would do with adequate control authority"; it measures "what happens when the
  fine model's only observed action is STAY". Its offline behaviour (0 LEFT/0
  RIGHT in 4,317 rows, BRAKE on 2.8%) makes STAY-only the expected behaviour, but
  the closed-loop evidence for that is 19 samples.

**Verdict on confounding:** the comparison is *not* a controlled comparison of
policy quality. The latency difference changes the original's actions and their
timing, and it removes control from a quarter of the fine episodes. Because the
fine actions are no-ops, the latency does not mechanically flatter the fine
score, but it does mean the two arms were driven under materially different
control authority. A fair comparison needs a fixed decision interval longer than
both latencies (e.g. 2 s wall clock) and identical decision counts per episode.

---

## 7. VERIFIED FACTS / INTERPRETATIONS / UNANSWERED QUESTIONS

### 7.1 Verified facts

1. Offline metrics reproduce exactly from raw labels and probabilities
   (accuracy, macro/weighted F1, per-class P/R/F1, confusion matrices, log loss,
   Brier, ECE, MCE, confidence, predicted/actual counts).
2. The fine model is better offline on every proper scoring rule, on every one of
   the 32 seeds, and the effect survives seed-clustered bootstrap
   (accuracy +0.1503, CI [+0.1080, +0.1962]).
3. The fine model never predicts LEFT or RIGHT in 4,317 rows; its BRAKE recall is
   20.1% with 94.96% precision; its accuracy is only +2.48 pp above constant STAY.
4. Both gameplay arms crash 12/12; the crash indicator has zero variance and
   McNemar is degenerate.
5. Gameplay mean survival is +1.069 s for the fine arm with 95% CI
   [−1.375, +3.708], median difference 0.000, 5/4/3 wins, sign p = 1.00,
   exact sign-flip p = 0.46.
6. The only gameplay differences that are statistically detectable are lane
   changes (−0.75 per episode, p = 0.0039) and latency (+394 ms, p = 0.008).
7. BRAKE usage is exactly zero in both arms (0 actions, 0 braking ticks).
8. Distance is a deterministic function of survival (r = 1.0000), so it is not
   independent evidence.
9. The original's 21 applied actions include 9 LEFT; 1 additional LEFT was
   rejected as illegal at application time; 2 late STAY decisions were rejected
   as `wrong_episode`. The raw rejection count is 3, not 5.
10. The fine arm received zero control in 3 of 12 episodes: the controller was
    still completing an inference started on the previous episode when those
    episodes began, and each 2.4–2.6 s episode ended before the pipeline caught
    up.
11. Both sessions used identical seeds, config, initial traffic and prompt mode;
    the model files match their claimed SHA-256 hashes.
12. The teacher that generated the offline labels survives a mean of 67.2 s at
    density 1.5 (60 episodes of 60 s or 90 s: 3 crashes, 57 timeouts), versus
    6.9 s and 8.0 s for the two students.

### 7.2 Interpretations

1. **The offline gain is real but narrow.** It is a STAY-prior correction plus
   partial BRAKE detection, not lane-change competence. Macro F1 rises only
   because the original scored 0 on BRAKE and near-0 on LEFT; the fine model
   still scores 0 on both LEFT and RIGHT.
2. **The gameplay improvement is not statistically or practically convincing.**
   The mean is carried by two seeds (137% of the total), the median difference is
   zero, the confidence interval spans zero, the design's MDE is 3.6× the observed
   effect, and every episode still ends in a crash within 17 s.
3. **The apparent improvement is the original's self-inflicted lane changes.**
   On the 9 seeds where the original changed lanes, it survived longer on 5 and
   shorter on 4; on the 3 seeds where neither changed lanes, outcomes were
   bit-identical. The comparison is "lane changes vs no lane changes", not
   "better model vs worse model".
4. **The latency confound degrades the original arm's lane changes but is inert
   for the fine arm's outcomes.** It cannot be dismissed, but it is not the
   explanation for the fine arm's survival times — those are the deterministic
   no-action trajectories.
5. **Offline accuracy is a poor predictor of closed-loop competence here.** The
   teacher survives 67 s; the student that agrees with it on 86.6% of labels
   survives 8 s. Classic compounding error: missing 80% of BRAKE labels and
   100% of lane changes is fatal at 28 m/s.
6. **The fine model's high confidence (0.969 mean, 0.955 margin) is
   miscalibrated for the closed-loop setting.** Offline ECE is excellent (0.0395)
   because the test distribution is 84% STAY; the model is confidently wrong
   about the situations that matter.

### 7.3 Unanswered questions

1. Would the fine model ever brake or change lanes given more control
   opportunities? 19 decisions, all STAY, cannot answer this. A run with a fixed
   2 s decision interval would.
2. Is the fine model's gameplay advantage (if any) reproducible on fresh seeds?
   n = 12 with MDE 3.8 s cannot distinguish +1 s from 0.
3. How much of the original's 9 lane changes were latency-induced errors rather
   than policy errors? A counterfactual run of the original at ~1.58 s latency,
   or of the fine model at ~1.17 s latency, would separate the two.
4. What is the fine model's true action distribution on gameplay-like states?
   The offline test split is teacher-generated and 84% STAY; the gameplay state
   distribution is different and only 19 samples were observed.
5. Why do the two model directories ship different tokenizer files, and which one
   was used by each run? The recorded token counts cannot distinguish them.
6. Does either model ever complete an episode? Zero completions in 24 attempts,
   and the teacher completes 57/60 at the same density, so the gap is behavioural,
   not environmental.

---

## 8. Reproduction

All scripts are in `LANESHIFT\final-audit-stats-20261105\scripts\`. They are
read-only with respect to inputs and write only to `…\out\`.

```powershell
cd C:\Users\Dhane\dev\cris@ds\LANESHIFT\final-audit-stats-20261105

# environment actually used
..\.venv-host\Scripts\python.exe -c "import sys, numpy, tokenizers; print(sys.version, numpy.__version__, tokenizers.__version__)"

# offline
..\.venv-host\Scripts\python.exe scripts\explore_offline.py       # schema/alignment checks
..\.venv-host\Scripts\python.exe scripts\offline_metrics.py       # pooled metrics + eval_metrics comparison
..\.venv-host\Scripts\python.exe scripts\offline_seed_stats.py    # 32-seed paired stats + cluster bootstrap
..\.venv-host\Scripts\python.exe scripts\offline_baselines.py     # constant-STAY baselines

# gameplay
..\.venv-host\Scripts\python.exe scripts\gameplay_reconstruct.py  # rebuild both sessions from raw logs
..\.venv-host\Scripts\python.exe scripts\gameplay_stats.py        # paired tests, cadence, crash stats
..\.venv-host\Scripts\python.exe scripts\latency_analysis.py      # latency / staleness / lane-gap analysis
..\.venv-host\Scripts\python.exe scripts\crash_forensics.py       # final-tick state of every episode
..\.venv-host\Scripts\python.exe scripts\consistency_checks.py    # determinism, score identity, hashes
..\.venv-host\Scripts\python.exe scripts\tokenizer_check.py       # tokenizer parity
..\.venv-host\Scripts\python.exe scripts\power_analysis.py        # MDE / required n
..\.venv-host\Scripts\python.exe scripts\make_tables.py           # out\tables.md
..\.venv-host\Scripts\python.exe scripts\build_report.py          # this report
```

Environment: Python 3.14.4 (ARM64), numpy 2.5.4, tokenizers 0.23.3.
`scipy`, `pandas` and `scikit-learn` are **not** installed; the Wilcoxon
signed-rank test (exact subset-sum and tie-corrected normal approximation), the
exact sign test, the exact sign-flip permutation test, the paired t-test (via a
continued-fraction incomplete beta) and McNemar's exact test are implemented
from scratch in `scripts/stats_utils.py`.

Resampling details (all fixed-seed, reproducible):

| Procedure | Resamples | Seed |
|---|---|---|
| Offline per-seed bootstrap CI | 20,000 | 20261105 |
| Offline cluster bootstrap (accuracy/log loss/Brier) | 20,000 | 11 / 12 / 13 |
| Offline cluster bootstrap (macro F1) | 20,000 | 14 / 15 |
| Offline sign-flip permutation | 20,000 | 7 |
| Gameplay paired bootstrap CI | 20,000 | 20261105 |
| Gameplay sign-flip permutation (Monte Carlo) | 100,000 | 42 |
| Gameplay sign-flip permutation (exact, 2^12) | 4,096 | — |
| Power analysis bootstrap | 20,000 | 20261105 |

Intermediate outputs (all machine-readable) are in `…\out\`:
`offline_metrics.json`, `offline_seed_stats.json`, `offline_baselines.json`,
`gameplay_reconstruction.json`, `gameplay_stats.json`, `latency_analysis.json`,
`crash_forensics.json`, `consistency_checks.json`, `power_analysis.json`,
`tokenizer_check.json`, `tables.md`.

---

## 9. Artifact hashes

#### Table J. Artifact hashes (SHA-256)

| Artifact | SHA-256 | Bytes |
|---|---|---|
| test_labels | `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4` | 3,146,096 |
| test_provenance | `5427c5968e5ef4adfd630331f4c55ef607037f2e117c56d4e6d97463af1a851b` | 5,031,022 |
| predictions_original | `00df69012ad2ef1d84b129b809743e5debb981d25705f0d85338f6e7700ba9ca` | 1,446,210 |
| predictions_fine | `51b2f41f03068c4c06e85dae7a1f26800c6d8f7dcbf1b9e17fad510afad274ed` | 1,446,288 |
| eval_metrics | `c185fb391c3826497028b33329fd0133b54832a7b4803e5bf63bed6ebce2dc3b` | 4,185 |
| original_onnx | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` | 7,012,901 |
| original_onnx_data | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` | 1,506,886,400 |
| fine_onnx | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` | 7,011,147 |
| fine_onnx_data | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` | 1,506,886,400 |
| original_session_events.jsonl | `e5b192e1cb3647e5d307fa1fcb32e33b14518f35e9d90720454e47ac71b4ce0d` | 285,478 |
| original_session_episodes.jsonl | `522545db462be6f57412f980dba423ce6070e82f008396439bab9e60caf7ceda` | 3,920 |
| original_session_telemetry.jsonl | `50dbbdc439744e081ec7499e9519c761d27a13470c81ec0934240d9dfc10f34d` | 4,106,622 |
| original_session_manifest.json | `8d6bca420a87d554d8f2b24cce111c2245d955e396e678562c231cf174433e4c` | 311 |
| summary_original-12.csv | `6b818f2e0bae2613dc2a0cc0389cb258618fbda4d68804868b180b938316fe7a` | 1,070 |
| summary_original-12.episodes.jsonl | `cfd2b656dc9344b90aeabd2ffc57035bc1cc0c3b9d1cdaa7d3d06197937af90f` | 4,328 |
| summary_original-12.onnx_kev.metrics.json | `089b4ad7c1ffe9341596f158533ac9b712e8461a4a2da6fa2189bbad7ed57b69` | 1,217 |
| fine_session_events.jsonl | `68e9fc3ae54f01c7857b8fd56efc75fd36b5fb24c20d632a2c4df324419cbe8a` | 242,016 |
| fine_session_episodes.jsonl | `bb703619fdf10da8d2f7609880edad7c0c27ce3d386e5c0247d6de91d414775a` | 3,843 |
| fine_session_telemetry.jsonl | `b2bf40fbbba613db17e3a6f7609949a07201933e5ed0b604c942a0ba373482d5` | 4,416,931 |
| fine_session_manifest.json | `dd46b69403abb433fd454513c513bb87bd39e5d17671df96ed8b6a163dbc2645` | 311 |
| summary_fine-12.csv | `1cf903c363dcd78acc2a1f1ec0fb9d076c07170effe20a63268ec0ebf088c3cd` | 1,062 |
| summary_fine-12.episodes.jsonl | `789244a31aeb52d9d3170c3333b3afce0d1f8dcccb5ab7d47c66b8ac597f00f2` | 4,251 |
| summary_fine-12.onnx_kev.metrics.json | `8f2b50cced5ae88aae8b15f05ba00867d738dbb185890c10831c44bb55f739a4` | 1,189 |
| kev-original/kev07b_fp16.data | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` | 1,506,886,400 |
| kev-original/kev07b_fp16_L256.onnx | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` | 7,012,901 |
| kev-original/tokenizer.json | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` | 12,807,196 |
| kev-finetuned-subset-e1/kev07b_fp16.data | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` | 1,506,886,400 |
| kev-finetuned-subset-e1/kev07b_fp16_L256.onnx | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` | 7,011,147 |
| kev-finetuned-subset-e1/tokenizer.json | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` | 19,989,339 |
| kev-finetuned-subset-e1/tokenizer_config.json | `20b487e1f89ac8b41e43b9d765cff6b7dea5e5ea871accdf1a1aba9357b8f393` | 1,469 |
| kev-finetuned-subset-e1/verify_fine_onnx.json | `1b03776854f83e58303c88620f1ae0823ebd7044cffe13cee76b61d8f1331107` | 1,037 |

Hashes were computed with `Get-FileHash -Algorithm SHA256` and with an
independent streaming SHA-256 in `scripts/consistency_checks.py`; both agree.

### 9.1 Audit code hashes (so drift is detectable)

```text
2400711b9d909ccbf33d4d974706c1325e34e90ad1e9f915a02df44f829af273  build_report.py
7df7285fe7d3c0f754c837d44bd23e920d1dc19e947ccfafb1b4cff4a8dae7c7  consistency_checks.py
44d831135a2bc1e44f46fb82c6438c7883c89274859cc393929bb0683b8b6c3d  crash_forensics.py
d11a6fa5245ed419cb1a3ff750aa2b319a1606f909114523cfdbdabaec49f9d4  explore_cadence.py
b0611a7e5800f8aac09c809c5a2a2551a5c3b5820d6163f3238f497c70318d09  explore_offline.py
da096e5768accbb4cd173708570c5ed6c60d8e7dc0a98c8c2f50c751b78f185e  gameplay_reconstruct.py
cc890a4728352620de2e1c8daf2b810b9d4db35f635128efb84ce60ca46047c0  gameplay_stats.py
dcb7cf3af2f1e1dfb7dc2f33f1344d13996bdc08b75f65f373e71c6c986f436d  latency_analysis.py
619fbde6702d9c798f42b18c3788b15cf91e87e996d60448b834651bc4de6434  make_tables.py
c250fbbc487210b3f4d0459bcbf1b0339d0aa1accc80efeb6524e84a3a8b597e  offline_baselines.py
dc46382969e6e6641f7e73cc93ed6b0d0556348fe676afbee768f95b4f9f172f  offline_metrics.py
d2c0e9de413fd32e6239b3fb82a94bcfa64a628e738787b3af35e165308bef5a  offline_seed_stats.py
dd73e10f6e6e36bbda15c4773c1a7cd7944afce3d91c8e64ad5e71117ac91f61  power_analysis.py
82b45cb74aabf47c13e43793d9d1a51f4483317e3a123eecea87a14274af8bca  stats_utils.py
a9d80d4925d997b71aa1127042b630611fa6e104c52eae8dce8b853267b4f4ef  tokenizer_check.py
```
