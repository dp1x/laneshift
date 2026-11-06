# LANESHIFT — Final Deep Technical Research Audit

**Date:** 2026-11-05  
**Mode:** read-only. No GPU, no training, no dataset/model/code changes, no commits or pushes.  
**Scope:** the full LANESHIFT Kev-0.8B experiment: synthetic dataset, human recordings, subset fine-tune, offline evaluation, ONNX export, and matched closed-loop gameplay.

This report consolidates three independent read-only audits:

- Data/synthetic demonstrations: `LANESHIFT\final-audit-data-20261105\REPORT.md`
- Model/training: `LANESHIFT\final-audit-model-20261105\REPORT.md`
- Statistics/gameplay: `LANESHIFT\final-audit-stats-20261105\REPORT.md`
- Independent main-agent recalculation: `LANESHIFT\final-audit-main-20261105\recalculated_stats.json`

---

## 0. Executive summary

LANESHIFT is an end-to-end applied imitation-learning testbed: a Godot 2D driving simulator, a schema-2 scene observation, a Kev-0.8B decision model, a synthetic safe-teacher data pipeline, LoRA fine-tuning, ONNX export, and offline plus closed-loop evaluation.

The dataset is real and internally clean: **25,427 rows** (16,795 train / 4,315 validation / 4,317 test), **25,368 safe_teacher + 59 human**, all hashes verify, all prompts fit L256, seeds and episodes are disjoint, and the test file is untouched. But only **0.23%** of rows are human, the teacher crashes in **7/192** episodes, and the test set is **unseen seeds on the same six-cell scenario grid**, re-weighted toward density 1.5 — not unseen scenario families.

The first LoRA experiment used a **4,096-row, teacher-only subset**, one epoch, 512 optimizer steps. It learned **STAY** and a high-precision subset of **BRAKE**, but **never predicted LEFT or RIGHT** on 4,317 test rows. Fine-tuned accuracy is **0.8663**, but a constant-STAY predictor already scores **0.8416**; the real gain over that trivial baseline is **+2.48 percentage points** (95% CI +1.98 to +3.13). Macro F1 rises from 0.2143 to 0.3145, almost entirely from BRAKE.

The offline improvement is statistically robust across the 32 test seeds (fine wins 32/32; accuracy delta +0.1503, cluster CI +0.1080 to +0.1962). It does **not** translate into driving: in 12 matched gameplay seeds both models crash **12/12**; survival difference is **+1.07 s** (95% CI −1.38 to +3.71, sign p=1.00, exact sign-flip p=0.46), and the mean is carried by two seeds (137% of the sum; without them the mean is −0.48 s). The fine arm changes lanes **0** times and brakes **0** times; its STAY actions are a physical no-op. There is **no statistically or practically convincing closed-loop improvement**.

**Honest conclusion:** fine-tuning improved teacher-label prediction and calibration on unseen seeds from the same scenario family, but it did not learn lane changes and did not improve driving. The experiment is a well-instrumented **negative result** about data coverage, rare-manoeuvre learning, and the gap between offline accuracy and closed-loop control.

---

## 0.1 Methodology

Three independent read-only subagents were run against the same workspace. The data
audit re-derived all 25,427 rows, hashes, provenance, splits and teacher
outcomes from files and recordings. The model audit inspected the pinned Kev
source, both checkpoints, adapter/head tensors, training metrics, offline
predictions and ONNX artifacts. The statistics audit recomputed every offline
metric from raw labels and probabilities, rebuilt both gameplay sessions from
raw `events.jsonl`/`telemetry.jsonl`, and applied seed-clustered bootstrap,
sign, Wilcoxon and sign-flip tests. The main agent independently recalculated the
offline and paired gameplay statistics (`final-audit-main-20261105\calc_stats.py`)
and reconciled discrepancies. No audit wrote to the dataset, models, code or
recordings.

## 1. What was built

- Godot 4.7.2 LANESHIFT 2D three-lane driving simulator with deterministic seeded traffic, schema-2 observations, telemetry, and recording.
- Kev-0.8B integration: Qwen3.5-0.8B-Base `dc7cdfe2…`, released Kev checkpoint `jaredpalmer/kev-0.8b@v1.0`, pointer head, scene prompt, L256 ONNX serving.
- Synthetic demonstration pipeline: safe_teacher policy, seed/density grid, 60/90 s episodes, 2.0× simulation, every second accepted decision (~0.466 s sim), exact inference tokenizer gate.
- Dataset builder with provenance, seed-disjoint splits, human/teacher separation, legal-action criteria, and L256 enforcement.
- One LoRA fine-tune: rank 16, alpha 32, dropout 0.05, 12 target modules, head_dim 256, fp32 weights, bf16 compute, 4,096 rows, one epoch.
- Offline evaluator on the untouched 4,317-row test split.
- Fine-tuned L256 ONNX export with a new 1.507 GB fp16 blob, plus original-vs-fine matched gameplay.

## 2. Original research hypothesis

Fine-tuning the pretrained Kev-0.8B decision model with LoRA on synthetic safe-teacher demonstrations plus a small human block would improve action prediction and closed-loop driving on unseen seeds, moving toward a usable Kev-0.8B driver. A secondary hypothesis was that genuine human demonstrations would add value beyond the synthetic teacher.

## 3. How the synthetic dataset was generated

The `safe_teacher` policy drove the simulator over a seed plan spanning six scenario cells:

| Cell | Density | Duration | Train seeds | Val | Test |
|---|---:|---:|---:|---:|---:|
| A | 0.7 | 60 s | 8 | 2 | 2 |
| B | 1.0 | 60 s | 20 | 4 | 4 |
| C | 1.5 | 60 s | 22 | 10 | 10 |
| D | 1.5 | 90 s | 10 | 4 | 4 |
| E | 2.5 | 60 s | 38 | 6 | 6 |
| F | 3.5 | 60 s | 30 | 6 | 6 |
| H human | 0.7 | — | 3 | 0 | 0 |

The builder rendered the exact scene prompt used by `OnnxKevPolicy._encode`, restricted criteria to legal actions, enforced ≤256 tokens, and excluded 20 over-budget teacher rows plus one unsafe human lane change. Result: **25,427 rows = 16,795 train / 4,315 validation / 4,317 test**, **195 seeds/episodes**, **25,368 safe_teacher + 59 human**. Action totals: STAY 21,179, BRAKE 3,484, LEFT 378, RIGHT 386.

## 4. Was the dataset diverse and representative?

**Partly.** It covers all six cells and all five densities, but the test is not a held-out scenario family:

- Test = **unseen seeds** 920001–920032 on the **same six-cell grid** and same teacher/prompt/action space.
- The test mixture is materially re-weighted: density 1.5 is **47.6%** of test rows vs **27.4%** of train; density 2.5 is **16.6%** vs **28.3%**. Cell/density total-variation distance = **0.2023**.
- Per-cell action distributions are close (TV 0.007–0.050), so the conditional policy is familiar; the scenario mix is not identical.
- The test has **122 turn rows (2.83%)** vs **105 in the subset (2.56%)**.

Human data is tiny: **59 rows (0.23%)**, one session, three episodes, 29 LEFT + 30 RIGHT, zero STAY/BRAKE. The 12.5-hour session span contains **44,668 s of idle time** and only **395.7 s of simulated play**. One unsafe row was excluded.

Teacher quality is imperfect: **7 crashes in 192 episodes (3.65%)** — seeds 900047, 900086, 900107, 900113, 910008, 920017, 920021. **518 pre-crash teacher rows** ship with `label_quality: ok`; no post-crash labels, and a 1,545-row re-derivation from recordings matched 100%.

## 5. Full train split vs the actual fine-tune subset

| Metric | Full train | 4,096 subset | Subset/train |
|---|---:|---:|---:|
| Rows | 16,795 | 4,096 | 24.4% |
| Seeds/episodes | 131 | 124 | 94.7% |
| STAY | 13,921 (82.89%) | 3,439 (83.96%) | 24.7% |
| BRAKE | 2,350 (13.99%) | 552 (13.48%) | 23.5% |
| LEFT | 262 (1.56%) | 55 (1.34%) | 21.0% |
| RIGHT | 262 (1.56%) | 50 (1.22%) | 19.1% |
| Teacher turns | 465 | 105 | 22.6% |
| Seeds with a teacher turn | 125 | 73 | 58.4% |
| Max teacher turns/seed | 12 | 4 | — |
| Seeds losing every turn | — | 52 of 125 | — |

The aggregate turn rate barely changed (2.78% → 2.56% teacher rows), but the absolute turn count and turn-bearing contexts collapsed. **All 105 retained turns are isolated frames**: 0/105 have another turn within four decisions in the same seed; 90/105 are preceded by STAY; 51 of 73 turn-bearing seeds have exactly one turn. This is **context fragmentation**, not merely class imbalance.

There is also a subset-selection defect: the top-up fallback over-weighted a few seeds (122 seeds have 32 rows, seed 900001 has **128**, seed 900002 has **64**). One train row of the single train↔test duplicate prompt is inside the subset (both STAY, 181 tokens).

## 6. What fine-tuning learned

- **STAY** was learned strongly: recall 0.9983, F1 0.9263, 4,198 predicted.
- **BRAKE** was partially learned: 119 predictions, precision 0.9496, recall 0.2011, F1 0.3319; 113 correct high-confidence BRAKE.
- **LEFT/RIGHT** were not learned: 0 predictions of each; 122/122 turn rows top-1 STAY; true turn rank 3–4; mean P(LEFT)=0.0124, P(RIGHT)=0.0142, essentially the subset priors (0.0134/0.0122); log-odds turn-vs-STAY −4.34/−4.18.
- The original model had a LEFT bias (625 LEFT predictions, mostly false) and zero BRAKE/RIGHT. The fine-tune corrected LEFT into STAY and learned BRAKE, but never raised either turn above STAY.

## 7. Why lane-change behaviour collapsed

**Demonstrated:** labels, legal masks, decoding, loss, checkpoint loading and ONNX export are correct. All 55 LEFT and 67 RIGHT test rows have the turn in `criteria`; the fine model assigns nonzero probability but ranks it below STAY. Temperature cannot change argmax. The head and adapter were the real fine weights.

**Leading explanation (not proven causally):**

1. **Rare, fragmented coverage:** 105 turn rows, 73 seeds, 51 singletons, max 4/seed, 0/105 with a neighbouring turn, 52/125 turn-bearing seeds dropped. The model never saw a manoeuvre as a sequence.
2. **Short, heavily clipped update:** one epoch / 512 steps, LR 2e-5, **506/512 steps clipped at norm 1.0**, mean pre-clip grad norm 9.10. Head moved only ~1.2% and adapter ~2.4% of their Frobenius norms. BRAKE had 5.26× more rows and was learned; turns were not.
3. **Original head prior:** the released head already over-predicted LEFT and never predicted RIGHT/BRAKE. Fine-tuning at low LR did not re-rank turns.
4. **Calibration is not the cause:** ECE improved to 0.0395; the fine head temperature is 1.0 vs original 2.3511, but argmax is unchanged by temperature.

## 8. Why offline accuracy improved without driving improvement

Test accuracy is dominated by STAY (3,633/4,317 = **84.16%** constant-STAY baseline). The fine model scores **86.63%**, only **+2.48 pp** over always-STAY (95% CI +1.98 to +3.13). Macro F1 rises 0.2143 → 0.3145, but LEFT/RIGHT F1 remain zero and BRAKE recall is only 20.1%. The accuracy gain is a better STAY/BRAKE boundary plus calibration, not a driving policy.

## 9. Offline statistics (4,317 rows, 32 test seeds)

| Metric | Original | Fine | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7160 | 0.8663 | +0.1503 |
| Macro F1 | 0.2143 | 0.3145 | +0.1002 |
| Weighted F1 | 0.7093 | 0.8228 | +0.1135 |
| Log loss | 0.9268 | 0.4093 | −0.5175 |
| Brier | 0.5417 | 0.2090 | −0.3327 |
| ECE | 0.2876 | 0.0395 | −0.2481 |
| Top-2 accuracy | 0.9166 | 0.9717 | +0.0551 |

Seed-cluster bootstrap: accuracy delta **+0.1503 [0.1080, 0.1962]**; macro-F1 delta **+0.1002 [0.0789, 0.1232]**. Fine wins all 32 seeds on accuracy and macro F1 (sign p=4.7e-10). Row-level standard errors understate precision by 2.6× (ICC 0.055/0.022; design effect 8.4/3.9).

Per-class test performance:

| Action | Support | Original P / R / F1 | Fine P / R / F1 |
|---|---:|---|---|
| STAY | 3,633 | 0.836 / 0.849 / 0.843 | 0.864 / 0.998 / 0.926 |
| BRAKE | 562 | 0.000 / 0.000 / 0.000 | 0.950 / 0.201 / 0.332 |
| LEFT | 55 | 0.008 / 0.091 / 0.015 | 0.000 / 0.000 / 0.000 |
| RIGHT | 67 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

LEFT was legal in 3,046 test rows and RIGHT in 2,966; the fine model predicted
neither in any row. Constant-STAY baseline accuracy is 0.8416; constant-STAY
macro F1 is 0.2285.

Confusion matrices (LEFT, STAY, RIGHT, BRAKE):

```text
original: [[5,50,0,0],[547,3086,0,0],[8,59,0,0],[65,497,0,0]]
fine:     [[0,55,0,0],[0,3627,0,6],[0,67,0,0],[0,449,0,113]]
```

## 10. Matched gameplay (12 fresh seeds, 60 s, density 1.5)

Both arms used identical seeds 930001–930012, initial traffic, duration, density and 2.0× simulation; scene prompt, L256, CPU ONNX. Raw sessions: `session_20261105_201308_176` (original), `session_20261105_201408_325` (fine).

| Metric | Original | Fine |
|---|---:|---:|
| Crashes | 12/12 | 12/12 |
| Completions | 0 | 0 |
| Mean survival | 6.886 s | 7.956 s |
| Median survival | 6.183 s | 5.350 s |
| Mean distance | 182.3 | 212.2 |
| Mean score | 115.2 | 127.7 |
| Mean overtaken_count | 2.50 | 2.67 |
| LEFT/RIGHT/STAY/BRAKE | 9/0/12/0 | 0/0/19/0 |
| BRAKE frequency | 0% | 0% |
| Lane changes | 9 | 0 |
| Applied decisions | 21 | 19 |
| Mean latency | 1.170 s | 1.581 s |

Paired tests, fine − original:

| Metric | Mean diff | 95% CI | Median | W/L/T | Sign p | Wilcoxon | Sign-flip |
|---|---:|---|---:|---|---:|---:|---:|
| Survival | +1.069 s | [−1.375, +3.708] | 0 | 5/4/3 | 1.000 | 0.570 | 0.461 |
| Distance | +29.94 | [−38.50, +103.84] | 0 | 5/4/3 | 1.000 | 0.570 | 0.461 |
| Score | +12.51 | [−48.41, +72.83] | 0 | 5/4/3 | 1.000 | 0.734 | 0.734 |
| Overtaken | +0.167 | [−1.50, +1.75] | 0 | 3/2/7 | 1.000 | 1.000 | 0.938 |
| Lane changes | −0.750 | [−1.00, −0.50] | −1 | 0/9/3 | 0.0039 | 0.0034 | 0.0039 |
| Latency (ms) | +393.7 | [+240.2, +516.9] | +478 | 8/1/0 | 0.039 | 0.0078 | 0.0078 |

- All 12 crash; crash rate has zero variance; McNemar is degenerate.
- Distance is a deterministic function of survival (r=1.0000); score is dominated by survival (r=0.959). Only overtakes are partly independent, and their difference is not significant.
- The top two seeds account for **137%** of the survival sum; without them the mean is **−0.48 s**; without the top three, −1.23 s.
- MDE at n=12 is **3.82 s**; observed +1.07 s is 28% of MDE. About **154 seeds** would be needed for 80% power at the observed effect.
- The fine arm had **3 episodes with zero applied control**; STAY is a physical no-op. On the three seeds where the original also only issued STAY (930004, 930006, 930007), outcomes are bit-identical despite different decision counts.
- Latency confound: original 1.170 s vs fine 1.581 s; 24 vs 19 delivered, 21 vs 19 applied, 35 vs 30 inference runs, 11/11 stale drops. The original's LEFT actions were executed on 2.4 s-stale observations; the fine arm was barely exercised (19 decisions).

**Correction:** raw gameplay rejections are **3**, not 5; only **1 of 10** LEFT decisions was rejected. The summary CSV double-counted `protocol_rejections` and `rejected_actions`. The two 15 s smoke sessions are seed-mismatched (930001 vs 930002) and are not a model comparison.

## 11. Answers to the 14 questions

1. **What did we build?** A reproducible 2D driving imitation-learning stack: Godot simulator, Kev-0.8B scene-policy integration, safe-teacher synthetic data pipeline, 25k dataset, LoRA fine-tune, L256 ONNX export, offline and closed-loop evaluation.
2. **Original hypothesis?** Fine-tuning Kev on synthetic teacher + human demos improves unseen-seed action prediction and closed-loop driving.
3. **How was synthetic data generated?** Safe-teacher rollouts over a six-cell density/duration seed grid, 2.0× simulation, every second accepted decision, exact scene prompt, legal-action criteria, L256 gate, seed-disjoint builder.
4. **Was it diverse/representative?** All six cells/densities are present, but the test is unseen seeds on the same grid, re-weighted (TV 0.2023; density 1.5 47.6% vs 27.4%). It is not a held-out scenario family.
5. **What did fine-tuning learn?** STAY and a high-precision BRAKE subset; calibration; no LEFT/RIGHT.
6. **Why did lane changes collapse?** Rare fragmented turn coverage (105 rows, 73 seeds, 51 singletons, max 4/seed, 52/125 turn-bearing seeds dropped) plus one epoch, 506/512 clipped steps, low LR, small head/adapter movement, and an original LEFT-biased head. Inference/labels/masks are clean.
7. **Why accuracy up without driving up?** STAY is 84.16% of test; fine accuracy 86.63% is +2.48 pp over constant STAY. Macro F1 improves only through BRAKE; LEFT/RIGHT remain zero.
8. **Did it improve unseen-seed generalisation?** Offline yes: fine wins all 32 unseen test seeds; accuracy +0.1503 [0.1080,0.1962], macro F1 +0.1002 [0.0789,0.1232]. Closed-loop no.
9. **Did it improve closed-loop driving?** No convincing evidence: both crash 12/12; survival +1.07 s CI [−1.38,+3.71], p≈1; median 0; two seeds carry the mean; fine never changes lane/brakes.
10. **Strongest findings?** Dataset integrity; zero-turn collapse; +2.48 pp over constant STAY; test is same-family/reweighted; closed-loop null; latency confound; subset fragmentation.
11. **Limitations/confounders?** One training run/seed; one epoch (not the planned three); 4,096-row subset; seed imbalance; one train↔test duplicate prompt; 7 teacher crashes; human data 0.23%; no scenario-family holdout; n=12 all-crash gameplay; CPU latency 1.17 vs 1.58 s; different tokenizer files (vocab 248,066 vs 248,077; state tokens matched 41/41 logged states); ONNX parity 0.053 logits; head temperature reset; no repeated training seeds.
12. **Defensible claims?** Dataset counts/hashes/labels; zero-turn behavior; offline metrics and 32/32 seed improvement; fine beats original offline; fine beats constant STAY by a small margin; ONNX fine graph differs; both gameplay arms crash 12/12; latency difference; test is unseen seeds, not unseen scenarios.
13. **Not defensible?** That fine-tuning improves driving; that it generalizes to novel scenario families; that the cause of zero turns is proven; that human data helped; that the model is safe or deployable; that the marginal gameplay mean is a real improvement.
14. **Three future experiments?** (a) Turn/brake-enriched, context-clustered dataset (retain all 465 teacher turns and contiguous turn sequences) plus 2–3 epochs, fixed decision cadence and held-out scenario families; (b) DAgger/on-policy correction with safe teacher/human interventions around lane changes and braking; (c) latency-matched closed-loop benchmark with ≥154 seeds (more for score/overtakes) and ablations of head LR, class weighting, LoRA targets, and human-vs-synthetic mixture.

## 12. Verified facts vs interpretations vs unanswered

### VERIFIED FACTS
- 25,427 rows; 16,795/4,315/4,317; 25,368 teacher + 59 human; 195 episodes/seeds; all hashes verify; test hash `d16f65ab…` unchanged.
- 7 teacher crashes (3.65%); 518 pre-crash rows; 20 over-budget teacher +1 unsafe human excluded.
- Subset 4,096; 124 seeds; 105 teacher turns (55 L/50 R); 73 turn-bearing seeds; 51 singletons; 0/105 with a neighbouring turn; 52/125 turn-bearing seeds lost.
- Fine model: 0 LEFT/0 RIGHT/119 BRAKE/4,198 STAY; P(turn) 1.24%/1.42%; log-odds −4.34/−4.18; accuracy 0.8663; macro F1 0.3145; ECE 0.0395.
- Fine wins all 32 test seeds offline; cluster CIs above zero.
- Both gameplay arms crash 12/12; survival +1.07 s CI [−1.38,+3.71]; median 0; 5/4/3; sign p=1; fine 0 lane changes/0 brakes; original 9 lane changes; raw rejections 3, not 5.
- Fine ONNX graph/blob differ from original; max logit difference 0.2083; fine ONNX vs fine torch parity 0.0531.

### INTERPRETATIONS
- The zero-turn failure is primarily rare/fragmented turn coverage plus a short, clipped, low-LR update; BRAKE's 5.26× more rows were learned.
- Offline accuracy is inflated by the STAY majority; it is not a driving metric.
- The test measures unseen seeds under a familiar, re-weighted scenario family; it does not test novel scenario types.
- The gameplay mean is not a robust improvement; it is driven by two seeds and confounded by latency/control opportunity.

### UNANSWERED QUESTIONS
- Would 2–3 epochs on the same subset learn turns?
- Would a turn-clustered subset or full 16,795-row train set learn turns?
- Would a higher head LR or class-weighted loss fix the collapse without changing the data?
- Would a held-out scenario family (new density/cell) expose larger generalization failures?
- Does the 59-row human block help or hurt at this scale?
- Would a latency-matched GPU run change the gameplay ranking?

### EXTRAPOLATIONS / FUTURE APPLICATIONS
- The result is consistent with behavior-cloning long-tail failure: nominal
  coverage and high offline accuracy can coexist with zero rare-manoeuvre
  competence and no closed-loop gain.
- The same pattern is likely in robotics manipulation/navigation when rare
  recovery or lane-change-like manoeuvres are sampled as isolated frames rather
  than contiguous sequences.
- The methodology (exact-token dataset gate, seed-disjoint splits, provenance,
  offline + closed-loop + latency audit) generalises to other decision-model
  fine-tunes; the specific numbers do not.
- Any claim that synthetic teacher data alone can teach rare safety-critical
  manoeuvres at this scale is unsupported by this experiment.

## 13. Relation to imitation learning, synthetic demonstrations, decision models, robotics

This is a textbook behavior-cloning failure mode: the policy learns the marginal action distribution (STAY) and a high-precision subset of BRAKE, but not the rare state-conditioned manoeuvre (lane change). The dataset has broad nominal coverage but almost no repeated turn contexts; the model sees 105 isolated turn frames once. Offline accuracy on teacher labels is high because the teacher and test are STAY-dominated; closed-loop performance collapses because the policy never turns or brakes and the simulator is deterministic and safety-critical. The result argues for:

- long-tail data collection and sequence-level manoeuvre coverage, not just row counts;
- on-policy correction/DAgger rather than pure offline cloning;
- closed-loop evaluation with enough seeds and matched inference latency;
- separating "predict the teacher" from "drive safely".

## 14. Reproducibility

Read-only audit scripts and outputs:

- `LANESHIFT\final-audit-main-20261105\calc_stats.py` and `recalculated_stats.json`
- `LANESHIFT\final-audit-data-20261105\` (scripts + report)
- `LANESHIFT\final-audit-stats-20261105\scripts\` and `out\`
- `LANESHIFT\final-audit-model-20261105\REPORT.md`

Re-run the independent statistics:

```powershell
C:\Users\Dhane\dev\cris@ds\LANESHIFT\.venv-host\Scripts\python.exe `
  C:\Users\Dhane\dev\cris@ds\LANESHIFT\final-audit-main-20261105\calc_stats.py
```

No training is required or permitted for this audit. The original training command and configs are in `kev-deliverables-20261105\run\training_config.json` and `training_metrics.json`.

## 15. Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `kev_train.jsonl` | `ca2d34f365e074803b99421f83579017a8b7970e83be1f7c2aa74e3382b63ff3` |
| `kev_test.jsonl` | `d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4` |
| `manifest.json` | `fef79c5cee92c1a5a498655d987567a9eb5b3cb834892496c70293ae1d408110` |
| Fine adapter | `142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8` |
| Fine head | `7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38` |
| Fine ONNX | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` |
| Fine ONNX data | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` |
| Original ONNX | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| Original ONNX data | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |
| Offline eval metrics | `c185fb391c3826497028b33329fd0133b54832a7b4803e5bf63bed6ebce2dc3b` |
| Original predictions | `00df69012ad2ef1d84b129b809743e5debb981d25705f0d85338f6e7700ba9ca` |
| Fine predictions | `51b2f41f03068c4c06e85dae7a1f26800c6d8f7dcbf1b9e17fad510afad274ed` |

Gameplay raw sessions: `session_20261105_201308_176` (original), `session_20261105_201408_325` (fine); both preserved under `LANESHIFT\cris@ds\.portable-dev\recordings`.

## 16. Honest conclusion

LANESHIFT demonstrates a complete, reproducible applied-ML loop and a clean negative result. The dataset and experiment are real; the fine-tune is real; the offline gain is real but narrow and dominated by STAY/BRAKE. The model did **not** learn lane changes, did **not** improve closed-loop driving, and did **not** demonstrate generalisation to unseen scenario families. The strongest scientific value is the measured gap between teacher-label accuracy and closed-loop competence, and the evidence that rare, fragmented manoeuvre coverage plus a short clipped update can produce a policy that looks better offline while behaving worse on the road.

Do not deploy this checkpoint. The next experiment should target turn/brake coverage and closed-loop evaluation, not another accuracy run on the same distribution.
