# LANESHIFT — reproducible research release

This branch contains the complete LANESHIFT Kev-0.8B research artifact:
the Godot 2D simulator, Kev scene-policy integration, synthetic teacher pipeline,
the 25,427-row `kev-v2-scene` dataset, the exact 4,096-row training subset,
offline predictions/metrics, matched gameplay results, and three independent
read-only audits.

## Actual experiment

The fine-tune used **4,096 rows from the 16,795-row training split**, **one
epoch**, **512 optimizer steps**, batch 1, accumulation 8, LR 2e-5, LoRA rank 16,
head dimension 256, fp32 LoRA weights, bf16 compute. This is the run that
produced the checkpoint; the handoff's default of three epochs was not executed.

The dataset is **25,427 rows**: 16,795 train / 4,315 validation / 4,317 test;
25,368 `safe_teacher` + 59 human. The test split is untouched.

## Result

- Offline accuracy: original **0.7160**, fine-tuned **0.8663**
- Macro F1: **0.2143 → 0.3145**
- Constant-STAY accuracy: **0.8416**; the fine model is only **+2.48 pp** above it
- Fine model: **0 LEFT, 0 RIGHT, 119 BRAKE, 4,198 STAY** on 4,317 test rows
- Matched gameplay, 12 fresh seeds: **both models crash 12/12**
- Survival difference: **+1.07 s**, 95% CI **[−1.38, +3.71]**, median 0, sign p=1.00
- Fine arm: **0 lane changes, 0 brakes**; original: 9 lane changes
- No statistically or practically convincing closed-loop improvement

This is a negative result about rare-manoeuvre coverage, short clipped updates,
and the gap between teacher-label accuracy and closed-loop driving. Do not deploy
the checkpoint as a driving policy.

## Layout

- `datasets/kev-v2-scene/` — full 25,427-row dataset, provenance, cards, seed plan, approvals
- `datasets/kev-subset-4096/` — exact 4,096-row fine-tune subset and provenance
- `research/offline/` — untouched-test predictions and metrics
- `research/gameplay/` — 12-seed original-vs-fine CSV/JSONL/metrics
- `research/training/` — actual training config, metrics, ONNX parity verification
- `research/audits/` — final synthesis and all three independent audits
- `research/scripts/` — statistics and turn-analysis scripts
- `research/model-artifacts/EXTERNAL-ARTIFACTS.md` — release-only model blobs and hashes
- `notebooks/kev_finetune.ipynb`, `notebooks/KEV-V2-SCENE.md` — fine-tuning and reproduction runbook
- `python/build_kev_v2_scene.py`, `python/validate_kev_v2_scene.py` — dataset builder/validator

## Model weights

The fine adapter, pointer head, ONNX graph and 1.507 GB external fp16 blob are
**not** ordinary Git files. Their hashes and release instructions are in
`research/model-artifacts/EXTERNAL-ARTIFACTS.md`.

The older 1,348-row rule-based pilot remains in the separate training repository
and is not duplicated in this Git release.

## Read the audits

Start with `research/audits/FINAL-REPORT.md`. The independent data, model and
statistics audits are alongside it.
