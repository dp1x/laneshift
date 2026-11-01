# Next human gate

The engineering can continue automatically through the decision dataset and the
small supervised baseline. The Kev fine-tune is the next decision that needs a
human.

## Where the project is

- The simulation testbed, telemetry, controller, local Kev inference, packaged
  watch mode and matched pilot harness work.
- The current gate is a real decision dataset and a baseline model.
- The decision dataset and baseline pilot is merged in
  [`TRAINING-PILOT.md`](TRAINING-PILOT.md). It used rule-based labels: 1,348
  decision rows over 15 seeds; the held-out MLP reached 0.8844 accuracy,
  0.4426 macro-F1, 0.3587 log loss, 0.1763 Brier and 0.0671 ECE. The balanced
  variant reached 0.8728 accuracy and 0.5574 macro-F1. These are imitation
  metrics, not driving results.
- The released Kev checkpoint is slow and uncertain. On `C:` it is about
  1.4-2.0 s per decision; pure Kev still chooses `STAY` at roughly 38-41%.
  The watch-mode safety layer is what keeps the car alive.

## Decisions that need a human

1. **Data source.** The original thesis needs real human demonstrations. The
   pilot uses rule-based labels, which are useful for plumbing but cannot prove
   that fine-tuning improves human-like driving. Recommended: 60-120 minutes of
   human play across 3-5 drivers and 30+ sessions, with approval.
2. **What counts as an approved decision.** Define safety, legality and
   quality criteria. Exclude `controller_fallback`, `safety_rule`, rejected and
   stale decisions from a pure Kev training set.
3. **Fine-tune target.** A small MLP/GRU on structured state is cheap and
   reproducible. Kev LoRA needs a GPU and a training budget. Do the baseline
   first; do not fine-tune Kev until the baseline and data audit pass.
4. **Compute and time budget.** Kev-0.8B fine-tuning on this CPU is not
   practical. Colab or another GPU is required. Decide the budget and the
   maximum training time.
5. **Evaluation and claim.** The minimum evidence is 20+ held-out seeds at
   60 seconds, original vs fine-tuned vs rule-based, with score, survival,
   collisions, action distribution, Brier score, log loss and ECE. No claim of
   improvement before that.
6. **Project scope.** The simulation testbed is a branch of the original
   phone/ADB/CV idea. Decide whether the next major investment is the phone
   capture/perception pipeline or the simulation fine-tune. The current repo
   can only support the latter without new capture work.

## Automatic work that can continue first

- Export one row per decision, not one row per 30 Hz snapshot.
- Split by seed/session, never adjacent frames.
- Train and calibrate a small MLP/GRU on rule-based data.
- Produce a data card and a failure report.
- Verify the pilot with no outcome leakage and no fine-tuning claims.

The fine-tune, the human data collection and the final research claim remain
human-gated.
