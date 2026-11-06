# Why the first LoRA result predicts zero turns

Analysis of the saved checkpoint and the complete test predictions. No GPU, no
retraining, no dataset changes.

## 1. What was actually loaded

- Fine checkpoint adapter SHA-256 `142e198a…`; head SHA-256 `7ed5d338…`
- `Checkpoint(...).meta.weights_dtype == "fp32"`
- `LocalPredictor(RUN, "cuda", LoadOptions(dtype=bf16, merge=True, temperature=1.0))`
- The fine predictions differ from the original; the fine `head.pt` tensors also
  differ from the original. The adapter and head were loaded.
- `head_lr: 0.0` in `training_config.json` does **not** freeze the head. Kev's
  CLI says `0 = same as --lr`; the head optimizer group uses
  `a.head_lr or a.lr` (`kev/train.py:578`). All four head tensors changed.
- Head temperature changed from the original's fitted **2.3511** to **1.0** on
  save. Evaluation explicitly used 1.0 for both models, so the comparison is
  fair at T=1.0; this is a metadata/calibration regression, not the cause of
  zero turns (temperature does not change argmax).

## 2. Action labels, legal masks, loss and decoding

- Test labels are correct: actual counts LEFT 55, RIGHT 67, BRAKE 562, STAY 3,633.
- Every LEFT/RIGHT test row has the action in its `criteria` legal set. The fine
  model assigns nonzero probability to it; no mask or label-index bug.
- Kev's loss is ordinary cross-entropy on the option index
  (`kev/train.py:50-51`); permutation adjustment updates the label index
  (`kev/train.py:33-35`). No label corruption.
- `LocalPredictor` returns per-option probabilities; the evaluation maps action
  names and takes argmax. STAY/BRAKE are classified correctly enough (STAY
  recall 0.998, BRAKE precision 0.950) to rule out a global decoding/action-map
  error.
- Inference is not the cause.

## 3. What the model actually predicts

Fine-tuned predictions: STAY 4,198, BRAKE 119, LEFT 0, RIGHT 0.

On the 55 LEFT rows:

- Fine top-1 STAY on **55/55**; LEFT rank 3 or 4 on **55/55**
- Mean P(LEFT) **0.0124**; mean P(STAY) **0.9002**
- Mean P(LEFT) is essentially the subset's LEFT prior (0.0134)
- Mean log-odds LEFT vs STAY: **-4.34** (original: -0.12)

On the 67 RIGHT rows:

- Fine top-1 STAY on **67/67**; RIGHT rank 3 or 4 on **67/67**
- Mean P(RIGHT) **0.0142**; mean P(STAY) **0.8816**
- Again close to the subset's RIGHT prior (0.0122)
- Mean log-odds RIGHT vs STAY: **-4.18** (original: -1.45)

The model learned the marginal action prior, not the conditional turn rule.
BRAKE is the exception: 552 training rows (5.3× the 105 turn rows), and it was
learned (F1 0.3319, precision 0.9496, recall 0.2011).

Confusion matrix, fine-tuned (LEFT, STAY, RIGHT, BRAKE):

```text
[[  0,  55,   0,   0],
 [  0,3627,   0,   6],
 [  0,  67,   0,   0],
 [  0, 449,   0, 113]]
```

Original confusion matrix:

```text
[[  5,  50,   0,   0],
 [547,3086,   0,   0],
 [  8,  59,   0,   0],
 [ 65, 497,   0,   0]]
```

The original has a LEFT bias (625 LEFT predictions, mostly false positives) and
never predicts BRAKE or RIGHT. The fine-tune corrected the LEFT bias into STAY
and learned BRAKE, but did not learn either turn.

## 4. Subset vs full training distribution

| Metric | Full train | 4,096 subset |
|---|---:|---:|
| LEFT | 262 (1.56%) | 55 (1.34%) |
| RIGHT | 262 (1.56%) | 50 (1.22%) |
| BRAKE | 2,350 (13.99%) | 552 (13.48%) |
| STAY | 13,921 (82.89%) | 3,439 (83.96%) |
| Teacher turn rows | 465 | 105 (22.6%) |
| Rows | 16,795 | 4,096 (24.4%) |
| Teacher seeds with any turn | 125 | **73** |
| Max teacher turn rows in one seed | **12** | **4** |

Aggregate imbalance alone is not the explanation: the subset's turn share is
only slightly below the full share. The real data defect is **turn-context
fragmentation**. The even 32-rows-per-seed sampler spread 105 turns across 73
seeds, at most 4 per seed. The full teacher set has turn-dense episodes (up to
12 teacher turn rows in one seed; the all-source maximum of 54 is the excluded
human weaving episode). The model saw each turn context roughly once, in a
single epoch.

## 5. Optimization evidence

- 512 optimizer steps on 4,096 records, one epoch
- 506/512 steps hit the global gradient-norm clip (MAX_GRAD_NORM = 1.0)
- Mean pre-clip gradient norm: **9.10**; max 34.22
- Head weights moved only ~**1%** of their scale (max abs diff 0.00137 on
  q/k weights, std ~0.018)
- All 372 adapter tensors changed, but max abs diff 0.00259; the effective
  update was small
- LR 2e-5, head LR also 2e-5 (`0` means same as LR)

This is a second, independent cause: a low-LR, heavily clipped, one-epoch
update cannot re-rank a rare action class against an 84% STAY prior. The model
found the loss minimum that predicts the prior and a small high-precision BRAKE
subset.

## 6. Representative missed turns

`ep_920013_7:217` — teacher LEFT, player lane R, target C, C+14 car:

```text
original: STAY 0.388, LEFT 0.463, BRAKE 0.149 -> LEFT
fine:     STAY 0.731, LEFT 0.026, BRAKE 0.243 -> STAY
```

`ep_920018_2:307` — teacher RIGHT, player lane L, target C, C+42 clear:

```text
original: STAY 0.526, RIGHT 0.230, BRAKE 0.244 -> STAY
fine:     STAY 0.951, RIGHT 0.015, BRAKE 0.034 -> STAY
```

`ep_920017_1:369` — teacher LEFT, dense traffic:

```text
original: STAY 0.375, LEFT 0.420, BRAKE 0.204 -> LEFT
fine:     STAY 0.567, LEFT 0.022, BRAKE 0.411 -> STAY
```

Kev should have selected LEFT/RIGHT: the action is the recorded teacher label
and is in the legal set. The fine model instead assigns 88–95% to STAY and
roughly 1–3% to the correct turn.

## 7. Cause ranking

1. **Training optimization / signal strength (primary).** One epoch, 512 steps,
   LR 2e-5, 506/512 clipped steps, tiny head/adapter movement. The turn class
   contributes ~2.6% of the loss; STAY dominates. The model learned the prior.
2. **Turn-context fragmentation (data).** 105 turns across 73 seeds, max 4 per
   seed, versus 465 teacher turns and up to 12 teacher turns per seed in the
   full set. The aggregate percentage is not the problem; the lack of repeated
   turn contexts is.
3. **Action-prior mismatch.** The original head has a strong LEFT bias and zero
   RIGHT/BRAKE. The fine-tune suppressed LEFT into STAY and learned BRAKE, but
   never moved either turn above STAY.
4. **Calibration/metadata (minor).** The fine checkpoint reset head temperature
   from 2.3511 to 1.0. The test comparison used T=1.0 for both. ECE is good
   (0.0395); temperature cannot change the argmax and is not the cause.
5. **Inference (not a cause).** Legal masks, labels, loss indexing and decoding
   are correct; the fine adapter and head are loaded.

## 8. Smallest sensible next experiment

Do **not** deploy this checkpoint as a driving policy.

Smallest single-change experiment: rerun the same 4,096-row subset for **2
epochs** with `--head_lr 2e-4` (10× the current head LR), keeping the same
audited data, LoRA and checkpoint. This tests whether the head simply failed to
adapt. Expected cost at measured speed: ~82 minutes.

If turns remain at zero, the next minimal data change is a **turn-clustered
4,096-row subset**: keep all 465 teacher turns from the turn-dense episodes,
add ~1,200 BRAKE and fill with STAY. That preserves teacher-only labels and
avoids human weaving while giving repeated turn contexts, rather than 1–4
isolated turns per seed.

The current checkpoint and the untouched test set are preserved.
