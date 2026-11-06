# Model / training audit — Kev-0.8B subset fine-tune

**Date:** 2026-11-05  
**Scope:** read-only audit of the original checkpoint, the subset fine-tune, the offline test predictions, and both ONNX exports. No training, installation, modification, or deletion was performed. The only file written is this report.

**Inputs**

- Kev source: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-upstream` at `2074cc44256200bb37540103fa4572901faa10f3`
- Original checkpoint: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\models\kev-0.8b`
- Fine checkpoint: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-deliverables-20261105\run`
- Offline results: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-deliverables-20261105\results`
- Fine ONNX: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\models\kev-finetuned-subset-e1`
- Original ONNX: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\models\kev-original`
- Prior turn analysis: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-analysis-20261105\ANALYSIS.md`
- Data audit: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\final-audit-data-20261105\REPORT.md`

**Verdict.** The fine checkpoint is genuine and was loaded by both the offline evaluation and the ONNX export. The fine-tune improved overall accuracy and calibration, learned STAY and a high-precision subset of BRAKE, but learned neither LEFT nor RIGHT. Labels, masks, decoding, and the loss are correct. The zero-turn result is consistent with a rare, fragmented turn class under a one-epoch, heavily clipped update; temperature and calibration are not the cause.

---

## VERIFIED FACTS

### 1. Checkpoint metadata and architecture

| Field | Original `models/kev-0.8b` | Fine `run/` |
|---|---|---|
| `base` | `Qwen/Qwen3.5-0.8B-Base` | same |
| `base_revision` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | same |
| `head.pt` `init_source.init_from` | `jaredpalmer/kev-0.8b` | `jaredpalmer/kev-0.8b@v1.0` |
| `head.pt` `init_source.resolved` | `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8` | `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` |
| `head.pt` `init_source` adapter/head SHA | `c81d5716…` / `39f4343c…` | `9b908623…` / `f400bd12…` |
| actual adapter file SHA-256 | `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` | `142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8` |
| actual head file SHA-256 | `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6` | `7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38` |
| LoRA rank | 16 | 16 |
| LoRA alpha / dropout | 32 / 0.05 | 32 / 0.05 |
| `lora_targets` | `all` | `all` |
| `head_dim` | 256 | 256 |
| `weights_dtype` | `fp32` | `fp32` |
| `temperature` | `2.3510958125672174` | `1.0` |
| `weights` | `lora` (field absent in original `head.pt`) | `lora` |

The original's own `head.pt` provenance points to an earlier snapshot (`54f4…`, adapter `c81d…`, head `39f4…`), but the files on disk are the released v1.0 bytes (`9b908…` / `f400b…`) — exactly the adapter/head hashes the fine run's `init_source` records. The tensor comparison below is therefore the released original vs the fine result.

- Adapter: **372 tensors** (186 `lora_A` + 186 `lora_B`), **10,822,656 parameters**. `adapter_config.json` lists `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, and 12 target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`. The original and fine adapter configs list the same set.
- Pointer head: 4 tensors, **524,800 parameters** — `q.weight [256, 1024]`, `q.bias [256]`, `k.weight [256, 1024]`, `k.bias [256]`, all fp32. It scores each option's closing token against the question's final token; the dot product is scaled by `1/sqrt(256) = 1/16` (`kev/model.py:208-218`). Adapter + head = **11,347,456 trainable parameters** (the training log's `11.3M`).
- The fine `training_config.json` and `head.pt` both record `init_source.tensors = 372`, `weights_sha256 = 9b908623…`, `head_sha256 = f400bd12…`, and the resolved snapshot `bf75…`.

### 2. `--init_from`: what was loaded, and how far it moved

`kev/train.py:553-556` calls `Checkpoint(a.init_from).warm_start(model, meta)`. `kev/checkpoint.py:318-331` compares the architecture fields first, loads all adapter tensors with `set_peft_model_state_dict`, and then loads `meta.head` with `model.head.load_state_dict`. The training log confirms:

```text
delta: warm start from .../snapshots/bf75a6a8848ea6960ff2ed108d9ed44c2941174f: 372 lora tensors and the pointer head loaded
```

`head_lr: 0.0` does **not** freeze the head. The CLI help says `0 = same as --lr` (`kev/train.py:376`); the head optimizer group is `a.head_lr or a.lr` (`kev/train.py:578`), and the OneCycleLR maximum is `[a.lr, a.head_lr or a.lr]` (`kev/train.py:588`). In this run that is `2e-5` for both groups. The empirical check agrees: all four head tensors changed.

**Head tensor differences (fine − original):**

| Tensor | Shape | max abs Δ | mean Δ | mean abs Δ | ‖original‖ | ‖fine‖ | ‖Δ‖/‖original‖ |
|---|---|---:|---:|---:|---:|---:|---:|
| `q.weight` | 256×1024 | 0.0013699154 | −2.935e−08 | 0.0001816514 | 9.2206459 | 9.2187824 | 1.297% |
| `q.bias` | 256 | 0.0005621845 | 5.335e−06 | 0.0001466818 | 0.2971398 | 0.2970411 | 0.997% |
| `k.weight` | 256×1024 | 0.0013545444 | −2.511e−07 | 0.0001579143 | 9.2169304 | 9.2140141 | 1.172% |
| `k.bias` | 256 | 0.0007598754 | 7.819e−06 | 0.0003025373 | 0.2898767 | 0.2891514 | 1.956% |
| **all head** | 524,800 | **0.0013699154** | — | **0.0001698364** | **13.0440036** | **13.0405842** | **1.236%** |

**Adapter differences (372 tensors, 10,822,656 elements):**

- 372/372 tensors changed.
- max abs Δ = **0.0025894188**.
- mean signed Δ = **1.5435e−07**; mean abs Δ = **0.0001895964**.
- ‖original‖ = **33.4701888**; ‖fine‖ = **33.4763539**; ‖Δ‖ = **0.8081153**; relative Frobenius movement = **2.414%**.

The head moved ~1.2% of its Frobenius norm and the adapter ~2.4%; the largest single weight change was 0.00137 in `q.weight` (weight std ≈ 0.018, so ~7.6% of one standard deviation) and 0.00259 in the adapter. This is real movement, but small relative to the parameter scales.

**The fine adapter and head were the ones used offline and in ONNX.**

- `logs/eval.log` records `original_checkpoint fp32 9b908623…`, `fine_checkpoint fp32 142e198a…`, and `loading fine_tuned: /workspace/runs/laneshift-kev-subset-e1`; the local `run/` files hash to the same `142e…` / `7ed5…` values.
- The fine predictions are not a copy of the original: **744 of 4,317** argmax decisions changed; the fine model predicts 119 BRAKE vs 0 for the original, and the original predicts 625 LEFT vs 0 for the fine model.
- `verify_fine_onnx.json` records the same fine adapter `142e…` and head `7ed5…`, and `verify_fine_onnx.py` loads `FINE_RUN` through `Checkpoint` before exporting/scoring the ONNX graph. The ONNX output differs from the original ONNX (Section 6).

### 3. Training recipe and loss

| Item | Value |
|---|---|
| Records | 4,096 (one epoch) |
| Optimizer steps | 512 |
| Batch / accumulation | 1 / 8 |
| Learning rate | `2e-5` (AdamW, weight decay 0.01) |
| Head learning rate | `0.0` → aliases `2e-5`, not frozen |
| Schedule | OneCycleLR, max_lr `[2e-5, 2e-5]`, `pct_start=0.1` |
| Gradient clip | global norm 1.0 (`MAX_GRAD_NORM`, `kev/train.py:503`) |
| Clipped steps | **506 / 512** |
| Pre-clip grad norm | mean **9.100446642578126**, max **34.220818** |
| Loss | **0.812** at step 10 → **0.393** at step 510 (~0.39) |
| Wall time | 2,463.34 s (41.1 min) |
| Forward tokens | 920,216 |

**Loss.** `question_loss` uses ordinary cross-entropy on the option index: `y = torch.tensor([q["label"]])` and `F.cross_entropy(z[None], y)` (`kev/train.py:50-51`). `permuted_copy` re-shuffles options for Choice questions with K ≥ 3 and rewrites the label to `perm.index(q["label"])` (`kev/train.py:33-35`), so a permutation cannot misalign labels. In this run `perm_kl = 0.0`, so the permutation branch at `kev/train.py:336` was not invoked and there was no permutation KL term; the only training loss was CE. `ord_w`, `label_smoothing`, `brier_w`, `focal_gamma`, and `anchor_w` were all 0.

**Optimization budget caveat.** The executed run was **1 epoch / 512 steps**. The handoff's default is 3 epochs (`train_gpu.sh:8`, `README.md:49`), which would have been 1,536 steps. The actual run therefore used one third of the planned optimization schedule. This is a material fact for the optimization interpretation in Section 8; it is not a code bug.

### 4. Test behavior: STAY and BRAKE vs turns

The untouched test split has 4,317 rows: **LEFT 55, RIGHT 67, BRAKE 562, STAY 3,633**. Both models were scored on the same records, option order, dtype path, and `temperature=1.0` (`kev-vast-eval.py:157-168`).

| Metric | Original | Fine-tuned |
|---|---:|---:|
| Accuracy | 0.7160 | **0.8663** |
| Macro F1 | 0.2143 | **0.3145** |
| Log loss | 0.9268 | **0.4093** |
| Brier | 0.5417 | **0.2090** |
| ECE | 0.2876 | **0.0395** |
| Mean confidence | 0.4647 | 0.9059 |
| Predicted LEFT / STAY / RIGHT / BRAKE | 625 / 3,692 / 0 / 0 | **0 / 4,198 / 0 / 119** |

**Fine confusion matrix** (rows = truth LEFT, STAY, RIGHT, BRAKE; columns = predicted LEFT, STAY, RIGHT, BRAKE):

```text
[[   0,   55,    0,    0],
 [   0, 3627,    0,    6],
 [   0,   67,    0,    0],
 [   0,  449,    0,  113]]
```

**Original confusion matrix:**

```text
[[   5,   50,    0,    0],
 [ 547, 3086,    0,    0],
 [   8,   59,    0,    0],
 [  65,  497,    0,    0]]
```

**Fine per-class:**

| Action | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| STAY | 3,633 | 0.8640 | **0.9983** | 0.9263 |
| BRAKE | 562 | **0.9496** | 0.2011 | 0.3319 |
| LEFT | 55 | 0.0000 | 0.0000 | 0.0000 |
| RIGHT | 67 | 0.0000 | 0.0000 | 0.0000 |

**Turn rows.** On every LEFT and RIGHT test row the fine model's top-1 is STAY; the correct turn is rank 3 or 4; its probability is 0.6–2.6%, close to the subset priors.

| Rows | n | Fine mean P(correct) | Fine mean P(STAY) | Fine mean log-odds (correct/STAY) | Fine rank counts | Original mean P(correct) | Original mean log-odds |
|---|---:|---:|---:|---:|---|---:|---:|
| LEFT | 55 | **0.012388** | 0.900159 | **−4.3373** | rank 3: 34; rank 4: 21 | 0.376750 | −0.1187 |
| RIGHT | 67 | **0.014236** | 0.881610 | **−4.1773** | rank 3: 55; rank 4: 12 | 0.124804 | −1.4459 |
| BRAKE | 562 | **0.355137** | 0.627590 | −0.5523 | rank 1: 113; rank 2: 449 | 0.218006 | −0.8754 |
| STAY | 3,633 | 0.918094 | — | 0 | rank 1: 3,627; rank 2: 6 | 0.451436 | 0 |

- Fine turn probabilities are tightly near the prior: LEFT mean 1.24% (range 0.63–2.60%), RIGHT mean 1.42% (range 0.68–2.52%). No turn row exceeds 2.6%.
- Among the 562 BRAKE rows, the fine BRAKE probabilities are bimodal: 113 are above 0.9 (111 between 0.9 and 0.99, 2 above 0.99), 449 are lower; median 0.264, 10th percentile 0.043, 90th percentile 0.983. Exactly 113 BRAKE-labelled rows have P(BRAKE) > P(STAY); with 6 STAY false positives that gives 119 BRAKE predictions, 113 correct.
- Fine BRAKE recall is only 20.1%, but its precision is 95.0%: it learned a high-confidence detector for a subset of BRAKE contexts, not a complete BRAKE policy.
- Fine STAY recall is 99.83%; only 6 STAY rows are misclassified, all as BRAKE.

**Fine vs original decisions.** 3,573 / 4,317 argmax decisions are unchanged; 744 changed. The fine model removed the original's 625 LEFT predictions (5 true LEFT, 547 STAY, 65 BRAKE, 8 RIGHT) and replaced them with STAY, while adding 119 BRAKE predictions (113 true BRAKE, 6 STAY). The original never predicted RIGHT or BRAKE.

**Class priors and optimizer exposure.** Subset priors: STAY 3,439/4,096 = 83.96%; BRAKE 552/4,096 = **13.48%**; LEFT 55/4,096 = **1.34%**; RIGHT 50/4,096 = **1.22%**; turns 105/4,096 = **2.56%**. At 512 steps × accumulation 8, the expected number of steps containing at least one example is **96.0 for turns**, **351.2 for BRAKE**, **52.5 for LEFT**, and **47.9 for RIGHT**. The observed turn probabilities (1.24% LEFT, 1.42% RIGHT) are close to the subset priors (1.34%, 1.22%), within 0.2 percentage points; the observed BRAKE probability on BRAKE rows (35.5%) is well above its 13.48% prior.

**Turn fragmentation in the subset.** The 105 turn rows are spread over **73 seeds**: 51 seeds have 1 turn, 14 have 2, 6 have 3, 2 have 4; max 4 per seed, mean 1.44. All **105/105** turn rows have no adjacent sampled turn within 4 decisions. The full teacher-only train set has **465** turn rows and up to **12** per seed; the subset retains **105 (22.6%)**, and **52 of 125** teacher turn-bearing seeds lose every turn. The full all-source train set's max 54 turns per seed is the excluded human weaving episode. BRAKE, by contrast, has 552 rows across 116 seeds (mean 4.76/seed, max 10/seed); it is also non-adjacent (0 of 436 adjacent pairs within 4 decisions), but it has 5.26× the rows and 1.6× the seeds of the turn class.

### 5. Bug rule-out: labels, masks, decoding, loss

| Candidate | Ruled out? | Evidence |
|---|---|---|
| Label / class-index bug | **Yes** | Test counts match eval metrics (55/67/562/3,633). Data audit verified the label is in the legal set on **25,427/25,427** rows and the class index is consistent. Every turn test row has its label in `offered` (55/55 LEFT, 67/67 RIGHT). |
| Option-mask bug | **Yes** | The fine model assigns nonzero probability to the correct turn on **122/122** turn rows (LEFT min 0.006339, RIGHT min 0.006794). A mask that zeroed the turn would produce 0. Offered sets are respected; the label is always offered. |
| Decoding / action-map bug | **Yes** | `score()` normalizes over the four actions and takes `argmax` (`kev-vast-eval.py:120-150`); stored `predicted` equals argmax on **4,317/4,317** rows for both models. Fine STAY recall 99.83% and BRAKE precision 94.96% are coherent; a global action-map error would not produce that pattern. |
| Loss / label-index bug | **Yes** | Training loss is pure CE on `q["label"]` (`kev/train.py:50-51`); `ord_w=0`, `label_smoothing=0`, `brier_w=0`, `focal_gamma=0`, `anchor_w=0`, and `perm_kl=0` (so the permutation branch was inactive). Loss fell 0.812 → 0.393. |
| Training mask / option-isolation bug | **No evidence** | `option_isolation=0`; all options are visible. The model learned STAY and a BRAKE subset, so the training path is not globally broken. |

The failure is therefore behavioral, not a label/mask/decoding/loss implementation bug.

### 6. ONNX export and parity

| Artifact | Fine | Original |
|---|---|---|
| ONNX graph SHA-256 | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| External data SHA-256 | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |
| Graph size | 7,011,147 B | 7,012,901 B |
| Data size | 1,506,886,400 B | 1,506,886,400 B |
| Graph-structure SHA-256 (initializers stripped) | `fffb5469ab16bf419ea9a9d0e76047ecf4a105adc64ad6e4faa1acd9a72a361b` | `89da2de2f74d567f2caa937fd169dba81fad956c7247a9b6bd47f924e251f78c` |

Both graphs are opset 17 / IR 8. The inputs are `input_ids [1,256]`, `position_ids [1,256]`, `attention_mask [1,256]`, `decide_idx [1]`, and dynamic `opt_idx ['K']`; the output is dynamic `logits ['K']`. Both use **324 external initializers** in `kev07b_fp16.data`; the first is `lm.embed_tokens.weight` at offset 0, length 508,559,360. The fine graph is not a reused original graph or blob: both full-file hashes and the graph-structure hashes differ.

**Fine vs original ONNX logits** (same 35-real-token probe, padded to L256):

| Model | Logits | Softmax |
|---|---|---|
| Fine | `[0.3782473, 0.4216261]` | `[0.4891570, 0.5108430]` |
| Original | `[0.2640466, 0.2133202]` | `[0.5126789, 0.4873211]` |

Max abs logit difference = **0.2083058**; the argmax flips from option 1 (original) to option 2 (fine).

**Fine ONNX vs fine torch parity:** torch fp32 logits `[0.4052193, 0.4746889]`; ONNX fp16 logits `[0.3782473, 0.4216261]`; differences `[-0.0269720, -0.0530629]`; max abs = **0.0530629** (the reported 0.053). Softmax differs by ~0.0066 per option (0.489157/0.510843 vs 0.482640/0.517360), with the same argmax. This is consistent with fp16-export rounding of the same fine checkpoint; the export loads `FINE_RUN` and the recorded fine adapter/head hashes match.

### 7. Temperature and calibration

| Item | Original | Fine |
|---|---|---|
| `head.pt` temperature | **2.3510958125672174** | **1.0** |
| Evaluation temperature | 1.0 | 1.0 |
| ECE at T=1.0 | 0.2876 | **0.0395** |
| ECE after applying 2.3511 to the stored T=1 probabilities | 0.3721 | 0.2006 |
| Argmax changes when applying 2.3511 | **0** | **0** |

The original's fitted temperature is documented in its `temperature_fit` metadata (five folds, selected 2.3511; out-of-fold ECE 0.0322 on its development rows). The fine `head.pt` has no `temperature_fit` and stores 1.0. The evaluation explicitly used `LoadOptions(..., temperature=1.0)` for both models (`kev-vast-eval.py:168`), and `Checkpoint.load` sets `m.head.temperature` to that override (`kev/checkpoint.py:241`). `PointerHead.forward` divides logits by T only in eval mode (`kev/model.py:218`); softmax is monotonic, so any positive T preserves the argmax. Empirically, applying 2.3511 to either model's T=1 probabilities changes **zero** argmax decisions.

The fine model's ECE 0.0395 is good at T=1.0. The original's ECE at T=1.0 is 0.2876; applying its own 2.3511 to this gameplay test makes it worse (0.3721), so the fine checkpoint's reset to 1.0 is a metadata/calibration regression relative to the original's fitted value, not a cause of the zero-turn result. Overall ECE is dominated by the 84% STAY class; good ECE does not imply rare-turn discrimination.

---

## INTERPRETATIONS

### The near-prior turn output is the expected CE solution for a rare, unrepeated class

The facts above support a two-part explanation:

1. **Coverage / fragmentation (leading hypothesis).** The subset contains 105 turn rows (2.56%) spread across 73 seeds, with 51 seeds contributing exactly one turn and a maximum of four; every turn row is isolated (no adjacent sampled turn within four decisions), and 52 of the 125 teacher turn-bearing seeds lost every turn. The full teacher set has 465 turns and up to 12 per seed. Each turn context is effectively seen once in one epoch. Under CE, a model that cannot identify the conditional rule minimizes loss by predicting the class prior. The observed turn probabilities — 1.24% LEFT and 1.42% RIGHT — are close to the subset priors (1.34% and 1.22%), and the log-odds are strongly negative (−4.34 and −4.18). This is the signature of prior-fitting, not a mask or label bug.
2. **BRAKE had enough signal to learn a subset detector.** BRAKE has 552 rows (13.48%) across 116 seeds, 5.26× the turn rows and 1.6× the seeds; an expected 351 of 512 optimizer steps contain at least one BRAKE versus 96 for turns. A prior-only model already assigns ~13.5% to BRAKE, and the fine model moves to 35.5% on BRAKE rows: 113 true BRAKE rows are above 0.9 (plus 5 false positives above 0.9), and overall BRAKE precision is 95.0%. That is a learned conditional signal, albeit only 20.1% recall. The contrast between BRAKE and turns is therefore a contrast in coverage and repeatability, not evidence that the model cannot learn a fourth action.

### Optimization is a contributing hypothesis, not a global failure

The run used 512 steps (one epoch, one third of the handoff's default 3 epochs), LR 2e-5, and 506/512 steps hit the 1.0 global-norm clip. Mean pre-clip norm was 9.10 (average scale ≈ 1/9.1 ≈ 0.11) and max 34.22 (≈ 1/34.2 ≈ 0.03). The head moved only 1.24% of its Frobenius norm and the adapter 2.41%; max single-weight changes were 0.00137 and 0.00259. This is a small effective update. It plausibly contributes to the failure to re-rank a 1.2% class against an 84% STAY prior. It cannot be the whole story, because the same optimizer learned STAY and a BRAKE subset. The executed 1-epoch/512-step budget versus the handoff's 3-epoch default is a material reduction in signal.

### The original prior explains the original model's errors, not the fine model's zero turns

The original model predicted 625 LEFT (only 5 true), 0 RIGHT, and 0 BRAKE. The fine model suppressed LEFT to 0, replaced those predictions with STAY, and learned 119 BRAKE. The original's LEFT bias therefore shaped the original's error pattern, but the fine model overrode it; it did not inherit a "LEFT-only" policy. The original prior is a contributing context, not a sufficient explanation for why both turns remained at zero.

### Calibration and temperature are not the cause

The original carries T=2.3511; the fine checkpoint stores T=1.0. Both were evaluated at T=1.0. The fine ECE is 0.0395; the original ECE at T=1.0 is 0.2876. Applying the original's 2.3511 to either model changes zero argmax decisions, and makes ECE worse on this gameplay test (0.3721 original, 0.2006 fine). Temperature cannot reorder options; it is a metadata/calibration difference, not the cause of zero turns. The fine model's good ECE reflects its confident STAY/BRAKE behavior, not turn discrimination.

### Demonstrated facts vs hypotheses

| Candidate | Demonstrated facts | Status |
|---|---|---|
| Turn coverage / fragmentation | 105 turns, 73 seeds, 51 singletons, max 4/seed, 105/105 isolated, 52/125 teacher turn seeds dropped; full teacher 465 turns, max 12/seed | **Leading hypothesis**, supported by the near-prior outputs and the BRAKE contrast; not proven causal without retraining |
| Optimization / signal strength | 512 steps (1 epoch vs handoff 3), 506/512 clipped, mean grad 9.10/max 34.22, head 1.24% / adapter 2.41% relative movement | **Contributing hypothesis**; not a global failure because BRAKE learned |
| Original action prior | Original 625 LEFT / 0 RIGHT / 0 BRAKE; fine 0 LEFT / 0 RIGHT / 119 BRAKE | **Contributing context**; explains original errors, not the fine zero turns |
| Calibration / temperature | T 2.3511 vs 1.0; eval T=1.0; ECE 0.0395 vs 0.2876; 0 argmax changes with T | **Ruled out as cause** |
| Label / mask / decoding / loss bugs | 25,427/25,427 legal labels; 122/122 turn rows nonzero; predicted==argmax 4,317/4,317; pure CE | **Ruled out** |

**Cause ranking:** (1) rare, fragmented turn coverage plus prior-fitting; (2) short, heavily clipped one-epoch optimization; (3) original LEFT-biased prior as context; (4) calibration/temperature is not causal; (5) no label/mask/decoding/loss bug.

---

## UNANSWERED QUESTIONS

1. Would the same subset for 2 epochs with `--head_lr 2e-4` (the prior analysis's smallest experiment) move either turn above STAY? Not run.
2. Would a turn-clustered subset — all 465 teacher turns, repeated turn-dense episodes, ~1,200 BRAKE, STAY fill — fix the zero-turn result? Not run.
3. What is the causal split between coverage/fragmentation and optimization? No controlled ablation isolates them.
4. Why did RIGHT remain at zero even though the original assigned it some probability (mean 12.5% on RIGHT rows) and the fine model suppressed it to 1.42%? Unknown.
5. Why does BRAKE learn only a 20%-recall, 95%-precision subset rather than a broader BRAKE policy? The bimodal BRAKE probability distribution suggests context-specific patterns; not identified.
6. Does the fine ONNX fp16 export preserve the zero-turn behavior on the full test set? Only one 35-token non-turn probe was checked; no full ONNX evaluation exists.
7. Is the fine checkpoint's T=1.0 calibration adequate for deployment? ECE 0.0395 on this test is good, but it is not a fitted temperature and the test distribution differs from training.
8. Does the test-set density shift (1.5-density rows are 47.6% of test vs 27.4% of train; 20.2 TV points) affect the turn/BRAKE conclusions? Not isolated.
9. Are the 105 retained turn rows representative of the full teacher turn distribution, given that 52 turn-bearing seeds were dropped entirely? Unknown.
10. Would class weighting, focal loss, or turn oversampling help? Not tested; `focal_gamma=0`, `label_smoothing=0`, and the subset used the natural distribution with no rare-action oversampling.

The checkpoint is valid and the evaluation is fair, but it should not be deployed as a driving policy: it predicts zero turns and only 20% of BRAKE.
