# Original vs fine-tuned Kev — matched LANESHIFT gameplay

Date: 2026-11-05. Local CPU only. No cloud, no retraining, no dataset changes.

## Exported model verification

The fine graph is a new export from the actual run checkpoint, not the original
weights.

| Artifact | SHA-256 |
|---|---|
| Original ONNX | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| Original data blob | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |
| Fine ONNX | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` |
| Fine data blob | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` |
| Fine adapter | `142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8` |
| Fine head | `7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38` |

Fine graph: 7.0 MB, new 1.507 GB fp16 blob, L256, dynamic option count.
On the same 256-token input:

```text
original ONNX logits [0.264047, 0.213320]  probs [0.512679, 0.487321]
fine ONNX     logits [0.378247, 0.421626]  probs [0.489157, 0.510843]
fine torch    logits [0.405219, 0.474689]
```

- Max fine-ONNX vs original-ONNX logit difference: **0.2083**
- Max fine-ONNX vs fine-torch logit difference: **0.0531**
- The graph, blob, adapter and head all differ from the original. The fine
  weights are real.

## Matched conditions

- Fresh seeds: **930001–930012** (12 seeds)
- 60 s episodes, traffic density **1.5**, simulation speed **2.0×**
- L256 ONNX, scene prompt, CPU execution, 8 threads
- Same Godot build, project, seeds, duration, density and prompt for both models
- Recordings: `session_20261105_201308_176` (original) and
  `session_20261105_201408_325` (fine-tuned)

## Results

| Metric | Original Kev | Fine-tuned Kev |
|---|---:|---:|
| Episodes | 12 | 12 |
| Crashes | **12/12** | **12/12** |
| Completions | 0 | 0 |
| Mean survival | 6.886 s | 7.956 s |
| Median survival | **6.183 s** | 5.350 s |
| Mean distance | 182.3 | 212.2 |
| Median distance | **162.6** | 139.3 |
| Mean score | 115.2 | 127.7 |
| Median score | **87.9** | 64.2 |
| Mean overtaken_count | 2.50 | 2.67 |
| Median overtaken_count | **1.5** | 1.0 |
| Actions applied | 21 | 19 |
| LEFT / RIGHT / STAY / BRAKE | 9 / 0 / 12 / 0 | **0 / 0 / 19 / 0** |
| BRAKE frequency | 0.0% | 0.0% |
| Lane changes | **9** | **0** |
| Rejected actions | **5** | 0 |
| Final lanes | 0:7, 1:5 | 1:12 |
| Mean decision latency | 1.170 s | 1.581 s |
| Decisions requested | 24 | 19 |

Paired per-seed differences, fine-tuned minus original:

| Metric | Mean diff | Median diff | Fine wins | Original wins | Ties |
|---|---:|---:|---:|---:|---:|
| Survival | +1.069 s | 0.000 | 5 | 4 | 3 |
| Distance | +29.94 | 0.000 | 5 | 4 | 3 |
| Score | +12.51 | 0.000 | 5 | 4 | 3 |
| Overtaken count | +0.167 | 0.000 | 3 | 2 | 7 |

The mean improvement is driven by two seeds (930005 and 930012); the median is
zero or worse. The fine model never changes lane and never brakes.

## Which model drives better?

**Neither model is a competent driver.** Both crash on every one of the 12
matched seeds. The fine-tuned model has a slightly better mean survival,
distance and score, but a worse median survival and median score. It stays in
lane 1 for all 12 episodes, never changes lane and never brakes. The original
attempts nine lane changes, five of which are rejected, and ends outside the
start lane in seven episodes. The original is more active and more error-prone;
the fine-tuned model is inert but slightly less bad on two outlier seeds.

This is not evidence that the fine-tune drives better. The offline accuracy gain
does not translate into a usable driving policy, consistent with the zero-turn
offline result.

## Important confound

The models have different ONNX latency: **1.170 s** original vs **1.581 s**
fine-tuned per decision. The game therefore applied 21 original actions versus
19 fine-tuned actions over the same conditions, and the fine model had fewer
control opportunities. Both latency values are far above the game's decision
cadence. A fair control-policy comparison needs a fixed decision interval longer
than both latencies, and ideally the same inference backend and batch size. This
run measures the deployed CPU setup, not an isolated policy difference.

## Preserved results

- `original-12.csv`, `original-12.episodes.jsonl`, `original-12.onnx_kev.metrics.json`
- `fine-12.csv`, `fine-12.episodes.jsonl`, `fine-12.onnx_kev.metrics.json`
- 15 s smoke results for each model
- `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\models\kev-finetuned-subset-e1\verify_fine_onnx.json`
- New ONNX graph and blob under `models\kev-finetuned-subset-e1\`
- Original model files and the untouched test split remain unchanged

The experiment runner was corrected only to pass `--kev-prompt scene` and the
model's external-data path. Game mechanics, both checkpoints and the dataset
were not altered.
