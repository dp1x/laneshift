# Kev LoRA subset run — actual results

Date: 2026-11-05. Instance: Vast #59269999, 1× NVIDIA A100-PCIE-40GB,
40,960 MiB, driver 595.99.02, CUDA 13.2.

## Environment

- Python 3.12.15, isolated venv `/workspace/kev-venv`
- PyTorch 2.8.0+cu128, CUDA available, BF16 supported
- Transformers 5.22.1, PEFT 0.21.2, pinned Kev `2074cc44256200bb37540103fa4572901faa10f3`
- Original checkpoint: `jaredpalmer/kev-0.8b@v1.0`, `--init_from`
- Reference Gated DeltaNet kernels remained active; the optional CUDA kernel
  install was aborted. Training logs confirm the fallback path was still used.

## Training subset

4,096 rows from the existing training split only. The original dataset and test
split were not modified.

- 124 seeds / 124 episodes, all traffic densities 0.7–3.5
- Teacher-only. All 59 human rows excluded to avoid teaching weaving.
- Crash episodes excluded; duplicate prompt hashes removed.
- Actions: STAY 3,439; BRAKE 552; LEFT 55; RIGHT 50. Natural distribution,
  with no rare-action oversampling.
- One epoch, batch 1, accum 8, bf16 compute, **fp32 LoRA weights**, checkpointing
  on, `--max_state 384`.

## Training result

| Metric | Value |
|---|---:|
| Records | 4,096 |
| Optimizer steps | 512 |
| Wall time | 2,463.34 s (41.1 min) |
| Mean step | 4.811 s |
| Mean per record | 0.603 s |
| Peak GPU allocation | 3.36 GB |
| Peak RSS | 5.67 GB |
| Checkpoint save | 6.9 s |
| Loss trend | 0.812 → ~0.39 |

Final adapter sha256:
`142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8`
Final head sha256:
`7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38`
The adapter differs from the released checkpoint and is `weights_dtype=fp32`.

## Test-set comparison

Complete untouched test split: 4,317 rows. Both models used the same records,
option sets, option order and temperature 1.0.

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Original Kev | 0.7160 | 0.2143 | 0.9268 | 0.5417 | 0.2876 |
| Fine-tuned | **0.8663** | **0.3145** | **0.4093** | **0.2090** | **0.0395** |

Per-class, fine-tuned:

| Action | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| STAY | 3,633 | 0.8640 | 0.9983 | 0.9263 |
| BRAKE | 562 | 0.9496 | 0.2011 | 0.3319 |
| LEFT | 55 | 0.0000 | 0.0000 | 0.0000 |
| RIGHT | 67 | 0.0000 | 0.0000 | 0.0000 |

Fine-tuned predictions: STAY 4,198, BRAKE 119, LEFT 0, RIGHT 0.
Original predictions: STAY 3,692, LEFT 625, BRAKE 0, RIGHT 0.

## Verdict

The checkpoint is valid and the fine-tune materially improves accuracy,
log-loss, Brier score and calibration. It **does not learn lane changes**: it
predicts zero LEFT and zero RIGHT on the complete test set. BRAKE recall is only
20%. The accuracy gain is driven by STAY (99.8% recall) and a small number of
high-precision BRAKE predictions.

This is a real one-epoch result, not a deployable driving policy. The 4,096-row
subset contained only 105 turn examples, which was insufficient. A follow-up
should use a turn-enriched subset (roughly 10–15% LEFT/RIGHT, still no human
weaving) and/or additional epochs before claiming a working driving checkpoint.

## Artifacts

- `run/` — adapter, pointer head, tokenizer, training config, metrics
- `results/` — `eval_metrics.json`, both prediction files, checkpoint manifest
- `logs/` — setup, smoke, training, evaluation logs
- `subset/` — the exact 4,096-row subset, provenance and card
- `smoke/` — one-step smoke metrics
- `SHA256SUMS` — hashes of all local deliverables

The Vast connection closed during the final provenance download. The truncated
provenance copy was replaced with the intact local source file whose SHA-256
matches the remote hash `451b9e1c…`.
