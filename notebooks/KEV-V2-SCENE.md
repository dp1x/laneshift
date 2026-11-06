# kev-v2-scene fine-tuning runbook

This is the production dataset path for the next Kev-0.8B LoRA run. It follows
the probe and the human-session audit; the dataset card is the authority for
counts and hashes.

## Dataset locations

- Full JSONL data: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\.portable-dev\kev-v2-scene`
- Cards, manifest and seed plan: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\datasets\kev-v2-scene`
- Recording sources: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\.portable-dev\recordings`
- Seed plan: `C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-v2-dataset-20261104\seed_plan.json`

Upload these seven files to `/content/data` in the Colab notebook:

`kev_train.jsonl`, `kev_validation.jsonl`, `kev_test.jsonl`, their three
`*.provenance.jsonl` files, and `manifest.json`.

## Prompt and context length

The prompt is the schema-2 **scene** renderer, `vehicle_limit=9`. It is the
same `OnnxKevPolicy._encode` path used by `play_kev.ps1` and
`export_human_decisions.py`. Every included row is at most **256 total tokens**;
the dataset builder rejects anything over that budget.

`--max_state 384` is Kev's minimum **training state-token cap**. It is not a
384-token model context: Kev's `MAX_STATE`/`MAX_TRAIN_STATE` constants govern
the encoder, while the Qwen backbone has a much larger window. All production
rows also fit the L256 serving graph.

The original and fine-tuned checkpoints must be evaluated at the **same graph
length**. For the production comparison, export both at L256 and pass
`--onnx-length 256`. A 384-token fine-tuned graph compared with the original
L256 graph is not a controlled comparison. If a 384-context experiment is
wanted, export both checkpoints at 384 and compare them at 384.

## Training

The existing `notebooks/kev_finetune.ipynb` accepts the new scene rows (its
validation cell checks for `v2\n` as well as `c1\n`). Use the data files above.
The reference recipe is LoRA rank 16, head dimension 256, lr `2e-5`, batch 1,
accumulation 8, no distractor augmentation:

```text
python -m kev.train \
  --data /content/data/kev_train.jsonl \
  --base Qwen/Qwen3.5-0.8B-Base \
  --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --init_from jaredpalmer/kev-0.8b@v1.0 \
  --lora 16 --head_dim 256 --lora_targets all \
  --lr 2e-5 --epochs 3 --batch 1 --accum 8 \
  --dtype bf16 --weights_dtype fp32 --checkpointing 1 \
  --max_state 384 \
  --p_none 0 --p_none_distract 0 --p_distract 0 --p_none_pair 0 \
  --device cuda --seed 1 --out /content/runs/laneshift-kev-v2-scene
```

Use fp32 dtype on a T4 and 2 epochs; use bf16 compute and **fp32 LoRA weights**
on L4/A100/H100. Keep `--weights_dtype fp32`: bf16 weights disable merge parity
for the fine-tuned side. Kev's LoRA path has no mid-run resume, so use an
on-demand GPU and restart from scratch if preempted. Keep `kev_test.jsonl`
untouched until the final evaluation.

## Evaluation

1. Offline: score the original and fine-tuned checkpoints on the same
   validation/test rows, with the same option order and temperature. The
   notebook writes `eval_metrics.json`, per-row predictions, and
   `checkpoint_manifest.json`.
2. Gameplay: export **both** checkpoints to ONNX at L256. The fine-tuned
   export must write a new external-data blob; `export_reuse.py` reuses the
   original blob and is not valid for fine-tuned weights.
3. Run both through `python/experiment_runner.py` on the same test seeds and
   conditions, with `--policies onnx_kev --onnx-length 256`, and report score,
   survival, collisions, action distribution, decisions, and latency.
4. Do not use the test seeds for hyperparameter selection or early stopping.

## Regeneration

```powershell
python python\build_kev_v2_scene.py `
  --plan C:\Users\Dhane\dev\cris@ds\LANESHIFT\kev-v2-dataset-20261104\seed_plan.json `
  --output C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\.portable-dev\kev-v2-scene `
  --public-output C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\datasets\kev-v2-scene

python python\validate_kev_v2_scene.py `
  --root C:\Users\Dhane\dev\cris@ds\LANESHIFT\cris@ds\.portable-dev\kev-v2-scene
```

The builder reads the preserved recordings; it does not modify them.
