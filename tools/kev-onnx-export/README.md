# Kev ONNX exporter source

This directory is the source used to export the Kev-0.8B decision pass to ONNX.
It contains no weights, graph, tokenizer or external-data blob.

## Files

- `kev_onnx.py` — `Decide` wrapper, checkpoint loader and tracing patches.
- `inputs.py` — tiny fixed input record used for graph export.
- `export_onnx.py` — exports graphs and writes a new external fp16 blob.
- `export_reuse.py` — exports a fixed-length graph that reuses the **existing**
  blob. The shipped L256 graph used this path.
- `run_onnx.py` — ONNX Runtime parity runner.
- `test256.py` — L256 graph smoke test.
- `bench_torch.py` — PyTorch reference timing.
- `REPORT.md`, `REPORT-REBUILD.md` — export history and parity notes.

## Run

Use the Kev repository at the pinned commit
`2074cc44256200bb37540103fa4572901faa10f3` and an environment with PyTorch,
`onnx`, `onnxruntime` and the base tokenizer.

```powershell
python export_onnx.py
```

The source downloads `jaredpalmer/kev-0.8b@v1.0` through
`kev.checkpoint.Checkpoint`. The exported graph is shape-compatible with
LANESHIFT's `python/policies/onnx_kev.py`:

- inputs: `input_ids`, `position_ids`, `attention_mask`, `decide_idx`,
  `opt_idx`
- output: `logits`
- fixed sequence length 256; `opt_idx` is dynamic.

## Important: fine-tuned checkpoints

`export_reuse.py` checks the state-dict layout and points the graph at the
existing `kev07b_fp16.data`. A fine-tuned checkpoint has the same shapes, so the
size check passes while the values remain the **original** weights. For a
fine-tuned checkpoint, write a new blob from the merged state dict (the
`build_data` path in `export_onnx.py`) and attach that blob to the new graph.

Verify torch/ONNX parity before using the graph. The shipped files and their
SHA-256 hashes are in `docs/MODELS.md` and `docs/model-manifest.json`.
