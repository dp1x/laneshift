# Kev model artifacts and download instructions

**No model weights, external-data blobs or tokenizers are stored in this Git
repository.** This file and `docs/model-manifest.json` carry the identity,
hashes, local locations and reconstruction instructions.

## Released checkpoint

| Field | Value |
| --- | --- |
| Model | `jaredpalmer/kev-0.8b@v1.0` |
| Base | `Qwen/Qwen3.5-0.8B-Base` |
| Base revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05 |
| Head | Pointer head, dimension 256 |
| Option isolation | Off |
| Special embeddings | Off |
| Weights dtype | fp32 |
| Shipped temperature | 2.3510958125672174 |
| Upstream repository | https://github.com/jaredpalmer/kev |
| Upstream commit used for this project | `2074cc44256200bb37540103fa4572901faa10f3` |
| Hugging Face model page | https://huggingface.co/jaredpalmer/kev-0.8b |

## Local ONNX inference artifacts

These are the files the game uses. They live outside Git:

```text
C:\LANESHIFT2D-Kev-model\
  kev07b_fp16_L256.onnx
  kev07b_fp16.data
  tokenizer.json
```

A preserved transfer copy is at
`R:\LANESHIFT-TRANSFER\LANESHIFT2D-Kev-model\`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `kev07b_fp16_L256.onnx` | 7,012,901 | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| `kev07b_fp16.data` | 1,506,886,400 | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |
| `tokenizer.json` | 12,807,196 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |

Verify with:

```powershell
Get-FileHash C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx -Algorithm SHA256
Get-FileHash C:\LANESHIFT2D-Kev-model\kev07b_fp16.data -Algorithm SHA256
Get-FileHash C:\LANESHIFT2D-Kev-model\tokenizer.json -Algorithm SHA256
```

## Rebuilding the ONNX graph

The exporter source is in `tools/kev-onnx-export/`. It is not a complete
training environment; use a clone of the pinned Kev repository.

1. Clone and install Kev:

   ```powershell
   git clone https://github.com/jaredpalmer/kev.git C:\Users\Dhane\dev\kev
   git -C C:\Users\Dhane\dev\kev checkout 2074cc44256200bb37540103fa4572901faa10f3
   ```

2. Use the Kev environment with PyTorch, `onnx`, `onnxruntime` and the base
   tokenizer. The exporter's `kev_onnx.py` loads
   `jaredpalmer/kev-0.8b@v1.0`; Hugging Face downloads the adapter and head.
3. Export the fixed-length graph and external fp16 blob:

   ```powershell
   python tools\kev-onnx-export\export_onnx.py
   ```

   The shipped L256 graph was produced by `export_reuse.py` reusing the
   existing blob. **For a fine-tuned checkpoint, do not use that reuse path**:
   it checks the state-dict layout and points the graph at the old blob. Write
   a new blob from the fine-tuned state dict (the `build_data` path in
   `export_onnx.py`, or a 256-token adaptation of it).
4. Copy the tokenizer from the base model or the Kev checkpoint, and verify all
   three hashes above. The ONNX graph and `.data` blob are deliberately not
   committed.

## Fine-tuned artifacts

A fine-tuned Kev checkpoint produces an adapter/head directory (tens of MB) and
an ONNX graph plus a new external-data blob (~1.5 GB). Store those in Hugging
Face Hub or Google Drive, not Git. Record their hashes in a new manifest entry
and use the `fine_tuned_model` label source when loading them.

The recommended first run is in `notebooks/kev_gpu_benchmark.ipynb` and the
project training notes. The base checkpoint must be loaded with `--init_from
jaredpalmer/kev-0.8b@v1.0`, not trained from the bare Qwen base.
