# Reproduce the LANESHIFT research release

All paths are relative to the repository root. No training is required to
reproduce the published evaluation; the original and fine-tuned predictions and
gameplay logs are checked in.

## 1. Verify the data

```powershell
# full dataset (16 files)
Push-Location datasets\kev-v2-scene
Get-Content SHA256SUMS | ForEach-Object {
  $h, $p = $_ -split '  ', 2
  if ((Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $h) { throw "bad hash: $p" }
}
Pop-Location

# exact 4,096-row subset
Push-Location datasets\kev-subset-4096
Get-Content SHA256SUMS | ForEach-Object {
  $h, $p = $_ -split '  ', 2
  if ((Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $h) { throw "bad hash: $p" }
}
Pop-Location
```

The untouched test hash is
`d16f65ab244b587ad0b7a56e702eb426543dedd4f7c5e4548af9d7bf58471da4`.

## 2. Recalculate the offline statistics

```powershell
C:\Users\Dhane\dev\cris@ds\LANESHIFT\.venv-host\Scripts\python.exe `
  research\scripts\calc_stats.py
```

The script reads `datasets/kev-v2-scene/kev_test.jsonl` and
`research/offline/predictions_*.jsonl`, then writes
`research/audits/RECALCULATED-STATS.json`.

## 3. Inspect the actual training configuration

`research/training/training_config.json` records:

- data: the 4,096-row subset
- epochs: 1
- optimizer steps: 512
- batch 1, accumulation 8
- LR 2e-5
- LoRA rank 16, head_dim 256, fp32 weights, bf16 compute
- gradient clip 1.0

`research/training/training_metrics.json` records 506/512 clipped steps,
mean pre-clip gradient norm 9.10, max 34.22, loss 0.812 → 0.393, and wall time
2,463.34 s.

## 4. Inspect the model artifacts

The weights are release assets, not Git files. See
`research/model-artifacts/EXTERNAL-ARTIFACTS.md` for hashes and `gh release`
commands. The original ONNX graph/blob and the fine ONNX graph/blob are separate;
the fine export uses a new 1.507 GB blob, never the original weights.

## 5. Gameplay comparison

`research/gameplay/original-12.episodes.jsonl` and
`research/gameplay/fine-12.episodes.jsonl` contain the 12 paired seeds
930001–930012. The raw recordings are intentionally excluded as disposable
runtime data. The comparison and statistics are in
`research/audits/GAMEPLAY-COMPARISON.md` and
`research/audits/STATISTICS-AUDIT.md`.

## 6. Dataset builder/validator

```powershell
C:\Users\Dhane\dev\cris@ds\LANESHIFT\.venv-host\Scripts\python.exe `
  python\validate_kev_v2_scene.py --root datasets\kev-v2-scene
```

The builder is `python/build_kev_v2_scene.py`; it expects the preserved
recordings, which are not part of this Git release. The checked-in dataset and
cards are the authoritative research artifact.
