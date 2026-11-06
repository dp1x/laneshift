# External model artifacts — Kev-0.8B LANESHIFT fine-tune

These files are release assets, not ordinary Git files. They are deliberately
ignored by `.gitignore` (`models/`, `*.onnx`, `*.data`, `*.safetensors`, `*.pt`).

## Fine-tuned release: `kev-0.8b-laneshift-v2-subset-e1`

Actual training: 4,096 rows, one epoch, 512 steps. Checkpoint directory:
`LANESHIFT/kev-deliverables-20261105/run` (local only).

| Asset | Size | SHA-256 |
|---|---:|---|
| `adapter_model.safetensors` | 43.3 MB | `142e198a66fde6204d85bcd42be9f8170cbcf4741fcc5a739184b42004a90ba8` |
| `head.pt` | 2.1 MB | `7ed5d33839f8e1aedc3322a5c186fcd4c4d3dbba7ac30d992828d9e8cff6bc38` |
| `kev07b_fp16_L256.onnx` | 7.0 MB | `b952923cf621d1b3b57d058ea6888b5530c23d0c471b0fe6ccced26085e13a28` |
| `kev07b_fp16.data` (4 release parts) | 1.507 GB | `bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940` |
| `tokenizer.json` | 20.0 MB | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` |

## Original release: `jaredpalmer/kev-0.8b@v1.0`

| Asset | Size | SHA-256 |
|---|---:|---|
| `kev07b_fp16_L256.onnx` | 7.0 MB | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| `kev07b_fp16.data` (4 release parts) | 1.437 GB | `3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e` |
| `adapter_model.safetensors` | 43.3 MB | `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` |
| `head.pt` | 2.1 MB | `f400bd12d3a9d749e32423c9a5953f32e195148ad0835580729c1a047b37864f` |

The fine graph was exported with a **new** external blob. `export_reuse.py`
must not be used for a fine-tuned checkpoint: it reuses the original blob and
therefore serves original weights.

## Published release

https://github.com/dp1x/laneshift/releases/tag/kev-0.8b-laneshift-v2-subset-e1

The two large external-data blobs were split into four 400 MB release parts to
avoid GitHub's single-upload connection limit. Reassemble them next to the
corresponding `.onnx` graph:

```bash
# fine model
cat laneshift-kev-v2-subset-e1-L256.data.part-* > kev07b_fp16.data
sha256sum kev07b_fp16.data
# expect bccebbaab31de577b9d831ec24026d195a9b93fc63124f747f4921a62a0d3940

# original model
cat kev-original-L256.data.part-* > kev07b_fp16.data
sha256sum kev07b_fp16.data
# expect 3b03a84c1c2c6b1d3a99dd469ab12f0ebb48b22680d594a4b3838253ea54e07e
```

Release assets: 13 files — fine adapter, head, ONNX graph, four data parts and
tokenizer; original ONNX graph and four data parts. The adapter/head are the
actual trained weights; the ONNX graph was exported from them with a new blob.

Do not commit these blobs to Git.
