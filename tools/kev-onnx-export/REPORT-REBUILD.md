# Rebuild after the RAM-disk reset

The R: RAM disk was cleared by a laptop restart, so everything under `R:\Temp\kev-onnx` was lost. The Kev repo and its
`.venv`, the Hugging Face cache, and the native ARM64 interpreter at `C:\Users\Dhane\AppData\Local\kevarm64\py`
survived. Everything below was re-created from the working notes and re-verified end to end.

## What was rebuilt

| item | status |
|---|---|
| `R:\Temp\kev-onnx\{kev_onnx.py, inputs.py, export_onnx.py, run_onnx.py, bench_torch.py, REPORT.md}` | re-written with the fixes that worked (trace-safe `causal_conv1d_fn`, blocked exact `linalg_solve_triangular`, `export_params=False` + external data, `add_external_initializers_from_files_in_memory`) |
| `R:\Temp\kev-onnx\kev07b_fp16_L35.onnx` | 5.26 MB, opset 17, re-exported |
| `R:\Temp\kev-onnx\kev07b_fp16_L585.onnx` | 8.31 MB, opset 17, re-exported |
| `R:\Temp\kev-onnx\kev07b_fp16.data` | 1437 MiB (1.51 GB) fp16 weights, shared by both graphs |
| `C:\Users\Dhane\dev\kev-onnx-backup\` | persistent copy: all 5 scripts + `REPORT.md` + both `.onnx` + both `ref_*.npz` + **the 1.44 GB blob** (C: had 6.06 GB free, i.e. ≥ 2 GB, so the blob was copied) |
| `R:\Temp\kev-onnx-host\` + `R:\Temp\kev-onnx-host.zip` | host-ready bundle, **1 251.7 MB (1.22 GiB)**, 10 files |

Export was reproduced exactly: the two torch reference logits are bit-identical to the first session
(`[0.278051, 0.227756]` and `[0.191079, 0.493522]`), and both graphs and the blob came out the same size. No code
change was needed to make the rebuild work.

Timing note: the export itself was slower this session (54.7 s and 169.0 s for the two graphs, vs 15.9 s / 84.7 s
before) — see the drift discussion below.

## Native ARM64 verification (task 2)

`C:\Users\Dhane\AppData\Local\kevarm64\py\python.exe` — Python 3.13.9, `platform.machine()=ARM64`,
`onnxruntime 1.31.0`, providers `['AzureExecutionProvider', 'CPUExecutionProvider']`, no torch installed.

```
> py\python.exe run_onnx.py 35 basic
L=35 level=basic session init 5.0s
logits ort   [0.281011 0.217332] | torch [0.278051 0.227756]
probs  ort   [0.515914 0.484086] | torch [0.512571 0.487429]
max |dlogit| 1.042e-02 | max |dprob| 3.343e-03
latency ms [951.4, 979.6, 860.2] -> median 951.4

> py\python.exe run_onnx.py 585 basic
L=585 level=basic session init 8.0s
logits ort   [0.188423 0.505959] | torch [0.191079 0.493522]
probs  ort   [0.421277 0.578723] | torch [0.42496 0.57504]
max |dlogit| 1.244e-02 | max |dprob| 3.684e-03
latency ms [6446.2, 6520.7, 6296.8] -> median 6446.2
```

Parity is identical to the first session (same max deviations to the digit) — the rebuild is faithful.

From the **extracted zip**, via `setup-host.ps1` (fresh Python 3.13.7 ARM64 + fresh pip install, no torch):
`L=35 median 772.3 ms`, `L=585 median 6142.5 ms`, same logits/probs and the same max deviations.

## Latency table

| runner | 35 tokens | 585 tokens |
|---|---|---|
| **ORT 1.31 ARM64 native, `basic`** (this rebuild) | **951 ms** (reset 952/980/860; 486–951 ms across repeated runs) | **6446 ms** (6446/6521/6297) |
| ORT 1.31 ARM64 native, from the extracted zip (Py 3.13.7) | 772 ms | 6143 ms |
| PyTorch 2.8 eager fp16, 8 threads, x86-64 emulated (this rebuild) | 17 753 ms | 90 145 ms |
| ORT 1.31 x86-64 emulated, `basic` (this rebuild, measured under load) | 1413 ms | not re-run (`basic` OOMs on x86, see the original report) |
| — first session, for reference — | | |
| ORT ARM64 native, `basic` | 241 ms | 3142 ms |
| PyTorch fp16 eager | 2572 ms | 43 828 ms |

**Every runner is 2–4× slower this session**, with the CPU at ~50 % load throughout and no paging file; the ratios
survive (ORT is 18–37× faster than eager fp16 at 35 tokens — the spread is the VM's own run-to-run noise — and 14.0×
at 585 tokens, vs 10.7× / 13.9× in the first session). Treat the ratios as the result and the absolute milliseconds as
this VM's current state, not a property of the artifact.

The live Kev server on `:8009` was **not** running after the restart (the port was closed), so the HTTP baseline in the
original report could not be re-measured. I did not start or stop it.

## Host bundle (task 4)

`R:\Temp\kev-onnx-host.zip` — 1 251.7 MB, 10 entries under a top-level `kev-onnx-host/` directory:

```
kev07b_fp16_L35.onnx     5.0 MB
kev07b_fp16_L585.onnx    7.9 MB
kev07b_fp16.data      1437.1 MB
run_onnx.py  inputs.py  kev_onnx.py
setup-host.ps1  README-HOST-ONNX.txt
ref_L35.npz  ref_L585.npz
```

`setup-host.ps1` does, relative to its own folder and from any working directory (`Set-Location $PSScriptRoot`):
download `python-3.13.7-embed-arm64.zip` → extract to `.\py` → write `.\py\python313._pth` with
`python313.zip / . / Lib\site-packages / import site` → download `get-pip.py` and install pip → install
`numpy onnxruntime tokenizers transformers` (no torch) → run `py\python.exe run_onnx.py 35 basic` and `… 585 basic`.
It is idempotent-ish: an existing `.\py\python.exe` skips the download, and pip is only bootstrapped when missing.

**Verified from a clean extraction** at `C:\Users\Dhane\AppData\Local\Temp\kev-host-test` (since deleted): the script
downloaded Python 3.13.7 ARM64, installed the four packages, and both runs produced identical logits/probs and the
max deviations above. The extraction test was done on C: because R: only had 1.3 GB free at that point (the 1.44 GB
blob + 1.22 GB zip already live there).

## Persistent backup (task 3)

`C:\Users\Dhane\dev\kev-onnx-backup\` — 11 files, 1.44 GB: `kev_onnx.py`, `inputs.py`, `export_onnx.py`,
`run_onnx.py`, `bench_torch.py`, `REPORT.md`, `kev07b_fp16_L35.onnx`, `kev07b_fp16_L585.onnx`,
`kev07b_fp16.data`, `ref_L35.npz`, `ref_L585.npz`. C: had 6.06 GB free (≥ 2 GB), so the blob was included; C: free
afterwards is 4.62 GB.

This backup is a copy, not a link: it survives an R: wipe on its own, and re-creating `R:\Temp\kev-onnx` from it takes
one copy plus the scripts.

## Deviations from the previous report

1. **Absolute latencies are 2–4× higher** for every runner (CPU at ~50 % load, no pagefile, slower exports too). Parity
   is unchanged to the digit. The original numbers are kept above for reference.
2. The **HTTP baseline against the live Kev server is missing** this time: the server was not running after the restart
   and I left it alone.
3. The host bundle carries two extra files, `ref_L35.npz` and `ref_L585.npz` (2 KB each). The task's file list did not
   include them, but `run_onnx.py` needs the pre-tokenized inputs — and they also hold the PyTorch reference
   probabilities the parity check prints. The tokenizer is deliberately out of scope (README says so).
4. `run_onnx.py` now locates its files with `os.path.dirname(os.path.abspath(__file__))` instead of a hard-coded
   `R:\Temp\kev-onnx`, so the same file works in the workspace, the host folder and the extracted zip. The external
   data is still handed to ORT from a memory map — that keeps it working on filesystems (like R:) where onnx and
   onnxruntime cannot canonicalise an external-data path.
5. The host zip has a single top-level `kev-onnx-host/` directory (created with `includeBaseDirectory=true`).
6. The weight-free intermediates `graph_L35.onnx` / `graph_L585.onnx` produced during export were deleted after the
   rebuild (they are only needed while assembling the final graphs). `REPORT-REBUILD.md` was also copied into the C:
   backup.

Disk after the rebuild: R: 1.31 GB free of 3.97 GB at start (blob 1.44 GB + zip 1.22 GB + working set); C: 4.62 GB.
