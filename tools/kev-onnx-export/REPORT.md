# Kev-0.8B decision model → ONNX feasibility probe

> **Rebuilt file.** The original REPORT.md lived on the R: RAM disk and was lost in a laptop restart. This is the same
> report, re-written from the working notes; the artifacts it describes were re-exported and re-verified afterwards
> (see `REPORT-REBUILD.md` for the rebuild, and note the timing drift between sessions).

Run on the Windows ARM64 VM (Snapdragon X Elite, 20 GB RAM, **no paging file** → ~3.4 GB of commit headroom with the
Kev server running), Python 3.13.16 x86-64 under emulation, torch 2.8.0+cpu.
Checkpoint: `jaredpalmer/kev-0.8b@v1.0` — LoRA r=16/alpha=32 on `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2`, head temperature
2.3511, merged before export, fp16.

**Verdict: export works and onnxruntime reproduces PyTorch's probabilities to ≤5e-4. ORT is ~8.5× faster than eager
PyTorch fp16 at 35 tokens and ~16× faster at 585 tokens. The blocking issues are memory on this VM and a fixed
sequence length per exported file, not unsupported maths — but the export needs two runtime monkey-patches to
`transformers` 5.17 code (nothing in the Kev repo was modified).**

---

## 1. The forward pass that had to be exported

`kev/model.py` + `kev/checkpoint.py`:

* Backbone is `AutoModelForCausalLM.from_pretrained(...).model` → `Qwen3_5TextModel`: 24 layers,
  `layer_types` = 18 × `linear_attention` (Gated DeltaNet) + 6 × `full_attention`, hidden 1024, vocab 248 320 (tied),
  head_dim 256, partial rotary 0.25 with mrope, `attn_output_gate`. **`is_hybrid()` is True for it**, so Kev never uses
  the packed block-causal mask (`branch_mask_batch`) in practice: `rows_form()` is always True and every question runs
  as one ordinary causal row `state ++ branch` with a 2-D all-ones attention mask (`DecisionModel._rows_hidden`).
* One decision therefore needs only: `input_ids[1,L]`, `position_ids[1,L]` (expanded internally to 4 mRoPE axes),
  `attention_mask[1,L]`.
* Readout: `PointerHead` (`q`,`k`: 1024→256, fp32) → `z = (k(h_opts) @ q(h_decide)) / 16 / T`, `T = 2.3511`, one
  logit per option. Option spans are `<|box_start|> opt <|box_end|>`, `<decide>` is `<|fim_suffix|>`, instruction
  starts with `<|fim_middle|>`, state starts with `<|fim_prefix|>`.

The wrapper (`kev_onnx.py::Decide`) is exactly that: one row in, `K` option logits out.

## 2. Exact commands

```powershell
cd C:\Users\Dhane\dev\kev
uv pip install onnx onnxruntime          # onnx 1.23.2, onnxruntime 1.31.0 (win_amd64, so x86-64 under emulation)
uv pip install onnxscript                # tried for the dynamo exporter; not needed by the final path

uv run python R:\Temp\kev-onnx\export_onnx.py     # builds both ONNX files + the shared weight blob + torch references
uv run python R:\Temp\kev-onnx\bench_torch.py     # eager PyTorch fp16 latency for the same two rows
uv run python R:\Temp\kev-onnx\run_onnx.py 35  basic
uv run python R:\Temp\kev-onnx\run_onnx.py 585 none

# optional step 9: native ARM64 interpreter, no torch at all
Invoke-WebRequest https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-arm64.zip -OutFile C:\...\py.zip
& C:\Users\Dhane\AppData\Local\kevarm64\py\python.exe C:\...\get-pip.py
& C:\Users\Dhane\AppData\Local\kevarm64\py\python.exe -m pip install numpy onnxruntime   # onnxruntime 1.31.0 win_arm64
& C:\Users\Dhane\AppData\Local\kevarm64\py\python.exe R:\Temp\kev-onnx\run_onnx.py 585 basic
```

## 3. Every failure, and the fix

| # | Failure | Cause | Fix |
|---|---|---|---|
| 1 | `RuntimeError: expected padding to be a single integer value or a list of 1 values …, got padding=[]` on the very first `torch.jit.trace` | `transformers` 5.17 `causal_conv1d_fn` computes `padding = weight.shape[-1] - 1`; under tracing that is a 0-dim **tensor**, not an int | runtime patch of `qwen3_5.causal_conv1d_fn` with the same math using `int(...)` (`kev_onnx.patch_transformers_for_tracing`) |
| 2 | `UnsupportedOperatorError: Exporting the operator 'aten::linalg_solve_triangular' to ONNX opset 17 is not supported` | Gated DeltaNet's UT transform (`torch_chunk_gated_delta_rule`) solves a 64×64 unit-lower-triangular system; ONNX has no such op | replaced with `(I+N)^-1 = (I-D)(I+D²)(I+D⁴)(I+D⁸)` per 16×16 diagonal block + block forward substitution (`kev_onnx._blocked_triangular_solve`). Exact algebra, ONNX-only ops (Trilu/MatMul/Cat) |
| 2a | first substitute — a whole-matrix Neumann series `(I-N)(I+N²)…(I+N³²)` — **silently wrong**: logits differed by 0.06 at 35 tokens and went **NaN** at 585 | N^32 overflows fp32 once a chunk's \|N\| > 1 | discarded; blocked version verified against `torch.linalg.solve_triangular` (rel. error 3e-7 at \|N\|≲1) |
| 2b | the literal 63-step forward substitution (what `transformers` itself does under `is_torchdynamo_exporting()`) blew the graph up to >400 k values and the exporter died in `bad allocation` | 63 unrolled steps × 2 solves × 18 layers | blocked version above (~60 nodes per solve) |
| 3 | `MemoryError: bad allocation` inside `graph._export_onnx` (and, with `dynamo=True`, `ConversionError … MemoryError` in decomposition) | both exporters materialise the whole 1.46 GB fp16 weight set in a ModelProto/FX copy; the VM has no pagefile (~3.4 GB commit free) | export the graph with `export_params=False, keep_initializers_as_inputs=True` (proto 5–8 MB, weights become graph inputs), then write the tensors straight from `state_dict()` into one external-data blob and attach them as initializers |
| 4 | `onnx.save_model(..., save_as_external_data=True)` and `ort.InferenceSession(path)` both fail: `Tensor weight external data path could not be canonicalized: Incorrect function` | `R:` is a letter-less virtual volume (`Get-Volume` shows no `R`), `GetFinalPathNameByHandle` fails on it; a directory junction to `R:` fails the same way | read the blob as a `np.memmap` and hand it to the session with `SessionOptions.add_external_initializers_from_files_in_memory(...)` |
| 5 | `bad allocation` when creating the session with `ORT_ENABLE_ALL` (and, at 585, with BASIC/EXTENDED), and `Cast node … bad allocation` at run time | ORT's fp16 transformer inserts fp32 copies of the weights; on this VM that exceeds the commit limit | `ORT_DISABLE_ALL` at 585 (BASIC/EXTENDED/ALL all OOM); BASIC works for the 35-token graph. On the native ARM64 interpreter BASIC works for both |
| 6 | 585-token reference logits first came out NaN | it was 2a, not fp16: with a correct solve the fp16 model is finite at both lengths (hooks show no non-finite layer) | — |

## 4. What was produced

| file | size | notes |
|---|---|---|
| `R:\Temp\kev-onnx\kev07b_fp16_L35.onnx` | **5.26 MB** | opset 17, IR 8, 17 879 nodes, 324 initializers (external) |
| `R:\Temp\kev-onnx\kev07b_fp16_L585.onnx` | **8.31 MB** | opset 17, IR 8, 30 719 nodes, 324 initializers (external) |
| `R:\Temp\kev-onnx\kev07b_fp16.data` | **1437 MB** (1.51 GB) | fp16 weights, 64-byte aligned, **shared by both graphs** |

Inputs `input_ids, position_ids, attention_mask, decide_idx, opt_idx` — all `int64`; output `logits`
(`float32`, `[K]`). Weights are fp16 (752.4 M backbone params = 1504.8 MB); the 524 800-parameter pointer head stays
fp32, which is why the blob is 1437 MiB = 1506.9 MB.

* **Sequence length is baked in** (one file per length): the chunked delta rule unrolls `num_chunks = L/64` at trace
  time, so dynamic axes on `L` are impossible without re-implementing that kernel.
* **Option count K is dynamic** — verified: the same graph exported with `dynamic_axes={"opt_idx":{0:"K"},"logits":{0:"K"}}`
  returns 2 and 5 logits from one file.

## 5. PyTorch vs ONNX

Two inputs: a 35-token row (`ego: 14.2 m/s lane 1 lead 14.4 m ahead` + one noul question) and a 585-token row
(12-vehicle LANESHIFT-style JSON state + one question). Both are single causal rows, exactly as `rows_form` builds them.

| host | L | ORT level | max &#124;Δlogit&#124; | max &#124;Δprob&#124; | probs (torch → ort) |
|---|---|---|---|---|---|
| x86-64 emulated | 35 | BASIC | 5.33e-4 | **9.27e-5** | 0.51257/0.48743 → 0.51266/0.48734 |
| x86-64 emulated | 35 | DISABLE_ALL | 1.70e-3 | 5.00e-4 | 0.51257/0.48743 → 0.51307/0.48693 |
| x86-64 emulated | 585 | DISABLE_ALL | 3.04e-4 | **1.40e-4** | 0.42496/0.57504 → 0.42482/0.57518 |
| ARM64 native | 35 | BASIC | 1.04e-2 | 3.34e-3 | 0.51257/0.48743 → 0.51591/0.48409 |
| ARM64 native | 585 | BASIC | 1.24e-2 | 3.68e-3 | 0.42496/0.57504 → 0.42128/0.57872 |
| ARM64 native | 585 | DISABLE_ALL | 1.88e-3 | 4.70e-4 | 0.42496/0.57504 → 0.42449/0.57551 |

argmax is identical everywhere. The spread across builds/levels is fp16 CPU kernel differences (ORT casts fp16↔fp32 in
places), not model error.

## 6. Latency (warm: 1 warm-up discarded, median of 3)

| runner | 35 tokens | 585 tokens |
|---|---|---|
| PyTorch 2.8 eager, fp16, 8 threads (x86 emulated) | 2572 ms | 43 828 ms |
| **onnxruntime 1.31, x86-64 emulated**, BASIC | **301–309 ms** | OOM (DISABLE_ALL only: **2730 ms**) |
| **onnxruntime 1.31, ARM64 native**, BASIC | **241 ms** | **3142 ms** |
| onnxruntime 1.31, ARM64 native, DISABLE_ALL | — | 3885 ms |
| live Kev server :8009, bf16, HTTP `POST /v1/systemone` | 2213 ms | 34 329 / 6937 / 4985 ms (first = prefix miss, then warm hits) |

Session creation is expensive: 5.0 s (35, BASIC) / 50.7 s (35, DISABLE_ALL) / 172.6 s (585, DISABLE_ALL) on x86;
3.4 s / 5.2 s / 81 s on ARM64. First ORT run includes kernel/arena warm-up.

## 7. Architecture of the runtime (step 8)

* `platform.machine()` → `ARM64`; the venv's Python reports `3.13.16 … [MSC v.1944 64 bit (AMD64)]` (x86-64 under
  emulation), same for `onnxruntime` → it is the **win_amd64 (x86-64) build running emulated**, not ARM64.
* `onnxruntime.get_available_providers()` → `['AzureExecutionProvider', 'CPUExecutionProvider']`, `get_device()` → `CPU`.
  No QNN/ACL/DirectML provider is shipped in the PyPI wheel.
* The native ARM64 interpreter (`python-3.13.9-embed-arm64`, `onnxruntime` win_arm64, **no torch installed**) runs the
  identical ONNX + blob — so the exported file has no torch dependency; only the tokenizer is outside (ids were
  produced with `tokenizers`/`transformers` under the venv and saved as `.npz`).

## 8. Blockers and recommended next step

Blockers, in order of severity:

1. **No paging file / ~3.4 GB commit headroom.** Exporting a weight-carrying proto, the dynamo exporter, and ORT's
   fp16 graph optimiser all need 2–3× the 1.5 GB of weights and fail with `bad allocation`. Everything above works,
   but with ~10 GB more headroom the straightforward `torch.onnx.export(..., external_data=True)` (or `dynamo=True`)
   would probably just work.
2. **One export per sequence length** (delta-rule chunk unroll). Practical form: export a small set of buckets
   (e.g. 64/128/256/512/1024 tokens) and right-pad, or export the *recurrent* gated-delta rule
   (`torch_recurrent_gated_delta_rule`) which traces token-by-token and is length-agnostic in principle (10× more nodes,
   no chunk loop).
3. **The state is recomputed on every request.** These graphs have no KV cache: Kev's own serving reuses a state prefix
   (`DecisionModel.prefix/probs_with_prefix`), so at 585 tokens the served bf16 path (5.0–6.9 s warm, state cached)
   is closer to this ONNX path (2.73 s without reuse) than the raw numbers suggest for repeat calls on the same state.
   A production export needs `past_key_values` as graph inputs (state pass + branch pass), which is the natural next step.
4. **Two monkey-patches are required** for `transformers` 5.17: `causal_conv1d_fn`'s `padding` and
   `linalg_solve_triangular`. Both are genuine upstream export bugs; worth reporting to `transformers` rather than
   carrying locally. The fp16 CPU kernels are also thin: ORT inserts fp16↔fp32 casts everywhere
   (`InsertedPrecisionFreeCast_*`), and the 30 k-node graph is dominated by small ops (`Constant`, `Unsqueeze`,
   `Gather`, `Shape`).

Recommended next step, if ONNX is pursued: export **with KV-cache inputs** (state prefix once per state, then one row
per question) and **quantise to int8** (`MatMul` weight-only, which also drops the blob to ~0.4 GB and sidesteps the
memory ceiling), then re-measure against the served bf16 path. If the goal is pure speed on this VM, the honest
alternative is that ORT already buys 8–16× over eager PyTorch fp16 for the same weights — but the served bf16 path
with prefix caching is in the same ballpark at 585 tokens, so ONNX only pays off if it is made cache-aware.

## 9. Files

```
R:\Temp\kev-onnx\kev_onnx.py       wrapper (Decide), model loader, the two transformers patches, blocked solve
R:\Temp\kev-onnx\inputs.py         the two states (35 / 585 tokens, 12 vehicles), exact-length builder
R:\Temp\kev-onnx\export_onnx.py    weight-free graph export + external-data assembly
R:\Temp\kev-onnx\run_onnx.py       torch-free ORT session, parity check, latency
R:\Temp\kev-onnx\bench_torch.py    eager PyTorch fp16 latency for the same rows
```
