# Running LANESHIFT and Kev on the Windows ARM64 host

This document separates what was verified inside the Hyper-V guest from what
must be run on the physical ThinkPad. The guest cannot execute commands on the
host, and no host filesystem share is mounted.

## Current access boundary

From the guest:

- `\\tsclient\C` does not exist, and there is no Hyper-V shared folder.
- The guest has its own `C:`, `D:` (Windows install media), and `R:` RAM disk.
- Hyper-V integration services run, but none of them expose host execution to
  the guest.
- No command in this repository's verification was run on the physical host.
- Native host inference performance is **not measured**.
- A native ARM64 Python test environment was installed **inside the guest** to
  separate emulation cost from the host question. That measurement is real, but
  it is a virtualized guest measurement, not a host measurement.

The source changes, launcher, tests, and VM measurements below are real. The
host setup and benchmark commands are the pending part.

## Verified guest environment

| Item | Value |
| --- | --- |
| OS | Windows 10 IoT Enterprise LTSC 2024 Evaluation, build 26100, ARM64 VM |
| CPU seen by guest | Snapdragon X Elite X1E78100, 8 vCPUs |
| RAM | 16 GB |
| Python (default) | 3.11.15, **AMD64 under ARM64 emulation** (`PROCESSOR_IDENTIFIER` reports ARMv8/Qualcomm) |
| ONNX Runtime (default) | 1.31.1 x64, providers `AzureExecutionProvider`, `CPUExecutionProvider` |
| Native ARM64 test runtime | Python 3.13.9 ARM64 + ONNX Runtime 1.31.1 ARM64, installed inside the guest only for measurement |
| Godot | 4.7.2 stable, native ARM64 executable |
| Model | `C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx` (7,012,901 bytes) |
| External weights | `kev07b_fp16.data` (1,506,886,400 bytes) |
| Tokenizer | `tokenizer.json` (12,807,196 bytes) |
| Model SHA-256 | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |

The important mismatch is that Godot is native ARM64 while Python and ONNX
Runtime are x64 running under Windows Prism emulation. Native ARM64 Python and
a native `onnxruntime` wheel are the first host-side change to make.

## Guest measurements

Isolated steady-state inference on one fresh session, 256-token graph, scene
prompt, four-vehicle state:

| Intra-op threads | Mean | p50 | p95 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4051 ms | 3930 ms | 5302 ms | 5875 ms |
| 4 | 3660 ms | 3877 ms | 3946 ms | 4026 ms |
| 2 | 4842 ms | 4770 ms | 5217 ms | 5592 ms |

Session load took 5.9-12.4 s, tokenizer load about 1 s, and the first warm-up
about 1.9-3.3 s. Token encoding is about 2 ms; model execution dominates.

### Native ARM64 inside the same guest

To separate the emulation penalty from the host question, a native Windows
ARM64 Python 3.13.9 and ARM64 ONNX Runtime 1.31.1 environment was installed on
the guest's RAM disk and benchmarked on the same model and state:

| Intra-op threads | Mean | p50 | p95 | Session load |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3206 ms | 3208 ms | 3343 ms | 4.3 s |
| 8 | 3351 ms | 3383 ms | 3499 ms | 7.0 s |
| 2 | 4887 ms | 4642 ms | 6120 ms | 5.8 s |

Four threads beat eight, and the CPU memory arena did not help
(3307 ms mean, p95 3618 ms). The isolated native result is about 12% faster
than the isolated x64-emulated result (3660 ms mean, p95 3946 ms), with
tighter variance and a faster session load. In a 15-second live episode with
native ARM64, the controller measured 2602 ms mean model latency (p95 2635 ms)
and five applied `STAY` decisions, with no timeouts or errors.

The honest conclusion is that **native ARM64 alone is a modest win for this
0.8B fp16 graph, not a step-change.** The model is memory-bandwidth bound, and
the VM exposes only the CPU: no NPU or Adreno device is visible to the guest.
The physical host has more cores, no virtualization layer, and full memory
bandwidth, so it still needs to be measured. Do not promise a 2x speedup
before that measurement.

### Visible-window contention in the guest

The guest's visible window uses the **Microsoft Basic Render Driver** (software
D3D12), which competes with the model for CPU. In visible native ARM64 runs,
model latency ranged from 4.4 s to 11.3 s, while the same model headless was
2.6 s. With the launcher's 5 s observation-age bound, some answers were dropped
as stale. A 30 s visible run still advanced 1801 physics ticks and applied three
`STAY` decisions; the HUD showed `DEC`, `OK`, `STALE`, `FAIL`, and `TO` so the
drops were visible. This proves the non-blocking loop and explicit drop
handling. It does not prove that Kev controls well on the guest's software
renderer.

The physical host has an Adreno GPU and 12 CPU cores, so the contention should
be lower. Measure with `PLAY-KEV.bat -Windowed` first. If the host still drops
answers as stale, reduce the render rate or inference threads, or raise
`-MaxDecisionAge` **explicitly** and record the age tradeoff. Do not hide the
drop by applying an arbitrarily old action.

In live play with the x64-emulated Python and four threads, the controller
reported:

| Run | Decisions | Model latency mean | p95 | Timeouts | Model errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 episode x 10 s | 3 | 2302-2453 ms | - | 0 | 0 |
| 3 episodes x 10 s | 6 | 1971 ms | 2220 ms | 0 | 0 |

The game ran 600 physics ticks in 10 s and 1200 ticks in 20 s in headless and
visible runs. It did not pause while inference ran. The visible run rendered at
the normal frame rate and the in-game AI panel updated while the model thought.

The earlier 7-18 s latencies in `logs\kev-controller.log` were caused by
overlapping inference calls: a timed-out `asyncio.to_thread` kept running while
the next decision started. The controller now has one inference worker and
refuses to start a second decision while the previous native call is still
running. The 2-3 s figures above are from the serialized path.

## Host setup

### 1. Install native ARM64 tools

- Python 3.12 or 3.13 for Windows ARM64 from python.org. Choose the **ARM64**
  installer, not the x64 installer. Confirm:

  ```powershell
  python -c "import platform; print(platform.machine())"
  ```

  It must print `ARM64`.

- Godot 4.7.2 Stable for Windows ARM64. Put it on `PATH` or set `GODOT` to the
  executable path.

### 2. Create the host virtual environment

From the project root on the host:

```powershell
SETUP-HOST.bat
```

The script finds an ARM64 interpreter, creates `.venv-host`, installs
`python\requirements.txt`, and runs the environment check. If it cannot find
native ARM64 Python it stops with instructions instead of silently building an
emulated environment.

For a deliberate emulated test only:

```powershell
SETUP-HOST.bat -AllowEmulated -Python python
```

### 3. Place the model

Use one of these locations, or set the environment variables explicitly. The
transfer-package layout is found automatically: the extracted project and the
model folder are siblings.

```text
<transfer>\LANESHIFT2D-Kev-HOST\      (extracted project)
<transfer>\LANESHIFT2D-Kev-model\
  kev07b_fp16_L256.onnx
  kev07b_fp16.data
  tokenizer.json

<project>\models\kev07b_fp16_L256.onnx
<project>\models\kev07b_fp16.data
<project>\models\tokenizer.json

C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx
C:\LANESHIFT2D-Kev-model\kev07b_fp16.data
C:\LANESHIFT2D-Kev-model\tokenizer.json
```

Alternatively:

```powershell
$env:LANESHIFT_ONNX_MODEL = "D:\models\kev07b_fp16_L256.onnx"
$env:LANESHIFT_ONNX_TOKENIZER = "D:\models\tokenizer.json"
```

The launcher also accepts `-Model` and `-Tokenizer`. The external `.data` file
must exist next to the graph unless `--onnx-data` is supplied.

### 4. Check the environment

```powershell
HOST-CHECK.bat
```

This reports the OS architecture, Python architecture and emulation status,
ONNX Runtime version and providers, model/tokenizer sizes, and whether Godot is
on `PATH`. Use `HOST-CHECK.bat --strict` to fail on a missing requirement.

### 5. Benchmark before playing

```powershell
BENCH-KEV.bat --threads 2,4,8 --runs 15
```

The tool reports load time, warm-up, encode time, and median/p95/max total
latency. It writes JSON with `--json-out`. Run it with the game closed first,
then repeat with the game open to measure contention.

Current VM reference (do not treat as a host prediction): 4 threads is the
fastest of the tested settings and leaves half the vCPUs for Godot. On the host,
test 4, 6, and 8 threads.

## One-click host run

```powershell
PLAY-KEV.bat
```

The launcher:

- prefers `.venv-host\Scripts\python.exe`;
- checks the controller dependencies and prints the Python/ONNX Runtime
  architecture and providers;
- finds the model, external data, and tokenizer;
- starts exactly one controller process on `127.0.0.1:8766`;
- waits for the controller's `listening at` line, which is printed after the
  model loads and the warm-up finishes;
- starts LANESHIFT in live mode with Kev in control;
- writes `logs\kev-controller.log`, `logs\kev-controller.err.log`, and, for
  finite runs, `logs\kev-controller-metrics.json`;
- stops the controller when the game window closes.

Other entry points:

| File | Purpose |
| --- | --- |
| `PLAY-KEV.bat` | continuous live Kev, one inference in flight |
| `PLAY-KEV-EVAL.bat` | strict turn-based evaluation, simulation waits |
| `PLAY-KEV-WATCH.bat` | legacy 10 Hz watch mode with the rule safety heartbeat |
| `SETUP-HOST.bat` | create the ARM64 virtual environment |
| `HOST-CHECK.bat` | diagnose the Python/ONNX/Godot/model environment |
| `BENCH-KEV.bat` | measure load, encode, and inference latency |
| `RECORD-HUMAN.bat` | record keyboard play for the human dataset |

Useful live-run parameters:

```powershell
PLAY-KEV.bat -Windowed
PLAY-KEV.bat -Threads 6
PLAY-KEV.bat -Episodes 1 -MaxEpisodes 5 -Seeds 1,2,3,4,5 -EpisodeDuration 30
PLAY-KEV.bat -MaxDecisionAge 2 -DecisionInterval 2
```

In live mode the launcher cycles through the seed list, so a crash does not
repeat the identical scenario. `-MaxEpisodes` gives a finite total; without it,
the game keeps cycling until the window closes. `-Episodes` keeps the existing
per-seed meaning in eval and watch modes.

## Acceleration options on Snapdragon X Elite

These are the options and their actual status. Do not install a package just
because the hardware advertises an accelerator.

| Path | Status | Notes |
| --- | --- | --- |
| Native ARM64 CPU | **Confirmed and measured in the guest** | PyPI `onnxruntime` 1.31.1 publishes `cp311`-`cp314` `win_arm64` wheels. Native ARM64 in the guest: 3.2 s isolated, 2.6 s mean in live play. About 12% faster than x64 emulation in the isolated comparison, with tighter variance. Host numbers still pending. |
| Qualcomm NPU via QNN | **Possible, not verified for this graph** | `onnxruntime-qnn` 2.7.0 publishes `win_arm64` wheels. QNN HTP favors quantized, static-shape graphs with primitive ops. This model is a custom 0.8B fp16 transformer with external weights; fused attention/normalization and fp16 coverage may fail or fall back. Test on the host with `--onnx-provider qnn` and inspect the session log. |
| Adreno GPU via DirectML | **Not a Python path on ARM64 at the checked version** | `onnxruntime-directml` 1.24.4 publishes only `win_amd64` wheels. Do not force the x64 wheel into an ARM64 environment. A native ARM64 DirectML path would need a different runtime or a newer release that actually ships `win_arm64`. |
| ONNX Runtime GenAI | **Not applicable to this graph** | `onnxruntime-genai` has `win_arm64` wheels, but it targets its own generative-model format and cache management. LANESHIFT calls a fixed-shape decision graph directly. |
| Azure execution provider | **Not a local accelerator** | It appears in the x64 ONNX Runtime build but is for Azure endpoints, not the local CPU/GPU/NPU. |
| INT8 quantization | **Not done** | It could reduce memory bandwidth and may be required for a useful QNN path, but it changes the model's outputs. It needs calibration, evaluation on held-out seeds, and a separate checkpoint label. Do not substitute it for the original model in a comparison. |

The safe default is native ARM64 CPU with 4-8 intra-op threads. Investigate QNN
only after the CPU baseline is measured on the host.

## What to report from the host

Please capture these values so the VM and host can be compared honestly:

1. `HOST-CHECK.bat` output.
2. `BENCH-KEV.bat --threads 4,6,8 --runs 15 --json-out bench-host.json` output,
   with the game closed.
3. A 60-second live run: `PLAY-KEV.bat -Windowed -Episodes 1 -MaxEpisodes 1 -Seeds 5 -EpisodeDuration 60`.
4. `logs\kev-controller-metrics.json` and the latest telemetry session under
   `.portable-dev\recordings\`.
5. Whether the in-game AI panel shows `THINKING`, `ACTION`, `STALE`, `TIMEOUT`,
   or `ERROR`, and whether the road animation stays smooth while it changes.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Python ... cannot import onnxruntime` | Run `SETUP-HOST.bat`; confirm `python -c "import platform; print(platform.machine())"` says ARM64. |
| `ONNX model not found` | Set `LANESHIFT_ONNX_MODEL` or place the files under `models\` or `C:\LANESHIFT2D-Kev-model`. |
| `external weights ... not found` | Keep `kev07b_fp16.data` beside the `.onnx` file. |
| Controller exits before ready | Read `logs\kev-controller.err.log`; the model load error is printed there. |
| `THINKING` never becomes `ACTION` | Read `logs\kev-controller.log` and `logs\kev-controller-metrics.json`; check `model_errors`, `decision_timeouts`, `inference_busy`, and `stale_decisions`. |
| `STALE` on every answer | Inference is slower than `-MaxDecisionAge`; measure with `BENCH-KEV.bat`, reduce the render rate (`-MaxFps 30`) or inference threads, and use native ARM64 Python. If you raise `-MaxDecisionAge`, record the age tradeoff. The HUD and telemetry show the drop. |
| Game window is larger than the display | The default is fullscreen; pass `-Windowed` to maximize instead. The game prints `LANESHIFT_WINDOW window=... viewport=... screen=... scale=...` at startup. |
| Two controllers / port in use | Close the previous game window; the launcher stops its own controller, but a manually started controller can still own port 8766. |
