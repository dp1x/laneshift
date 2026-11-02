# LANESHIFT Checkpoint

## Live asynchronous mode (2026-11-02)

- `PLAY-KEV.bat` now runs **live mode**: the simulation continues at 60 Hz while Kev thinks in the controller process. One decision request is in flight; other state updates are marked `decision_request=false` and never occupy the decision slot.
- A returned action is applied only if it matches the current episode and request tick and its observation age is within `--max-decision-age-s` (5 simulated seconds by default in the launcher). Late answers are dropped and logged. A request timeout or controller `model_error` clears the slot and the episode continues. Live mode never substitutes a rule action.
- The controller has one inference worker. A timed-out native inference keeps running but the next decision is refused as `inference_busy` rather than overlapped. This removes the 7-18 s latency amplification seen in the previous logs.
- New shortcuts: `PLAY-KEV-WATCH.bat` for the old 10 Hz safety-heartbeat mode, `PLAY-KEV-EVAL.bat` for strict turn-based evaluation, `SETUP-HOST.bat`, `HOST-CHECK.bat`, and `BENCH-KEV.bat`.
- Verified in the guest VM: a visible 20 s episode ran 1200 physics ticks with 3 real Kev decisions and no freeze; a three-episode run made 6 decisions with 0 timeouts, 0 model errors, and 1.97 s mean model latency. Kev chose STAY every time; seeds 6 and 7 crashed at 3.6 s and 5.45 s. This is a model-quality result.
- The previous `inference_error` was a turn-based decision timeout: decision 3 was requested at sim 2.02 s, the controller hit its 20 s decision timeout, and `_fail_turn` ended the episode. The controller log's "opening handshake failed" trace was the launcher's TCP readiness probe, not the model.
- Native Windows ARM64 Python 3.13.9 + ONNX Runtime 1.31.1 was installed inside the guest and measured: 3206 ms mean isolated at 4 threads (p95 3343 ms), versus 3660 ms mean for x64 emulation at the same thread count, about a 12% gain. In a native live episode the controller measured 2602 ms mean (p95 2635 ms), 5 applied STAY decisions, 0 timeouts, 0 model errors. The model is memory-bandwidth bound; native ARM64 alone is not a step-change. The VM exposes no NPU or Adreno device.
- Visible guest runs use Microsoft Basic Render Driver (software D3D12). Model latency ranged 4.4-11.3 s under that contention, and some answers were dropped as stale under the 5 s bound. The HUD exposes `OK`, `STALE`, and `REJ`; a 30 s visible run still completed 1801 physics ticks and applied three STAY decisions. The controller now writes `logs\kev-controller-metrics.json` on client disconnect as well as graceful shutdown.
- Host access is still unavailable from the VM. Native ARM64 host inference and acceleration are not measured. Godot is native ARM64 in the guest as well.

## Previous checkpoint state

These notes describe the pre-live-mode state. Timing claims in this section are
historical; the live section above and `docs/HOST-WINDOWS-ARM64.md` carry the
current measurements.

- Godot 4.7.2 top-down three-lane driving game with a coherent pseudo-perspective road, depth fog, lane geometry, traffic sprites, crash feedback and a typographic corner HUD with no bordered scene panels.
- Title screen is a left-aligned composition over the live road, not a centered dialog. Pause and run-ended states use the same type and one primary action.
- The HUD has a large score/speed pair, small tracked captions, an AI instrument with action probability, action-history dots, latency and local WebSocket state, and a first-run controls line.
- Authoritative simulation with seeded traffic, discrete actions and episode summaries. Action counts count accepted actions rather than only action changes.
- Telemetry logger with an explicit session lifetime. Recording after a run has already started no longer disables the logger. Recovery files stay under the portable recordings root.
- Python controller with `dummy`, `rule_based`, local ONNX Kev and the HTTP Kev System One adapter. The ONNX path records a configurable model id, label source and SHA-256 of the model file.
- The packaged launcher uses `Start-Process` for the exported game so `--ws-port` survives the Windows argument binding, and waits for the controller before launching.
- The ONNX policy falls back to in-memory external weights on RAM disks. This is what makes the packaged agent work when the folder is extracted to `R:`; the first load is slower.
- Watch mode has a rule-based safety heartbeat and low-confidence veto, labelled `controller_fallback`. The experiment runner does not enable it.
- Matched experiment runner for baselines and ONNX. It passes the same seed list, duration, traffic density, simulation speed, timeout and staleness bounds to every policy.
- One-click local Kev play: `PLAY-KEV.bat` starts the ONNX controller and the game, opens the AI instrument and records the session.

## Verification

- `run_tests.bat`: 96 Python tests plus Godot import, simulation, game-flow, network, telemetry, telemetry-flow, turn-based, live-decision, observation, human-schema and options tests pass.
- Live asynchronous mode: a visible 20 s episode ran 1200 physics ticks with 3 real Kev decisions and no freeze; a three-episode headless run made 6 decisions with 0 timeouts, 0 model errors, and 1.97 s mean model latency. The controller reports 0 overlapping inferences. See `docs/LIVE-MODE.md`.
- Observation audit (schema 2): every rendered vehicle is included, off-screen traffic is separately labelled within `[-15, 160]` m, collision dimensions and previous actions are in the snapshot, and the scene prompt is 86-120 tokens versus 174-199 for the truncated lane prompt. See `docs/OBSERVATION-AUDIT.md`.
- `tests/telemetry_flow_test.gd` reproduces the real user path: start a run, play before Record, press Record, play, stop, then assert non-zero telemetry and events and a matching manifest.
- Historical local ONNX Kev measurement: 5 decisions in an 8-second episode, all accepted, `control_source=original_model`, mean model latency about 1.36 s.
- Historical packaged ONNX Kev measurement with the model on `C:\LANESHIFT2D-Kev-model`: normal ONNX Runtime path, 8 decisions in a 15-second episode, roughly 1.9 s per decision on an idle CPU. CPU contention pushed it past 8 s in some runs.
- Watch mode with safety heartbeat: 7 of 8 seeds survived the full 15 seconds; one seed crashed at 4.5 s. Pure Kev still chose `STAY` every time; the lane changes were safety-layer actions labelled `controller_fallback`.
- Turn-based evaluation (`PLAY-KEV-EVAL.bat`, safety disabled): the simulation froze at each decision, Kev received one state, and its exact actions were applied. The first recorded run made 3 decisions — `LEFT`, `RIGHT`, `LEFT` — all `original_model`, with no superseded states and no fallback. It crashed at 2.7 s because the third decision chose LEFT into a 1.1 s TTC gap while the centre lane was clear.
- Controlled 8-scenario probe: the compact prompt chose STAY in all 8; the lane prompt chose LEFT in 7 of 8, including the right-safer and open-road cases. Reversing option order still chose LEFT. The lane prompt changes the bias but zero-shot Kev is not yet state-sensitive. This is a model/data problem, not an action-application bug; fine-tuning is still required.
- Matched pilot: `rule_based` vs `onnx_kev`, same 5 seeds, 20 s, density 0.7, sim speed 1.0. See `docs/RESULTS.md` for the numbers.

## Current autonomous milestone

- Decision-level dataset export and a small supervised baseline are merged on the feature branch.
- Pilot: 1,348 decision rows over 15 rule-based seeds; held-out MLP 0.8844 accuracy, 0.4426 macro-F1, 0.3587 log loss, 0.1763 Brier, 0.0671 ECE. Balanced variant: 0.8728 accuracy, 0.5574 macro-F1.
- The labels are rule-based; the MLP has not driven the car and no fine-tuning has happened.
- Human gate: data source, approved labels, fine-tune target, GPU budget and the final evaluation claim. See `docs/NEXT-GATE.md` and `docs/TRAINING-PILOT.md`.

## Remaining work

- No fine-tuned Kev checkpoint exists. The `fine_tuned_model` path is implemented and can be labelled, but it has not been trained.
- The pilot is small. It is a sanity check, not a publishable comparison.
- Calibration has not been measured on held-out human play.
- Human recordings from the phone/ADB pipeline are not part of this repository; the testbed records keyboard play in the simulation.
- The HTTP torch path is still about 15 s per decision on this CPU. The ONNX path is the usable local path.
- Native Windows ARM64 host inference has not been measured. The guest uses x64 Python and ONNX Runtime under emulation. Use `SETUP-HOST.bat`, `HOST-CHECK.bat`, and `BENCH-KEV.bat` on the host.
- QNN/NPU and DirectML/GPU acceleration are not verified for this graph. Native ARM64 CPU is the confirmed baseline; `onnxruntime-directml` 1.24.4 has no `win_arm64` wheel, and QNN operator coverage for this transformer would need host testing.
- Live-mode action quality has not been evaluated on held-out seeds. The first verified runs chose STAY every time; a completed episode is an integration result, not a driving result.
