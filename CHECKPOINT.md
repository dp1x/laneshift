# LANESHIFT Checkpoint

## Current state

- Godot 4.7.2 top-down three-lane driving game with a coherent pseudo-perspective road, lane geometry, traffic sprites, crash feedback, corner HUD, fullscreen by default and a presentation-only title state.
- Authoritative simulation with seeded traffic, discrete actions and episode summaries. Action counts now count accepted actions rather than only action changes.
- Telemetry logger with an explicit session lifetime. Recording after a run has already started no longer disables the logger. Recovery files stay under the portable recordings root.
- Python controller with `dummy`, `rule_based`, local ONNX Kev and the HTTP Kev System One adapter. The ONNX path records a configurable model id, label source and SHA-256 of the model file.
- Matched experiment runner for baselines and ONNX. It passes the same seed list, duration, traffic density, simulation speed, timeout and staleness bounds to every policy.
- One-click local Kev play: `PLAY-KEV.bat` starts the ONNX controller and the game, opens the AI status panel and records the session.

## Verification

- `run_tests.bat`: 38 Python tests plus Godot import, simulation, game-flow, network, telemetry, telemetry-flow and options tests pass.
- `tests/telemetry_flow_test.gd` reproduces the real user path: start a run, play before Record, press Record, play, stop, then assert non-zero telemetry and events and a matching manifest.
- Local ONNX Kev: 5 decisions in an 8-second episode, all accepted, `control_source=original_model`, mean model latency about 1.36 s.
- Matched pilot: `rule_based` vs `onnx_kev`, same 5 seeds, 20 s, density 0.7, sim speed 1.0. See `docs/RESULTS.md` for the numbers.

## Remaining work

- No fine-tuned Kev checkpoint exists. The `fine_tuned_model` path is implemented and can be labelled, but it has not been trained.
- The pilot is small. It is a sanity check, not a publishable comparison.
- Calibration has not been measured on held-out human play.
- Human recordings from the phone/ADB pipeline are not part of this repository; the testbed records keyboard play in the simulation.
- The HTTP torch path is still about 15 s per decision on this CPU. The ONNX path is the usable local path.
