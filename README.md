# LANESHIFT

LANESHIFT is a small, original three-lane arcade motorcycle game and a reproducible controller experiment environment. Godot owns all simulation state; Python policies receive structured snapshots over a localhost WebSocket and return discrete actions. The game and telemetry work without Python.

## Requirements

- Godot 4.7.2 Stable
- Python 3.10 or newer
- Python dependency: `websockets` (listed in `python/requirements.txt`)

Install the Python dependency once:

```powershell
python -m pip install -r python/requirements.txt
```

Godot must be available as `godot` on `PATH`, or its executable can be passed to the Python experiment runner with `--godot`.

## Play

Run `run_game.bat` on Windows, or launch the project with `godot --path .`. Select **Start** on the title screen.

- `A` / Left Arrow: move left one lane
- `D` / Right Arrow: move right one lane
- Hold Space / Down Arrow: brake
- `Escape`: pause or resume
- `T`: switch between human and connected Python control while driving

The lane change eases over time; a steering press is applied once rather than repeating while held. Passing traffic awards a small bonus. A collision or configured episode duration ends the run. The result screen offers restart and menu options. Best score is stored in `user://highscore.save`. **Open Logs** on the title screen opens the local run directory.

To use Python control, start the controller first in a separate terminal, then launch the game and press `T` during a run:

```powershell
python python/server.py --policy rule_based
```

Available policies are `dummy` and `rule_based`, plus any model policy named in a model registry (see **Kev Model Control**). The dummy policy deliberately changes lanes so the bridge can be smoke-tested. A model policy must be configured with the checkpoint it really loads; there is no stub that silently returns `STAY` under a model name.

## Architecture

- `scripts/sim/simulation.gd` owns time, player actions, speed, seeded traffic, collisions, scoring, legal actions and snapshots.
- `scripts/main.gd` builds the procedural world, camera, HUD, input, presentation, and episode flow. Both keyboard and external input call the same simulation action method.
- `scripts/telemetry/telemetry_logger.gd` buffers JSONL records and writes batches on a worker thread; episode boundaries and exit drain the queue.
- `scripts/net/net_bridge.gd` is an optional Godot WebSocket client. It validates protocol messages and falls back to human control after disconnect or action timeout.
- `python/server.py` hosts the loopback WebSocket endpoint and keeps policy, protocol, action counts, and decision latency outside Godot. Inference runs asynchronously so a slow model never blocks the socket.
- `python/policies/` contains the scripted policy, the rule-based baseline, and the Kev System One adapter.
- `python/dataset_prep.py` joins event decisions to pre-action observations, separates outcomes from model inputs, and exports Kev-format training records.
- `python/experiment_runner.py` starts one controller and one headless Godot run per policy, reusing scenario seeds and writing a comparison CSV plus per-episode results.

All visuals are made from primitive Godot meshes and materials; there are no downloaded assets. The simulation is kinematic and lane-based rather than a rigid-body motorcycle model.

## Coordinates And Timing

- World `X` is lateral: negative is left, positive is right. Lane centers are `-4`, `0`, and `+4` meters.
- World `Y` is up. World `+Z` is forward. The player advances in `+Z`; traffic also moves in `+Z` at its own speed.
- `player.lane` is the physical lane the motorcycle currently occupies; `player.target_lane` is the lane it is moving toward. The two differ while a transition is in progress, and `lane_change_in_progress` says so explicitly.
- `legal_actions` is the simulation's own answer to "what may be done now": steering is unavailable during a transition, at a road edge, or while the lane-change cooldown is active.
- Positions are meters and velocities are meters per second. HUD speed is kilometers per hour.
- `relative_distance_z = traffic.z - player.z`: positive means ahead. `relative_speed_z = player.speed - traffic.speed`: positive means the player is closing.
- The default physics rate is 60 Hz, telemetry is sampled at approximately 30 Hz, and states are sent for decisions at approximately 10 Hz. These rates are independent settings, not protocol assumptions.
- Each snapshot has simulation time and physics tick plus Unix and monotonic wall-clock timestamps. Simulation time follows physics steps; wall time records observed process timing.

The traffic generator uses its own seeded `RandomNumberGenerator`. A reset with the same seed and configuration reproduces initial vehicles and the subsequent spawn schedule, independent of visual randomness. Player actions are expected to make the resulting trajectories diverge.

## Telemetry

Each game launch writes to a unique directory under Godot's user data location:

```text
user://runs/session_<timestamp>_<suffix>/
  telemetry.jsonl
  events.jsonl
  episodes.jsonl
```

On Windows this is normally `%APPDATA%\Godot\app_userdata\LANESHIFT\runs\`. Use **Open Logs** in the title menu to open the parent folder. Records are one JSON object per line.

`telemetry.jsonl` contains `state_id`, `episode_id`, `scenario_seed`, `sim_time`, `physics_tick`, player position/lane/target lane/transition state/speed/velocity/action, the authoritative `legal_actions`, nearby traffic IDs and their positions/velocities/lanes, relative gaps and speeds, score, distance, survival time, collision and episode status, control source, and wall timestamps. An episode that ends produces exactly one additional terminal snapshot with `snapshot_kind: "terminal"` and a unique state id; periodic sampling stops at that point, so no state is recorded twice for the same tick.

`events.jsonl` records scenario initialization (including configuration and initial traffic), episode reset, start and end, pause and resume, control connection/mode changes, collisions, overtakes, score milestones, external command receipt, validation, application and rejection, and action decisions. An `action_decision` contains requested and applied actions, acceptance, decision sequence, source labels, the controller-observed state ID/tick and exact `state_before` snapshot, plus separate application tick/time metadata and the controller's `decision_metadata`. For external control, `state_before` is the exact snapshot sent to the controller, even when the accepted command arrives several physics ticks later. Human actions use the current pre-action snapshot. Human actions are recorded even without a controller.

`episodes.jsonl` has one summary for each ended episode, with seed, score, distance, survival time, collision and overtake counts, final lane, action changes, and termination reason. Restart, menu exit, quit, and normal time-limit completion are also given termination reasons.

Writes are buffered away from the simulation thread, periodically queued, and drained at episode boundaries and shutdown. A failed write is retried, then preserved in an emergency JSONL file under `user://runs/_recovery/<session>/`; if that also fails the batch is reported and logging is disabled rather than silently discarding data. The write queue is bounded, and exceeding it disables logging with a visible warning instead of growing without limit. Log directories and high scores remain local to the Godot user data directory.

## WebSocket Protocol

The Python server binds only to `127.0.0.1:8765`; it does not listen on a network interface. Protocol version is `1`. Godot sends `hello`, `state`, and `event` JSON messages. A state message is the snapshot object with `type: "state"` and `protocol_version: 1`.

Python sends an action envelope such as:

```json
{"type":"action","protocol_version":1,"episode_id":"ep_123_1","seq":9,"tick":420,"action":"LEFT","duration":0.35,"label_source":"rule_based","decision_metadata":{}}
```

Actions are `LEFT`, `STAY`, `RIGHT`, and `BRAKE`. `seq` increases for each policy decision; `tick` identifies the state used for that decision. `duration` is optional and limited to 5 seconds. `decision_metadata` carries model provenance such as `model_id`, `model_alias`, `probabilities`, `model_latency_ms`, `client_latency_ms` and `request_id`.

Every state message carries the authoritative `legal_actions` list computed by the simulation, and the action must be legal in the state the controller observed. Godot re-checks that against the stored decision snapshot, so a command cannot become valid merely by arriving late. Godot acknowledges a validated envelope with `"status": "validated"` — that is a transport acknowledgement, not a claim that the simulation applied the action; the `external_action_applied` event and the `action_decision` event record what actually happened. Rejections are reported for invalid actions, old episode IDs, duplicate sequences, stale/future ticks, invalid durations and illegal decision-time actions. Tick age is bounded to 30 physics ticks. Reset messages can request a seeded scenario and supported configuration values; unknown keys and out-of-range settings are rejected. `ping` receives `pong`.

The server does not need to produce an action for every telemetry sample: it decides when it receives each decision-rate state, and a slow inference never blocks message processing. Decisions are discarded if the episode changed or the state became stale while the model was thinking. If external control is selected and no valid action arrives for 1.5 seconds, the game returns to human control. Disconnects also return a normal human-play session to keyboard control; reconnection is attempted by the game. The HUD shows both connection and selected control mode. `label_source` preserves training provenance (`human`, `rule_based`, `original_model`, `fine_tuned_model`, `synthetic`, or `controller_fallback`) independently of the engine control source.

## Experiments

Run reproducible episodes for the same seed list across one or more policies:

```powershell
python python/experiment_runner.py `
  --godot godot `
  --policies rule_based,dummy `
  --seeds 101,202,303 `
  --episodes-per-seed 1 `
  --duration 30 `
  --output results/comparison.csv
```

The runner starts and stops the local controller for each policy and runs Godot headlessly with the same seed/duration configuration. Its CSV reports mean score, distance and survival time, collision/completion frequency, final lanes, applied action counts, decision latency mean and percentiles, model latency, acknowledgements, invalid commands, and controller failures, plus the verified `model_id` when a checkpoint was actually loaded. Per-episode results are written next to the CSV as `<name>.episodes.jsonl`.

Use `--dev-seeds` for exploratory scenarios and `--test-seeds` for held-out scenarios; each comparison row records which split it belongs to, and no seed appears in both.

```powershell
python python/experiment_runner.py `
  --godot godot `
  --policies rule_based,kev_original `
  --dev-seeds 101,202 `
  --test-seeds 901,902 `
  --episodes-per-seed 2 `
  --duration 30 `
  --model-registry kev_models.json `
  --output results/comparison.csv
```

Godot also supports automated seeded episodes without the Python runner:

```powershell
godot --headless --path . -- --auto-run --seeds=101,202 --episodes=1 --episode-duration=30
```

This mode writes the same telemetry and prints one `LANESHIFT_EPISODE` JSON summary per episode. With no Python server connected it uses no external actions; use `experiment_runner.py` for controlled policy comparisons.

## Kev Model Control

The Python controller talks to a Kev System One server over HTTP; Godot never speaks HTTP and never learns about Kev. The three layers stay separate:

1. Godot owns the simulation and its WebSocket action interface (port `8765` by default).
2. The Python controller owns policy selection, the System One request, and the response parsing.
3. Kev serves inference over HTTP `POST /v1/systemone` (port `8009` by default).

### Starting a Kev server

Kev requires Python 3.12 or 3.13 and `uv`. Following the official project:

```powershell
git clone https://github.com/jaredpalmer/kev
cd kev
uv sync
uv run kev serve            # serves the released checkpoint on http://127.0.0.1:8009
```

Check the served identity before connecting a controller:

```powershell
python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8009/v1/models'))['models'])"
```

Record the `run` value of the served model. The controller refuses to label a run as a model unless that value matches `expected_run`, so a swapped or missing checkpoint fails loudly instead of producing fake model results.

### Configuring the model adapter

Create `kev_models.json`:

```json
{
  "kev_original": {
    "base_url": "http://127.0.0.1:8009",
    "model": "kev-latest",
    "expected_run": "jaredpalmer/kev-0.8b",
    "label_source": "original_model",
    "timeout": 0.4
  },
  "kev_finetuned": {
    "base_url": "http://127.0.0.1:8009",
    "model": "kev-latest",
    "expected_run": "runs/laneshift-finetune-v1",
    "label_source": "fine_tuned_model",
    "timeout": 0.4
  }
}
```

`label_source` is only accepted as `original_model` or `fine_tuned_model`, and the registry must name the checkpoint that is really running. The `dummy` and `rule_based` policies report themselves as `synthetic` and `rule_based` and never claim a model identity.

### Running the controller

```powershell
python python/server.py --policy kev_original --kev-config kev_models.json --port 8765
```

Launch the game, start a run, and press `T` to hand control to the controller. The HUD shows the connection state and the active control mode.

Useful flags: `--decision-timeout` bounds how long the controller waits for one decision, `--inference-timeout` bounds the HTTP call, `--kev-base-url` / `--kev-model` / `--kev-expected-run` / `--kev-label-source` configure a single model without a registry file, and `--metrics-out` writes a metrics JSON.

A model decision is only sent if it is one of the actions the simulator reported as legal in the state the model saw. Timeouts, malformed responses, illegal choices and identity mismatches fall back to a safe `STAY` labelled `controller_fallback`, and each decision records the model id, the full probability distribution, model-reported latency, client latency, request id, request payload and response payload.

### Exercising the bridge without a model

`python/tools/mock_kev_server.py` is a local test double for the Kev HTTP surface. It is not a model and must never be used to produce research results:

```powershell
python python/tools/mock_kev_server.py --port 8010 --transcript .godot/kev_transcript.jsonl
python python/server.py --policy kev_mock --kev-config .godot/registry.json --port 8765
```

## Dataset Preparation

Pass the three files from one session to prepare accepted decision examples:

```powershell
python python/dataset_prep.py `
  --telemetry "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\telemetry.jsonl" `
  --events "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\events.jsonl" `
  --episodes "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\episodes.jsonl" `
  --output data/candidates.jsonl `
  --eval-fraction 0.2 `
  --split-seed 71
```

The tool joins each accepted action to its exact embedded decision-time state or latest preceding telemetry sample, rejects unaccepted commands as labels, reports malformed/incomplete records, and writes a separate report JSON. It exports all episode outcomes by default, with `outcome` separate from the feature-only `state`; model training must consume `state` and `action`, not `outcome`. Evaluation splitting keeps whole episodes together and groups repeated runs of the same scenario seed on one side to prevent scenario leakage. `--successful-only` exports only timeout/completion episodes, while the default keeps crash episodes for analysis.

### Exporting Kev-format training records

Training records are never produced automatically. Supply a review file of explicitly approved labels, each entry naming the example and the action:

```json
{"example_id": "ep_101_1:action:3", "action": "LEFT", "approved": true, "reviewer": "name"}
```

Then export:

```powershell
python python/dataset_prep.py `
  --telemetry "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\telemetry.jsonl" `
  --events "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\events.jsonl" `
  --episodes "$env:APPDATA\Godot\app_userdata\LANESHIFT\runs\SESSION\episodes.jsonl" `
  --output data/kev_training.jsonl `
  --format kev `
  --approved-records data/approved.jsonl `
  --include-sources human,rule_based `
  --validation-fraction 0.1 `
  --test-fraction 0.1 `
  --split-seed 71
```

The export writes Kev's labelled choice schema, one example per line:

```json
{"state":{"sim_time":0.116,"player":{"lane":1,"speed":20.35},"traffic":[],"legal_actions":["LEFT","STAY","RIGHT","BRAKE"]},"questions":{"next_action":{"type":"choice","instructions":"Choose the safest useful action. Only choose a currently legal option.","criteria":{"LEFT":"Move one lane to the left.","STAY":"Remain in the current lane.","RIGHT":"Move one lane to the right.","BRAKE":"Reduce speed using the brake."},"label":"LEFT"}}}
```

Raw telemetry, events and episode logs are never modified; the exporter only writes derived files. Alongside the training rows it writes `<name>.provenance.jsonl` with episode id, scenario seed, ticks, label source, approval record and outcome, and `<name>.manifest.json` with the split policy, record counts, action distribution, source distribution and the scenario seeds used by each split. Train, validation and test are grouped by whole scenario seeds, so a repeated seed never appears on both sides. A split needs at least three distinct scenario seeds; with fewer, the export still writes the records and reports the reason instead of silently splitting frames.

Only explicitly approved examples with a valid, legal label are exported, and counts by action, source and episode are always reported. Sources are `human`, `rule_based`, `original_model`, `fine_tuned_model` and `synthetic`; `controller_fallback` decisions are never trainable.

### Fine-tuning

LaneShift does not fine-tune anything itself. Use the official Kev training interface with the exported files, following the current documentation in the Kev repository. The exported rows already match Kev's custom labelled-data shape (`state` plus a `questions` object whose choice question carries `label`), so they can be used directly with Kev's training command, for example:

```powershell
cd kev
uv run kev train --data ..\data\kev_training.train.jsonl --base jaredpalmer/kev-0.8b --out runs\laneshift-finetune-v1
```

Use the manifest to keep the validation and test scenarios out of training, and record the resulting run id: that value is what `expected_run` must contain before the fine-tuned checkpoint may be labelled `fine_tuned_model` in a controller registry. No fine-tuning has been run in this repository, and no claim about improved gameplay has been made.

## Checks

Run the Windows test launcher with `run_tests.bat`, or run individual checks:

```powershell
python -m unittest discover -s tests -v  # from the python directory
godot --headless --path . --editor --quit
godot --headless --path . --script res://tests/simulation_test.gd
godot --headless --path . --script res://tests/game_flow_test.gd
godot --headless --path . --script res://tests/network_test.gd
godot --headless --path . --script res://tests/telemetry_test.gd
```

The Godot checks cover deterministic seeds, physical lane versus target lane, the shared legal-action set, terminal snapshot uniqueness, synthetic keyboard input, title/pause/crash/restart/menu flow, command validation, decision-time legality and JSONL persistence including write-failure recovery. The Python tests cover policy provenance, action legality, the Kev request/response contract against a mock HTTP endpoint, decision timeouts, stale-episode cancellation, dataset joins/leakage, Kev-format export, seed-grouped splits and comparison metrics. `experiment_runner.py` provides the live Godot/Python bridge smoke and multi-policy integration check.

## Known Limitations

- Rendering is procedural low-poly and intentionally small in scope; this is not a physically realistic motorcycle simulator.
- Inference runs in the controller process, so each decision waits for one HTTP round trip; a slow model is bounded by `--decision-timeout` and falls back to a safe action.
- Experiment throughput follows Godot's simulation loop; accelerated execution and GPU model inference are not implemented.
- The mock Kev service is a test double. Any result produced with it is a bridge test, not a model result.
- No real Kev checkpoint has been served in this environment, so the live path is verified against a mock HTTP endpoint that reproduces the documented contract.
- A fine-tuned checkpoint does not exist yet; the `fine_tuned_model` path is unexercised.
