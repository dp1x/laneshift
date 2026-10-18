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

Available policies are `dummy`, `rule_based`, `model`, and `fine_tuned_model`. The model options are replaceable adapters that currently return `STAY`; no pretrained or fine-tuned model is bundled or implied. The dummy policy deliberately changes lanes so the bridge can be smoke-tested.

## Architecture

- `scripts/sim/simulation.gd` owns time, player actions, speed, seeded traffic, collisions, scoring, and snapshots.
- `scripts/main.gd` builds the procedural world, camera, HUD, input, presentation, and episode flow. Both keyboard and external input call the same simulation action method.
- `scripts/telemetry/telemetry_logger.gd` buffers JSONL records and writes batches on a worker thread; episode boundaries and exit drain the queue.
- `scripts/net/net_bridge.gd` is an optional Godot WebSocket client. It validates protocol messages and falls back to human control after disconnect or action timeout.
- `python/server.py` hosts the loopback WebSocket endpoint and keeps policy, protocol, action counts, and decision latency outside Godot.
- `python/policies/` contains the scripted policy, rule-based baseline, and replaceable model adapter.
- `python/dataset_prep.py` joins event decisions to pre-action observations and separates outcomes from model inputs.
- `python/experiment_runner.py` starts one controller and one headless Godot run per policy, reusing scenario seeds and writing a comparison CSV.

All visuals are made from primitive Godot meshes and materials; there are no downloaded assets. The simulation is kinematic and lane-based rather than a rigid-body motorcycle model.

## Coordinates And Timing

- World `X` is lateral: negative is left, positive is right. Lane centers are `-4`, `0`, and `+4` meters.
- World `Y` is up. World `+Z` is forward. The player advances in `+Z`; traffic also moves in `+Z` at its own speed.
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

`telemetry.jsonl` contains `state_id`, `episode_id`, `scenario_seed`, `sim_time`, `physics_tick`, player position/lane/speed/velocity/action, nearby traffic IDs and their positions/velocities/lanes, relative gaps and speeds, score, distance, survival time, collision and episode status, control source, and wall timestamps.

`events.jsonl` records scenario initialization (including configuration and initial traffic), episode starts and ends, control connection/mode changes, collisions, overtakes, rejected external commands, and action decisions. An `action_decision` contains requested and applied actions, acceptance, decision sequence, source labels, the controller-observed state ID/tick and exact `state_before` snapshot, plus separate application tick/time metadata. For external control, `state_before` is the exact snapshot sent to the controller, even when the accepted command arrives several physics ticks later. Human actions use the current pre-action snapshot. Human actions are recorded even without a controller.

`episodes.jsonl` has one summary for each ended episode, with seed, score, distance, survival time, collision and overtake counts, final lane, action changes, and termination reason. Restart, menu exit, quit, and normal time-limit completion are also given termination reasons.

Writes are buffered away from the simulation thread, periodically queued, and drained at episode boundaries and shutdown. File errors produce a Godot warning while play continues. Log directories and high scores remain local to the Godot user data directory.

## WebSocket Protocol

The Python server binds only to `127.0.0.1:8765`; it does not listen on a network interface. Protocol version is `1`. Godot sends `hello`, `state`, and `event` JSON messages. A state message is the snapshot object with `type: "state"` and `protocol_version: 1`.

Python sends an action envelope such as:

```json
{"type":"action","protocol_version":1,"episode_id":"ep_123_1","seq":9,"tick":420,"action":"LEFT","duration":0.35,"label_source":"rule_based"}
```

Actions are `LEFT`, `STAY`, `RIGHT`, and `BRAKE`. `seq` increases for each policy decision; `tick` identifies the state used for that decision. `duration` is optional and limited to 5 seconds. Godot acknowledges valid envelopes and returns a rejection reason for invalid actions, old episode IDs, duplicate sequences, stale/future ticks, and invalid durations. Tick age is bounded to 30 physics ticks. Reset messages can request a seeded scenario and supported configuration values; unknown keys and out-of-range settings are rejected. `ping` receives `pong`.

The server does not need to produce an action for every telemetry sample: it decides when it receives each decision-rate state. If external control is selected and no valid action arrives for 1.5 seconds, the game returns to human control. Disconnects also return a normal human-play session to keyboard control; reconnection is attempted by the game. The HUD shows both connection and selected control mode. `label_source` preserves training provenance (`human`, `rule_based`, `original_model`, `fine_tuned_model`, or `synthetic`) independently of the engine control source.

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

The runner starts and stops the local controller for each policy and runs Godot headlessly with the same seed/duration configuration. Its CSV reports mean score, distance and survival time, collision/completion frequency, final lanes, applied action counts, decision latency mean and percentiles, acknowledgements, invalid commands, and controller failures. The server also writes per-policy metrics beside the CSV. Use held-out seeds for later model comparisons; keep inference in Python and use the same protocol for every model.

Godot also supports automated seeded episodes without the Python runner:

```powershell
godot --headless --path . -- --auto-run --seeds=101,202 --episodes=1 --episode-duration=30
```

This mode writes the same telemetry and prints one `LANESHIFT_EPISODE` JSON summary per episode. With no Python server connected it uses no external actions; use `experiment_runner.py` for controlled policy comparisons.

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

For supervised fine-tuning, select successful episodes and useful action segments only after reviewing the later outcome and the policy's intended behavior. Do not silently label all survival actions as good examples: retain crashes and rejected actions in the source logs, inspect them separately, and record the chosen provenance. This project prepares/validates candidates; it does not fine-tune a model.

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

The Godot checks cover deterministic seeds, shared action physics, synthetic keyboard input, title/pause/crash/restart/menu flow, command validation, and JSONL persistence. The Python tests cover policy/protocol behavior, dataset joins/leakage and episode splits, and comparison metrics. `experiment_runner.py` provides the live Godot/Python bridge smoke and multi-policy integration check.

## Known Limitations

- Rendering is procedural low-poly and intentionally small in scope; this is not a physically realistic motorcycle simulator.
- The default `model` and `fine_tuned_model` policies are adapter stubs until a real decision model is supplied.
- Experiment throughput follows Godot's simulation loop; high-speed accelerated execution and GPU model inference are not implemented.
- Automated headless checks validate simulation/protocol/persistence but do not replace a visual GUI play-through. The environment used for implementation did not provide an accessible desktop screenshot inspection path.
