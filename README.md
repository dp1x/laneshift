# LANESHIFT

A telemetry-first 2D driving simulator for experimenting with small AI decision models.

LANESHIFT is an early-stage research prototype. It provides a small driving environment in which a human or an external controller can play, while the simulator records state and outcomes directly.

The project is intended to support experiments in fine-tuning compact decision models. The game exports structured telemetry, records discrete events, and exposes a programmatic control interface so the same scenarios can be tested across different controllers.

## Current status

What works today:

- A playable Godot driving simulator with human keyboard controls, a title screen, HUD, pause, game over and restart.
- Simulation-state export and periodic telemetry at roughly 30 Hz of simulation time.
- Event-based action and episode logs.
- A Python controller communicating with the game over a localhost WebSocket.
- Dummy and rule-based policies, both usable as baselines.
- Dataset conversion and experiment tooling, including a Kev-format training export.
- A Kev System One adapter that has driven a real game episode with the released `jaredpalmer/kev-0.8b@v1.0` checkpoint running locally on CPU.

What has **not** been established:

- Fine-tuning has not been performed.
- No claim has been made that any model improves gameplay over the rule-based baseline.

The game itself is a rough 2D prototype. Its graphics are simple vector shapes, the driving model is kinematic rather than physical, and both are still being iterated on. Do not treat it as polished or production-ready.

## Running the game

Requires Godot 4.7.2 Stable.

```powershell
godot --path .
```

On Windows, `run_game.bat` does the same thing and checks that Godot is on `PATH`.

Controls:

| Key | Action |
| --- | --- |
| `A` / Left Arrow | Move one lane left |
| `D` / Right Arrow | Move one lane right |
| `Space` / Down Arrow | Brake (hold) |
| `T` | Switch between human and connected controller |
| `Escape` | Pause or resume |

Lane changes ease over about half a second and are applied once per press. Braking slows the bike, which changes how quickly traffic is caught. A collision ends the run.

## Recording human gameplay

Click **Record** in the top bar, or press `F9`, to start a recording session. The
button changes to **Stop**, and the bar shows `REC` with elapsed time, sample count,
and event count. Recording stays active across crashes and restarts until you stop it.
Every session writes `manifest.json`, `telemetry.jsonl`, `events.jsonl`, and
`episodes.jsonl` under `<portable root>/recordings/`.

The session manifest records the session ID, UTC start and end times, build commit,
engine version, recording schema version, status, and record counts. Human actions are
labelled `human`; they are never relabelled as model actions.

## Running the Python controller

The controller is a separate process. Start it first, then launch the game, start a run, and press `T` to hand control over.

```powershell
# dummy: scripted lane changes, used to verify the bridge
python python/server.py --policy dummy

# rule-based: traffic avoidance baseline
python python/server.py --policy rule_based
```

The controller binds to `127.0.0.1:8765` only. Godot connects to it, streams state snapshots, and applies the actions that come back. If the controller is not running, the game stays fully playable with the keyboard. If the controller disconnects or stops sending actions, the game reverts to keyboard control after 1.5 seconds.

## Running a Kev model

The controller speaks to a Kev System One server over HTTP; Godot never sees Kev. The server is a separate process, so the two ports stay independent: WebSocket `8765` for game control, HTTP `8009` for inference.

Start Kev itself, following the upstream project:

```powershell
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve --python 3.13
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@v1.0 --port 8009 --device cpu
```

On Windows, Hugging Face needs `HF_HUB_DISABLE_SYMLINKS=1` because creating symlinks in the cache needs privileges that a normal user does not have.

Then register the checkpoint and start the controller:

```powershell
python python/server.py --policy kev_original --kev-config kev_models.example.json --port 8765
```

`expected_run` must match the `run` value that `GET /v1/models` reports for the checkpoint that is actually loaded. The controller refuses to label a run as a model if it does not, so a swapped or missing checkpoint fails at startup instead of producing fake model data. `kev_models.example.json` is a template; copy it and adjust the values for your own setup. Never commit API keys; the controller reads `KEV_API_KEY` from the environment.

### Status of model integration

| State | Status |
| --- | --- |
| Local CPU inference with the real checkpoint | **Verified.** `jaredpalmer/kev-0.8b@v1.0` served locally on CPU and driving a real game episode. |
| Local CPU inference for a *fine-tuned* checkpoint | **Not done.** No fine-tuned checkpoint exists yet. |
| Mock adapter test only | The mock service in `python/tools/` is used by the test suite. It runs no model. |
| Fine-tuning | **Not performed.** No claim of improved gameplay has been made. |

### Verified on this machine

Kev-0.8B was served locally on CPU on a Snapdragon X Elite (Windows on ARM64) and drove a real game episode:

- Served identity: `run: jaredpalmer/kev-0.8b@v1.0`, `base: Qwen/Qwen3.5-0.8B-Base`, backend `torch`, dtype `bfloat16`, device `cpu`, temperature `2.35`.
- Inference latency: about 12.6 s to 24.6 s per decision, 16.2 s median. A standalone probe measured 5.3 s for a smaller state.
- Process footprint: about 2.2 GB resident.
- The game received four real decisions, each with its probability distribution and request id, and applied them.

Two constraints shaped that result. PyTorch publishes no `win_arm64` wheel, so it runs as an `x86_64` build under Windows emulation. And Kev defaults to float32 weights, which does not fit this machine's commit limit (it has no page file); `KEV_DTYPE=bf16` is the documented serving precision and halves the footprint.

At roughly 15 s per decision the model is far too slow to steer at the game's normal rate. Long runs need widened bounds:

```powershell
godot --headless --path . -- --auto-run --seeds=303 --episode-duration=100 --external-timeout-ms=40000 --max-stale-ticks=6000 --traffic-density=0.12
```

### What is still unproven

- No fine-tuned checkpoint exists, so the `fine_tuned_model` path is implemented but unrun.
- The model's gameplay performance has not been measured against the rule-based baseline.
- The measured latency is specific to emulated x86-64 CPU inference on this hardware. A native build, or a smaller state, would be materially faster.

## Telemetry and data

Every recording writes a new session directory under the portable root:

```text
<portable root>/recordings/session_<timestamp>_<suffix>/
  telemetry.jsonl
  events.jsonl
  episodes.jsonl
  manifest.json
```

In the editor, the portable root is the project-relative `.portable-dev/`
directory. In an exported build, it is the directory that contains the
executable. Application-managed output never goes to `user://`, `%APPDATA%`,
`%TEMP%`, or an arbitrary working directory.

The three files record different things:

- **`telemetry.jsonl`** holds periodic state snapshots, sampled on simulation time rather than render time. A snapshot is a complete description of the world at one physics tick.
- **`events.jsonl`** holds discrete events: scenario initialisation, episode start/reset/end, action requests, action application, action rejections, collisions, pause and resume, controller connection changes.
- **`episodes.jsonl`** holds one summary per finished episode.

A real snapshot, trimmed to the interesting fields:

```json
{
  "state_id": "ep_7_1:2",
  "episode_id": "ep_7_1",
  "scenario_seed": 7,
  "sim_time": 0.0333333333333333,
  "physics_tick": 2,
  "player": {
    "lane": 1,
    "target_lane": 1,
    "lane_change_in_progress": false,
    "speed": 20.1,
    "position": {"x": 0.0, "y": 0.0, "z": 0.669166666666667},
    "velocity": {"x": 0.0, "y": 0.0, "z": 20.1},
    "current_action": "STAY"
  },
  "traffic": [
    {"id": "veh_0", "lane": 1, "relative_distance_z": 31.4941673533122, "relative_speed_z": 4.89677772521973}
  ],
  "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
  "score": 0.233583333333333,
  "distance": 0.669166666666667,
  "survival_time": 0.0333333333333333,
  "episode_status": "running",
  "control_source": "external_ai"
}
```

A real action event, from a run driven by the released Kev-0.8B checkpoint:

```json
{
  "event_id": "ep_303_1:action:1",
  "decision_seq": 1,
  "controller_seq": 1,
  "decision_tick": 7,
  "application_tick": 765,
  "application_latency_ticks": 758,
  "requested_action": "STAY",
  "applied_action": "STAY",
  "accepted": true,
  "control_source": "original_model",
  "label_source": "original_model",
  "state_id": "ep_303_1:7"
}
```

Its `decision_metadata` records which checkpoint answered, what it answered and how long it took:

```json
{
  "model_id": "jaredpalmer/kev-0.8b@v1.0",
  "model_alias": "kev-latest",
  "probabilities": {"BRAKE": 0.2392, "LEFT": 0.1637, "RIGHT": 0.1657, "STAY": 0.4315},
  "confidence": 0.242,
  "model_latency_ms": 12622.6,
  "client_latency_ms": 12629.96,
  "request_id": "19f1892287c24cc7bf7b0de0ee97b367"
}
```

`decision_tick` is the tick the controller saw; `application_tick` is the tick the action was applied. They differ when a decision takes time to arrive, which is the normal case with a model in the loop.

A mock service must never appear in this trail as a model decision. The mock in `python/tools/` is a test double: it answers the documented HTTP contract but runs no model, so a decision it produces carries no `model_id` of a real checkpoint and must not be labelled `original_model` or `fine_tuned_model`. The controller enforces this by verifying the served checkpoint's identity against `expected_run` before it will label anything as a model.

### Coordinates and units

- `X` is lateral, `Y` is up, `Z` is forward. The player advances in `+Z`; traffic also moves in `+Z` at its own speed.
- The chase camera looks along `+Z`, so its screen-right vector is world `-X`. Lane index `0` is therefore the leftmost lane on screen and sits at `+4 m` in world `X`; lane `2` is the rightmost at `-4 m`.
- `player.lane` is the lane the motorcycle currently occupies. `player.target_lane` is where it is heading. They differ while `lane_change_in_progress` is true.
- Positions are metres, velocities are metres per second, HUD speed is km/h.
- `relative_distance_z = traffic.z - player.z`, so positive means the vehicle is ahead. `relative_speed_z = player.speed - traffic.speed`, so positive means the player is closing.
- `legal_actions` is the simulation's own answer to what may be done now. It is the same list used to validate keyboard input and external commands, so nothing has to guess whether a steering action is currently possible.

## Architecture

```text
Godot simulation -> state and event telemetry -> Python controller -> decision policy -> action back to Godot
```

The game owns the authoritative state. Coordinates, velocities, traffic positions and vehicle identities are read directly from the simulation's variables; nothing is inferred from rendered pixels, and no screenshot is ever sent to a model. The policy layer sits entirely outside the engine: Python receives structured snapshots, decides, and sends back one of four actions (`LEFT`, `STAY`, `RIGHT`, `BRAKE`).

Human input and controller input go through the same action entry point in the simulation, so both are subject to the same legality rules and produce the same telemetry.

## Testing

```powershell
run_tests.bat
```

That runs the Python suite and the Godot checks. Individually:

```powershell
cd python; python -m unittest discover -s tests -v
godot --headless --path . --editor --quit
godot --headless --path . --script res://tests/simulation_test.gd
godot --headless --path . --script res://tests/game_flow_test.gd
godot --headless --path . --script res://tests/network_test.gd
godot --headless --path . --script res://tests/telemetry_test.gd
godot --headless --path . --script res://tests/options_test.gd
```

The Godot checks cover seeded reproducibility, physical versus target lane, legal-action generation, terminal snapshot uniqueness, synthetic keyboard input, the title/pause/crash/restart/menu flow, command validation, and telemetry persistence including write-failure recovery. The Python checks cover policy provenance, action legality, the Kev request and response contract against a mock HTTP endpoint, decision timeouts, dataset joins and leakage, the Kev training export, seed-grouped splits, and comparison metrics.

The Python dependencies are in `python/requirements.txt` (`websockets`, `httpx`):

```powershell
python -m pip install -r python/requirements.txt
```

## Current limitations

- Real Kev-0.8B CPU inference works on this machine, but it is slow enough (about 15 s per decision) that it cannot steer at the game's normal decision rate.
- Fine-tuning has not been performed, and the model's effect on gameplay has not been measured.
- The game is a simple 2D experimental environment, not a realistic traffic simulator. Vehicles move along fixed lanes at constant speeds, and there is no acceleration model for traffic.
- The top-down renderer uses vector shapes and a fixed X/Z split: the road stays readable while the view still shows oncoming traffic. It is intentionally minimal.
- Steering is discrete: the motorcycle changes lanes on rails rather than being steered continuously.

## Roadmap

1. Collect and validate useful training data.
2. Fine-tune the decision model.
3. Compare the original and fine-tuned models on held-out scenarios.
4. Measure gameplay performance and local inference latency.
5. Make CPU inference fast enough to steer in real time, or move it off this machine.

## Repository layout

```text
project.godot            Godot project settings
kev_models.example.json  Template registry of model policies
scenes/main.tscn         Main scene; the world is built procedurally at runtime
scripts/main.gd          Top-down 2D rendering, HUD, input, episode flow, recording
scripts/sim/simulation.gd Authoritative simulation: state, actions, seeded traffic, collisions
scripts/net/net_bridge.gd WebSocket client and command validation
scripts/telemetry/telemetry_logger.gd  Buffered JSONL writer
python/server.py         Controller server and policy selection
python/policy.py         Policy interface and decision record
python/kev_state.py      The one state projection shared by inference and training
python/policies/         dummy, rule_based, and the Kev System One adapter
python/dataset_prep.py   Log validation, candidate building, Kev training export
python/experiment_runner.py  Seeded policy comparison across scenarios
python/tools/            Mock Kev service used by the tests (not a model)
python/tests/            Python test suite
tests/                   Godot headless test scripts
```

## Development and licence

Tools: Godot 4.7.2 Stable, Python 3.10 or newer, and `uv` if you intend to run a local Kev server.

No licence has been chosen for this repository yet. Add one before distributing it.
