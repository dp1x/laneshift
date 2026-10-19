# LANESHIFT

A telemetry-first driving simulator for experimenting with small AI decision models.

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
- A Kev System One adapter, tested against a mock service.

What has **not** been established:

- The real Kev-0.8B checkpoint has not been verified in this environment. The adapter is proven only against a mock service that reproduces the documented HTTP contract.
- No fine-tuning has been performed.
- No claim has been made that any model improves gameplay over the rule-based baseline.

The game itself is a rough prototype. Its graphics are functional low-poly shapes, the driving model is kinematic rather than physical, and both are still being iterated on. Do not treat it as polished or production-ready.

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

## Running the Python controller

The controller is a separate process. Start it first, then launch the game, start a run, and press `T` to hand control over.

```powershell
# dummy: scripted lane changes, used to verify the bridge
python python/server.py --policy dummy

# rule-based: traffic avoidance baseline
python python/server.py --policy rule_based
```

The controller binds to `127.0.0.1:8765` only. Godot connects to it, streams state snapshots, and applies the actions that come back. If the controller is not running, the game stays fully playable with the keyboard. If the controller disconnects or stops sending actions, the game reverts to keyboard control after 1.5 seconds.

## Telemetry and data

Every launch writes a new session directory under the Godot user data location:

```text
%APPDATA%\Godot\app_userdata\LANESHIFT\runs\session_<timestamp>_<suffix>\
  telemetry.jsonl
  events.jsonl
  episodes.jsonl
```

These files are runtime data. They are not tracked by Git.

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

A real action event, from a run driven by the Kev adapter against the mock service:

```json
{
  "event_id": "ep_5150_1:action:1",
  "decision_seq": 1,
  "controller_seq": 1,
  "decision_tick": 7,
  "application_tick": 7,
  "state_id": "ep_5150_1:7",
  "requested_action": "LEFT",
  "applied_action": "LEFT",
  "accepted": true,
  "control_source": "original_model",
  "label_source": "original_model"
}
```

`decision_tick` is the tick the controller saw; `application_tick` is the tick the action was applied. They differ when a decision takes time to arrive, which is the normal case with a model in the loop.

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
```

The Godot checks cover seeded reproducibility, physical versus target lane, legal-action generation, terminal snapshot uniqueness, synthetic keyboard input, the title/pause/crash/restart/menu flow, command validation, and telemetry persistence including write-failure recovery. The Python checks cover policy provenance, action legality, the Kev request and response contract against a mock HTTP endpoint, decision timeouts, dataset joins and leakage, the Kev training export, seed-grouped splits, and comparison metrics.

The Python dependencies are in `python/requirements.txt` (`websockets`, `httpx`):

```powershell
python -m pip install -r python/requirements.txt
```

## Current limitations

- Real Kev-0.8B CPU inference still needs to be verified on the target environment. The adapter is currently proven only against a mock service.
- Fine-tuning has not been performed, and the model's effect on gameplay has not been measured.
- The game is a simple experimental environment, not a realistic traffic simulator. Vehicles move along fixed lanes at constant speeds, and there is no acceleration model for traffic.
- The compatibility renderer is used by default so the game runs on machines without a GPU. It has no depth fog, so distance haze is approximated with a procedural sky gradient.
- Steering is discrete: the motorcycle changes lanes on rails rather than being steered continuously.

## Roadmap

1. Verify local Kev inference on the target hardware.
2. Collect and validate useful training data.
3. Fine-tune the decision model.
4. Compare the original and fine-tuned models on held-out scenarios.
5. Measure gameplay performance and local inference latency.

## Repository layout

```text
project.godot            Godot project settings
scenes/main.tscn         Main scene; the world is built procedurally at runtime
scripts/main.gd          World construction, camera, HUD, input, episode flow
scripts/sim/simulation.gd Authoritative simulation: state, actions, seeded traffic, collisions
scripts/net/net_bridge.gd WebSocket client and command validation
scripts/telemetry/telemetry_logger.gd  Buffered JSONL writer
python/server.py         Controller server and policy selection
python/policy.py         Policy interface and decision record
python/policies/         dummy, rule_based, and the Kev System One adapter
python/dataset_prep.py   Log validation, candidate building, Kev training export
python/experiment_runner.py  Seeded policy comparison across scenarios
python/tools/            Mock Kev service used by the tests
python/tests/            Python test suite
tests/                   Godot headless test scripts
```

## Development and licence

Tools: Godot 4.7.2 Stable, Python 3.10 or newer, and `uv` if you intend to run a local Kev server.

No licence has been chosen for this repository yet. Add one before distributing it.
