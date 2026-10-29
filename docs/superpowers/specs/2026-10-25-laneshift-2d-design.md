# LANESHIFT 2D — design

Date: 2026-10-25
Status: approved by the user ("ok then go ahead go go go")

## Goal

Replace the 3D presentation with a top-down 2D lane runner while keeping the existing
authoritative simulation, telemetry schema, WebSocket controller, recording flow, and
portable packaging. The game exists to produce clean decision data for small models,
not to be a 3D graphics showcase.

## Game

- Top-down view of a three-lane road. The bike sits near the bottom; the road scrolls.
- `A` / Left: move one lane left.
- `D` / Right: move one lane right.
- `Space` / Down: brake.
- `T`: switch between human and a connected controller.
- `Esc`: pause.
- `F9`: start/stop recording.
- Traffic occupies lanes and moves at its own speed. A collision ends the run.
- Score = distance + survival + overtakes, unchanged.
- Physics runs at 60 Hz. Model snapshots and decisions run at 10 Hz. The display
  interpolates naturally because physics is continuous.

## Presentation

- `Node2D` root with custom `_draw()`: grass, road, lane markings, roadside poles,
  traffic cars, the bike, and a speed streak effect.
- Fixed camera; no 3D nodes, lights, shadows, textures, meshes, or camera math.
- World X is mirrored on screen so lane 0 is left and lane 2 is right, matching the
  existing simulation's coordinate convention.
- HUD stays as `CanvasLayer`/`Control` nodes: score, distance, speed, best, Record
  button, REC indicator, connection state, controls hint.
- Menus, pause, game over, notice, and recording flow are preserved.

## Data and control

- `scripts/sim/simulation.gd` remains authoritative and unchanged.
- Telemetry schema unchanged.
- Controller protocol unchanged.
- `python/kev_state.py` should project only the nearest three vehicles to keep the
  Kev prompt small; this is a separate follow-up.
- The Record button writes under the portable root's `recordings/` directory.

## Testing

- Existing simulation tests remain valid.
- `game_flow_test.gd` drops the 3D road/ground/camera checks and adds a 2D screen
  direction check: `LEFT` must decrease the player's screen X.
- The rest of the flow test (start, input, delayed external action, pause, crash,
  restart, menu, auto-run, telemetry uniqueness) stays.

## Portable packaging

- Windows ARM64 Godot export with embedded PCK.
- `START-GAME.bat`, `SELF-TEST.bat`, `START-KEV.bat`.
- Bundled native ARM64 Python, ONNX Runtime, and the exported Kev ONNX model.
- All application output stays under the portable root.

## Non-goals

- Art redesign beyond clean vector shapes.
- New game mechanics.
- Training or model evaluation in this change.
- Fixing the ONNX export again; the persistent backup on C: is the source.
