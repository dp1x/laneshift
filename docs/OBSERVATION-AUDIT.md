# LANESHIFT observation audit and schema v2

This audit covers the traffic scene Kev sees at a decision. It is based on the
source, not on the previous prompts. The goal was complete situational
awareness, not model tuning. No weights or training data were changed.

Observation schema: **2**. Human recordings made with schema 1 remain raw and
are excluded from v2 training by `python/export_human_decisions.py` unless
`--allow-legacy` is passed explicitly.

> **Update 2026-11-02 (L256 budget):** the structured scene observation still
> retains every active vehicle, but the prompt is capped at nine
> safety-relevant records: nearest ahead/behind per lane, then nearest by
> distance. The final `X<n>` field is the number of relevant vehicles omitted
> from the text. On the 1,348 pilot decisions this removed all 77 L256
> overflows; the largest prompt was 253 total tokens. The legacy compact prompt
> is now the short `c1` / `P` / `V` renderer (maximum 125 total tokens), not the
> old YAML, and it no longer slices the state. The token numbers in the
> historical sections below describe the pre-cap audit.

## 1. What the game actually does

### 1.1 Road and lanes

- Three lanes. Lane index 0 is the driver's left, 1 the centre, 2 the right.
  World X centres are `[4.0, 0.0, -4.0]` metres; the chase camera looks along
  +Z, so lane 0 is at positive X. `scripts/sim/simulation.gd:13`.
- The player starts in lane 1 at `z = 0`, speed 20 m/s. The base speed is 20
  and the maximum is 28. `scripts/sim/simulation.gd:27-29,65-97`.
- There are no road curves, merges, exits, obstacles, signals, weather, or
  surface changes. Legal movement is bounded only by the lane count.

### 1.2 Spawning, movement and removal

- Initial traffic is 2-4 vehicles scaled by `traffic_density` and placed
  30 + 18*i metres ahead. The first is kept out of the player's lane.
  `scripts/sim/simulation.gd:106-113`.
- A spawned vehicle gets a random lane, a z jitter of ±4 m, a random constant
  speed in `[traffic_speed_min, traffic_speed_max]` (13-21 m/s in the game
  config), and a type of car or truck. Cars are 4.5 x 1.8 m; trucks are
  7.0 x 2.4 m. `scripts/sim/simulation.gd:116-132`,
  `scripts/main.gd:6-9`.
- New traffic is spawned while `next_spawn_z < player_z + 75`, using the
  interval 22-38 s divided by density. A vehicle is removed when
  `z < player_z - 15`. `scripts/sim/simulation.gd:176-190`.
- All traffic moves at constant speed along +Z. Traffic never changes lane,
  accelerates, brakes, follows, or reacts to the player.
  `scripts/sim/simulation.gd:173-175`.
- The player's z advances at `player_speed`; the world scrolls past.
  `scripts/sim/simulation.gd:168-171`.

### 1.3 Player movement and braking

- Not braking: `player_speed += 3.0 * dt`, capped at `max_speed` (28).
- Braking: `player_speed -= 14.0 * dt`, floored at `base_speed * 0.45` (9).
  `scripts/sim/simulation.gd:153-156`.
- `BRAKE` sets `braking = true` and a brake timer of `max(0.1, duration)`.
  The controller sends duration 0.35 s, so one brake action removes about
  4.9 m/s, then braking ends and the +3 m/s² recovery begins.
  `scripts/sim/simulation.gd:255-257`, `python/server.py:598`.
- `STAY` clears braking and the brake timer.
- A lane change sets `player_target_lane` one lane left or right and moves
  `player_x` toward the target at 15 m/s. The committed lane changes only when
  the player is within 0.05 m of the target. A 4 m lane therefore takes about
  0.27 s. `scripts/sim/simulation.gd:159-166,241-250`.
- Lane changes are legal only when none is in progress and the cooldown is 0.
  The cooldown is set to 0.08 s when a change starts, so it has expired by the
  next 1 s decision. `scripts/sim/simulation.gd:243-249,267-277`.
- A lane change cannot be cancelled or redirected while it is in progress:
  LEFT and RIGHT are absent from `legal_actions` during that time.

### 1.4 Collision detection

For every physics tick and every vehicle:

- lateral: `abs(player_x - lane_center(vehicle.lane)) < vehicle.width/2 + 0.45`
- longitudinal: `abs(vehicle.z - player_z) < vehicle.length/2 + 0.8`

If both hold, the episode ends with a crash.
`scripts/sim/simulation.gd:196-208`.

This uses the vehicle's real width and length, so truck dimensions matter.
The lane-change target is not pre-checked; collisions are found during the
move. Traffic-to-traffic collisions are not modelled.

### 1.5 Renderer visibility

The renderer draws every vehicle in `sim.traffic` whose signed gap is in
`[-12, 110]` metres. Outside that gate it returns before drawing.
`scripts/main.gd:486-489`. The 3D-to-screen projection uses
`depth = 1 + max(z_rel, -12)/55`; the draw gate is the visibility rule, not an
estimated range. `scripts/main.gd:333-339`.

There are no other occluders. Fog is drawn before traffic, so a vehicle at the
far edge is still drawn on top of the fog. A vehicle at `z_rel = 110` is
visible (a few pixels at the horizon); at `110.01` it is not.

### 1.6 What is active but off-screen

- Behind: traffic is removed at `z_rel < -15`, so the complete off-screen
  behind band is `[-15, -12)` m. It is at most a few metres and short-lived.
- Ahead: traffic can be driven beyond 110 m when the player brakes or traffic
  is faster. Active traffic ahead is included out to 160 m. Beyond 160 m it is
  counted but not sent to Kev, because it cannot affect the next manoeuvre.
- On the controlled probe, a vehicle at 130 m and one at 150 m were active but
  off-screen; both were included in the `OFF` section. A vehicle at 400 m was
  active but beyond the relevance horizon; it was counted and excluded.

## 2. Variable inventory

Legend: **Engine** = the simulation stores it; **Visible** = the player can
see it; **v1** = the old lane prompt; **v2** = the schema-2 observation
(object and prompt). "Object only" means the field is in the structured state
but is not repeated in the compact prompt text.

### 2.1 Player

| Variable | Engine | Visible | v1 | v2 | Units / sign | Snapshot | Source |
|---|---|---|---|---|---|---|---|
| lane | yes | yes, bike position | nearest-lane description | `P` field 1; object | 0/1/2 left/centre/right | yes | `simulation.gd:22` |
| target lane | yes | only as movement | yes | `P` field 2; object | lane index | yes | `simulation.gd:23` |
| x position | yes | yes | no | object `position.x`; progress in prompt | m, + left | yes | `simulation.gd:24` |
| z position | yes | relative only | no | object `position.z` | m, + forward | yes | `simulation.gd:25` |
| speed | yes | HUD/streaks | yes | `P` field 3; object | m/s | yes | `simulation.gd:27` |
| acceleration | derived in snapshot | no direct display | no | `P` field 4; object | m/s²; +3, -14, 0 | yes | `simulation.gd:298-301` |
| speed limits | config | HUD indirectly | no | object `speed_limits_mps` | m/s, brake floor/max | yes | `simulation.gd:314-316` |
| braking | yes | brake light | yes | `P` field 5; object | bool | yes | `simulation.gd:30` |
| lane change in progress | yes | yes | yes | `P` field 6; object | bool | yes | `simulation.gd:34` |
| lane change direction | derived | yes | target lane only | object `lane_change_direction` | LEFT/RIGHT/NONE | yes | `kev_state.py:scene_observation` |
| lane change progress | derived from x | yes | no | `P` field 7; object | 0-100 % | yes | `kev_state.py` |
| lane change remaining | derived at 15 m/s | no | no | `P` field 8; object (`..._estimate_s`) | s, estimate | yes | `kev_state.py` |
| lane change cooldown | yes | no | yes | `P` field 9; object | s, 0.08 after start | yes | `simulation.gd:33` |
| previous action | main history | no | `current_action` | `P` field 10; object | LEFT/STAY/RIGHT/BRAKE | yes | `main.gd:_observation_snapshot` |
| recent actions | main history | no | no | object only, 8 max | action labels | yes | `main.gd:133-133` |
| action timer | yes | no | no | no | s | yes in raw only | `simulation.gd:32` |
| brake timer | yes | no | no | no | s | yes in raw only | `simulation.gd:31` |
| velocity x | yes | lean | no | no | m/s, lateral | yes in raw only | `simulation.gd:26` |
| score/distance/survival | yes | HUD | no | no | mixed | yes in raw only | `simulation.gd:40-49` |
| control source | yes | HUD | no | no | human/model label | yes in raw only | `simulation.gd:51` |

### 2.2 Traffic, per vehicle

| Variable | Engine | Visible | v1 | v2 | Units / sign | Snapshot | Source |
|---|---|---|---|---|---|---|---|
| vehicle id | yes | no | no | object `vehicle_id`; row order in prompt | string | yes | `simulation.gd:118` |
| lane | yes | yes | yes | prompt lane letter; object | L/C/R | yes | `simulation.gd:123` |
| signed gap z | yes | yes | distance only | prompt `laneDZ`; object | m; + ahead, - behind | yes | `simulation.gd:289` |
| absolute speed | yes | relative motion | no | prompt `abs` (rich mode); object | m/s | yes | `simulation.gd:291` |
| relative speed | yes | relative motion | yes | object `relative_speed_z` | m/s; player minus vehicle | yes | `simulation.gd:292` |
| closing speed | derived | relative motion | movement text/TTC | prompt `close`; object | m/s; + = gap shrinking | yes | `kev_state.py` |
| acceleration | yes (constant 0) | no | no | object `acceleration_mps2` | m/s² | yes | `simulation.gd:292-296` |
| braking | not modelled | no | no | no | - | - | traffic has no controller |
| lane change state | not modelled | no | no | no | - | - | traffic keeps its lane |
| type | yes | yes | no | prompt `c`/`t`; object | car/truck | yes | `simulation.gd:129` |
| length | yes | yes | no | object `length_m` | m | yes | `simulation.gd:130` |
| width | yes | yes | no | object `width_m` | m | yes | `simulation.gd:131` |
| on-screen | renderer rule | yes | no | prompt section `ON`/`OFF`; object | bool | yes | `main.gd:1328-1336` |
| ahead/behind | signed gap | yes | front/rear section | prompt sign; object | + ahead | yes | `main.gd` |
| time to collision | derived | no | TTC text | object only | s; `abs(gap)/closing` | yes | `kev_state.py` |

### 2.3 Road and environment

| Variable | Engine | Visible | v1 | v2 | Notes / source |
|---|---|---|---|---|---|
| lane count | yes | yes | implicit | object `road.lane_count` | 3; `simulation.gd:14` |
| lane width | yes | yes | no | object `road.lane_width_m` | 4 m between centres |
| left/right boundary | derived | yes | legal actions | object `road.at_*_boundary` | lane 0 has no LEFT; lane 2 no RIGHT |
| obstacles | not modelled | no | no | no | none |
| road changes | not modelled | no | no | no | none |
| traffic density/spawn rate | config | not directly | no | no | scenario generation, not decision state |
| collision half-extents | yes | no | no | no | player 0.45 m / 0.8 m; vehicle width/2 and length/2 |
| score/episode time | yes | HUD | no | no | outcome/feedback, not perception |

### 2.4 Information the game does not model

- Traffic lane changes, turn signals, or predicated trajectories.
- Traffic acceleration, braking, following, or reaction to the player.
- Traffic-to-traffic collisions or spacing enforcement.
- Player steering angle, yaw, tire grip, or lateral acceleration beyond the
  fixed 15 m/s lane-change interpolation.
- Road curves, slopes, intersections, obstacles, weather, or visibility
  changes.
- A physics-based stopping distance or predicted future state.
- Absolute world heading; all motion is along +Z with lane changes in X.

## 3. Information gaps found

1. **Nearest-only lane prompt.** It kept one front and one rear vehicle per
   lane. In the 100 s session, decision 0 had 4 visible vehicles but the lane
   prompt represented 2. Across the 12 decisions, 6 had fewer lane records
   than visible vehicles. Controlled `multiple_same_lane` had 3 visible
   vehicles and 1 lane record.
2. **The lane prompt was truncated.** The L256 graph leaves 143 state tokens
   after the option branch. The rendered lane prompt was 174-199 tokens on
   those 12 states. `_encode` silently cut it at 143, dropping the right-hand
   information and the legal-action tail. The encoding recorded
   `state_tokens = 144` for every decision while the stored `state_text` was
   longer, so the truncation was invisible in normal logs.
3. **The on-screen flag used a field that was not in the snapshot.** The v2
   draft read `vehicle["z"]` from `sim.get_snapshot()`, which stores
   `position.z` and `relative_distance_z` but not a top-level `z`. After the
   player left z=0 every vehicle was marked off-screen. Fixed by using
   `relative_distance_z` and covered by `tests/observation_test.gd`.
4. **Collision dimensions were missing from the snapshot.** `get_snapshot`
   omitted `length` and `width`, so a truck was represented with car
   dimensions. Fixed by adding both to the snapshot.
5. **Previous action was lost.** `sim.current_action` resets to STAY when the
   action timer expires, which happens before the next 1 s decision. The main
   loop now records the actual recent actions and injects
   `previous_action`/`recent_actions` into the snapshot.
6. **Acceleration was not exposed.** The engine's speed law is deterministic
   (+3, -14, or 0), but no field reported it. It is now computed in
   `get_snapshot` and included as `acceleration`.
7. **Off-screen relevance was undefined and over-broad.** The draft used
   -60..220 m with a sign bug; active traffic is removed at -15 and only
   traffic inside 160 m ahead can affect the next manoeuvre. The range is now
   -15..160 m, with an explicit beyond-range count.
8. **Lane summaries could cite excluded vehicles.** The draft built summaries
   from all active traffic rather than the records included in the
   observation. Summaries now use only included records.
9. **Verbose scene prototype overflowed.** An earlier scene render needed 287
   state tokens for 2 vehicles against a 153-token budget. The compact table
   now needs 85 tokens for 2 vehicles and 120 for 5 vehicles.
10. **Old human data has no schema field.** All 35 human decisions in the
    current recordings root are schema 1. They are preserved raw and excluded
    from v2 training.

## 4. Revised observation (schema 2)

### 4.1 Structured object

`scene_observation(state)` returns:

- `schema_version: 2`
- `player`: lane, target lane, `position`, speed, acceleration, speed limits,
  braking, lane-change in progress/direction/progress/remaining estimate,
  cooldown, previous action, recent actions.
- `road`: lane count, lane width, left/right boundary flags.
- `on_screen`: one record per vehicle the renderer draws.
- `off_screen`: one record per active vehicle outside the draw gate and inside
  the relevance horizon.
- `lane_summaries`: per lane, vehicle count, ahead/behind counts, nearest
  ahead/behind with distance, closing speed, visibility, and TTC.
- `legal_actions`: the authoritative set from the simulation.
- `visibility`: draw gate, relevance range, and counts for active, on-screen,
  off-screen relevant, beyond-range, and invalid records.
- `derived_fields`: names of fields that are computed rather than stored.

Each vehicle record contains: label, vehicle id, lane, signed relative
distance, signed engine relative speed, closing speed, absolute speed,
acceleration, type, length, width, on-screen, ahead, and time to collision.

### 4.2 Scene prompt

The prompt is a table with an explicit version and section labels:

```text
v2
P C C 20 3 0 0 100 0 0 S
V laneDZ close abs type; +ahead -behind +closing; c car t truck
ON
L+33 +7 13 t
R+47 +2 18 c
R+51 +6 14 t
R+69 +2 18 t
OFF
none
L1/0 C0/0 R3/0 X0
```

Field order and units:

- `P lane target speed accel brake lc progress% rem_s cd_s previous`.
  Lane/target use `L`, `C`, `R`; `S`, `L`, `R`, `B` are the previous action;
  progress is 0-100; `rem` is a 15 m/s estimate.
- Vehicle row: `lane` immediately followed by the signed gap in metres,
  then closing speed in m/s, then absolute speed in m/s (rich mode), then
  `c` (car) or `t` (truck).
- `+ahead -behind`: signed gap is `vehicle.z - player.z`.
- `+closing`: positive means the absolute gap is shrinking, from either
  side. Negative means the vehicle and player are separating.
- `ON`: drawn by the renderer at this snapshot (`-12 <= z_rel <= 110`).
- `OFF`: active, outside the draw gate, and inside `[-15, 160]` m.
- Last line: `L ahead/behind counts`, `C ...`, `R ...`, built from all
  included records, followed by `X<n>`, the number of relevant records omitted
  from the prompt. The lane counts can therefore exceed the listed rows.
- The prompt lists at most nine records. It keeps the nearest ahead and behind
  vehicle in every lane, then fills remaining slots by on-screen status and
  absolute distance. The structured object still has every vehicle.

The prompt has two token modes:

- **Rich**: up to 7 selected vehicles; includes absolute speed.
- **Dense**: 8 or 9 selected vehicles; omits absolute speed from the text.
  Both modes use the same field semantics and section labels.

The L256 graph leaves 154-168 state tokens depending on the option branch.
Measured state tokens on the pilot: 101-153 after the cap, and the largest total
prompt is 253 tokens. The 77 states that previously exceeded the budget now
list nine records plus `X<n>`; none raise. The legacy compact prompt uses a
short `c1` / `P` / `V` format and peaks at 125 total tokens.

### 4.3 Visibility rules

- On-screen: exactly the renderer's `[-12, 110]` m gate.
- Off-screen relevant: `[-15, 160]` m. Behind is bounded by the removal
  threshold; ahead is a 50 m band beyond the horizon.
- Beyond range: active vehicles outside those bounds; only a count is kept.
- No distance-based "estimated visible" flag is trusted when the game
  supplies the renderer-aligned `on_screen` bit.

## 5. Action consequences

| Action | What the engine does | What the observation lets Kev see |
|---|---|---|
| LEFT | target = lane-1, lateral move at 15 m/s, legal only if not already changing and cooldown 0 | destination lane records and summary; nearest ahead/behind, closing speed, TTC; current lane occupancy |
| RIGHT | target = lane+1, same mechanics | same for the right lane |
| STAY | clears braking, no lateral move, speed recovers at +3 m/s² | current lane front/rear records; whether braking is already active |
| BRAKE | -14 m/s² for 0.35 s, floor 9 m/s, then recovery | current speed and acceleration, closing speeds, gap/TTC; brake removes about 4.9 m/s per action |

The observation does **not** contain a future-trajectory predictor. A lane
change completes in about 0.27 s, so "will this vehicle become a hazard during
the change" is assessed from constant-speed closing speed and TTC. Traffic
speed is constant, so this estimate is exact until the next player action.
There is no modelled traffic reaction, so no additional prediction is
invented.

Repeating a lane change: not legal during the 0.27 s move or the 0.08 s
cooldown. By the next 1 s decision both are finished, and `previous_action`
plus `recent_actions` show the repeat.

## 6. Controlled tests

`tests/controlled_observation_probe.gd` writes exact engine snapshots to JSON.
`python/tools/audit_scene_observation.py` compares the engine, renderer, old
lane prompt, and v2 scene observation.

| Scenario | Active | Visible | v1 records | v2 ON | v2 OFF | Result |
|---|---:|---:|---:|---:|---:|---|
| multiple same lane | 3 | 3 | 1 | 3 | 0 | pass |
| farther vehicle closing faster | 2 | 2 | 1 | 2 | 0 | pass |
| directly behind | 2 | 1 | 2 | 1 | 1 | pass |
| adjacent lanes + off-screen behind | 3 | 2 | 3 | 2 | 1 | pass |
| off-screen approaching + beyond range | 3 | 0 | 0 | 0 | 2 | pass |
| open road | 0 | 0 | 0 | 0 | 0 | pass |
| blocked current lane | 3 | 3 | 3 | 3 | 0 | pass |
| both adjacent lanes unsafe | 3 | 3 | 2 | 3 | 0 | pass |
| safe now, unsafe within horizon | 2 | 2 | 2 | 2 | 0 | pass |
| off-screen behind at -14 m | 2 | 1 | 2 | 1 | 1 | pass |

`SCENE_AUDIT_PASS`: every visible vehicle appears exactly once in `ON`, every
relevant off-screen vehicle appears exactly once in `OFF`, no id is duplicated
across sections, and vehicles beyond 160 m are counted but not listed.

Traffic lane-change scenarios are absent because the simulation has no
traffic lane-change mechanic.

## 7. Before and after on one real state

From the 100 s playtest, decision 0. The player is in the centre lane at
20 m/s. Four vehicles are rendered: a truck at +33.4 m left, a truck at
+50.9 m right, a car at +46.6 m right, and a truck at +68.8 m right.

Old lane prompt (182 tokens, silently truncated to 143 by the L256 graph):

```text
LANESHIFT DECISION STATE
Player: lane CENTER, target CENTER, speed 20.1 m/s, braking false.
Lane change in progress: false; cooldown 0.0 s; previous action: STAY.
LEFT LANE:
  Nearest vehicle ahead: 33.4 m, closing (relative speed +6.7 m/s), time to contact 5.0 s.
  No vehicle behind within observation range.
CENTER LANE:
  No vehicle ahead within observation range.
  No vehicle behind within observation range.
RIGHT LANE:
  Nearest vehicle ahead: 46.6 m, closing (relative speed +1.9 m/s), time to contact 25.1 s.
  No vehicle behind within observation range.
Legal actions: LEFT, STAY, RIGHT, BRAKE.
```

New scene prompt (108 tokens, no truncation):

```text
v2
P C C 20 3 0 0 100 0 0 S
V laneDZ close abs type; +ahead -behind +closing; c car t truck
ON
L+33 +7 13 t
R+47 +2 18 c
R+51 +6 14 t
R+69 +2 18 t
OFF
none
L1/0 C0/0 R3/0 X0
```

The old prompt represents 2 of 4 rendered vehicles and loses its tail to
truncation. The new prompt represents all 4 exactly once, with lane,
signed gap, closing speed, absolute speed, and type, plus per-lane occupancy.

## 8. Token count and latency

- Fixed graph: L256. Scene branch: 102 tokens. State budget: 153.
- Before: lane prompt 174-199 tokens on the 12 playtest decisions; encoding
  truncated every one to 143 state tokens.
- After: scene prompt 86-120 tokens on the same 12 states; 96-111 tokens in a
  live 5-vehicle scene. No truncation.
- Controlled scene token counts: 66 (open road), 85 (2), 107 (4), 120 (5),
  151 (11 records dense). After the 2026-11-02 cap, dense states list nine
  records plus `X<n>`; the largest of the 1,348 pilot prompts is 253 total
  tokens. A 20-record state lists nine and reports the remaining eleven.
- Side-by-side ONNX runs on the same 10 states: scene 81.3 mean state tokens
  versus lane 142.8. Mean latency was 1.67 s for scene and 1.58 s for lane in
  one run; a later run measured 2.55 s versus 2.90 s. The host is a 16 GB
  Snapdragon VM with no GPU and large run-to-run variance, so these latency
  differences are noise. The graph is padded to L256, so prompt length inside
  the budget does not change model compute.
- Live scene session: 3 decisions, state tokens 109-111, total prompt tokens
  199-212, model latency 3.25-3.56 s (mean 3.36 s). The earlier 100 s lane
  session averaged 6.85 s with p95 11.1 s; that was a different host load and
  is not a controlled A/B.

## 9. Human recording compatibility

- Human `action_decision` events now carry `state_before` from
  `_observation_snapshot()` with `observation_schema: 2`.
  `tests/human_schema_test.gd` records a human LEFT and asserts schema 2,
  traffic presence, `label_source=human`, and `human_action`.
- `python/export_human_decisions.py` writes the raw events unchanged to
  `raw_human.jsonl`, and only creates training rows when
  `state_before.observation_schema == 2`. Schema 1 records are counted in
  `legacy_schema_excluded` unless `--allow-legacy` is explicitly passed.
- Current recordings root: 35 human decisions, all schema 1, 0 training rows;
  raw records and their hashes are preserved. The pilot is not mixed into v2.
- The dataset card now records `observation_schema: 2` and each training
  example carries the same value.

## 10. What Kev can and cannot assess

It can assess:

- whether the destination lane is occupied ahead or behind;
- whether a faster vehicle is approaching from behind;
- whether a more distant vehicle is closing and at what TTC;
- whether the current lane is safe;
- whether braking reduces the player's speed and closing rate;
- whether a lane change can finish before an imminent contact, using the
  0.27 s move and the constant-speed closing estimate;
- whether a repeat lane change is legal from cooldown and previous action.

It cannot assess:

- a traffic vehicle's intention to change lanes (not modelled);
- traffic braking or accelerating (not modelled);
- a guaranteed future collision state (no predictive physics; only
  constant-speed TTC);
- exact brake duration effect unless it has learned the 0.35 s controller
  duration (the observation includes the current -14 m/s² acceleration).

## 11. Live result and remaining model issue

A 6 s evaluation run with schema 2 chose STAY, LEFT, STAY. At decision 1 the
observation clearly showed a truck 25 m ahead in the destination left lane,
closing at 10 m/s, with the left lane summary `L1/0`. Kev chose LEFT anyway and
crashed into that truck 2.7 s into the episode. The information was present;
the zero-shot policy did not use it. This is the expected next problem after
the observation fix, and it is not solved by adding more fields. Fine-tuning
is the next step and was intentionally not done here.

## 12. Files changed and tests

Changed:

- `scripts/sim/simulation.gd` — length, width, speed, acceleration, player
  acceleration/limits in snapshots.
- `scripts/main.gd` — schema 2 snapshot, renderer-aligned `on_screen` from
  `relative_distance_z`, previous/recent actions, human `state_before`.
- `python/kev_state.py` — schema 2 scene observation, visibility ranges,
  complete vehicle records, lane summaries, compact rich/dense prompt.
- `python/policies/onnx_kev.py` — scene prompt mode; explicit overflow error
  instead of truncation.
- `python/server.py`, `scripts/play_kev.ps1` — `scene` prompt option/default.
- `python/export_human_decisions.py` — observation schema gate, card field.
- `python/tests/test_scene_observation.py`,
  `python/tests/test_scene_token_budget.py` — scene records, render, token
  budget.
- `python/tools/audit_scene_observation.py`,
  `python/tools/measure_scene_latency.py` — controlled audit and measurement.
- `tests/observation_test.gd` — controlled engine states and visibility gate.
- `tests/human_schema_test.gd` — human schema 2 recording.
- `tests/controlled_observation_probe.gd` — ten controlled snapshots.
- `run_tests.bat`, `.github/workflows/portable-windows-arm64.yml` — new tests.

Passed:

- 90 Python tests (including 11 scene/token tests).
- Godot: `SIMULATION_TESTS_PASS`, `GAME_FLOW_TEST_PASS`, `NETWORK_TEST_PASS`,
  `TELEMETRY_TEST_PASS`, `TELEMETRY_FLOW_TEST_PASS`, `TURN_BASED_TEST_PASS`,
  `OBSERVATION_TEST_PASS`, `HUMAN_SCHEMA_TEST_PASS`, `OPTIONS_TEST_PASS`.
- `SCENE_AUDIT_PASS` on 10 controlled scenarios.
- Live turn-based scene session: 3 decisions, no prompt errors, actions
  applied with `label_source=original_model`.

Not verified:

- The packaged `LANESHIFT.exe` and `dist` ZIP were not rebuilt because this
  host has no Godot 4.7.2 export templates. Source runs are current; the
  portable build still contains the older executable and controller.
- The model's decision quality after the observation change; that requires
  fine-tuning and held-out evaluation.
