# LANESHIFT live mode

Live mode is the continuous Kev demo. The simulation runs at its normal 60 Hz
physics rate while the model thinks in a separate process. The game never waits
for inference, and the model's answer is applied as a discrete command when it
is still recent enough to describe the current road.

The name in the launcher is `PLAY-KEV.bat` (live), `PLAY-KEV-EVAL.bat` (strict
turn-based evaluation), and `PLAY-KEV-WATCH.bat` (legacy safety mode). The live
launcher passes `--cycle-seeds`, so each crash is followed by the next seed
rather than the identical scenario. Without `--max-episodes`, it keeps cycling
until the window closes.

## Why live mode exists

The previous `PLAY-KEV-EVAL.bat` path was turn-based on purpose: it froze the
simulation at each decision so the model saw exactly one state and its action
was applied without a timing race. That makes clean single-decision experiments,
but it looks and feels like a slideshow. It also aborted the episode on a
decision timeout, which is why a slow inference produced `inference_error`
after about two simulated seconds.

Live mode separates the two concerns:

| Mode | Simulation | Model input | Failure behavior | Use |
| --- | --- | --- | --- | --- |
| `live` | continuous 60 Hz | one observation per decision request | log, clear the request, continue | watch Kev drive |
| `eval` | frozen between decisions | exactly one observation per turn | record `model_error`, end the episode | judge one decision without timing races |
| `watch` | continuous 60 Hz | latest state, coalesced at 10 Hz | rule safety heartbeat and `controller_fallback` | the older safety-assisted demo |

## Decision schedule

Live mode uses a single decision slot.

1. The simulation advances every physics tick. It does not wait for the model.
2. A decision request starts when the model is idle and at least
   `--decision-interval-s` of simulation time has passed since the last request.
3. The request sends the exact observation snapshot, stores it by tick, and
   marks one request in flight.
4. The controller may stream other state updates at 10 Hz. Those updates are
   marked `decision_request=false`; they refresh the controller's notion of the
   latest state, but they never occupy the decision queue.
5. The next request cannot start until the current one is answered, fails, or
   times out. There is no unbounded queue. The controller also has a one-slot
   queue and one inference worker, so a timed-out native inference cannot be
   stacked under the next one.

The default is one request per second of simulation time, a 250 ms cooldown
after an answer, and at most one inference in flight. If inference takes longer
than a second, the effective cadence is the inference time plus the cooldown.

## Staleness

Every request carries its state id, physics tick, and simulation time. When an
answer arrives, the game computes:

```
decision_age_s = current_sim_time - request_sim_time
```

If `decision_age_s` is greater than `--max-decision-age-s`, the action is
dropped. The game logs `live_decision_dropped` with the age, the requested
action, and the model metadata. It does not apply the action, and it does not
pretend the model acted.

The default live bound is 5.0 simulated seconds. That is deliberately loose
enough to keep a slow host playable, and it is much looser than a strict
evaluation should use. At 20 m/s, a 5 s old observation is 100 m behind the
current bike. The observation range is 110 m ahead, so this is the edge of a
still-relevant answer, not a safe default for a research claim. Tighten it with
`--max-decision-age-s` when inference is fast.

An answer also has to belong to the current episode and match the tick of the
request that is currently in flight. An answer for an old request, a previous
episode, or no request is ignored and logged.

If every answer is stale, inference is slower than the bound. Measure it
(`BENCH-KEV.bat`), reduce render or inference contention, or raise the bound
deliberately. The HUD shows `OK` (applied), `STALE` (dropped for age), `REJ`
(rejected for legality), `FAIL` (errors plus timeouts), and `TO` (timeouts), so
a drop is visible without reading the logs.

## Actions between decisions

`LEFT` and `RIGHT` are one-shot lane-change commands. The game applies each one
once. It does not repeat a lane change every frame, and the simulation's own
legality rules reject a second change while one is in progress. After the lane
change finishes, the bike stays in the new lane until another command arrives.

`STAY` clears braking and does not issue a steering command. `BRAKE` is a short
brake application; the controller sends a duration, and the bike recovers speed
after it ends. Between decisions, the simulation keeps its normal speed
dynamics. The game does not synthesize a hidden action to fill the gap.

## Failures and timeouts

Live mode is for watching, not for hiding a broken model.

- **Late answer:** dropped as stale, counted in `live_decisions_stale`, logged.
- **No answer within `--request-timeout-ms`:** the request is cleared, counted
  in `live_decision_timeouts`, logged as `live_decision_timeout`, and the game
  continues. It does not end the episode.
- **Controller `model_error`:** the matching request is cleared, counted in
  `live_inference_errors`, logged, and the game continues.
- **Controller disconnect:** the game shows `OFFLINE`, clears the request, and
  keeps rendering. The bike coasts without new commands.
- **Controller timeout on its side:** the controller sends `model_error` in
  live mode rather than a rule action. The game never labels a fallback as Kev.

Strict evaluation keeps the opposite bias: a model failure is recorded and the
episode ends with `inference_error`. That preserves the claim that every
applied action in an evaluation episode came from the model.

## Diagnostics

The in-game AI panel shows, at 10 Hz:

- current state: `THINKING`, `ACTION`, `STALE`, `REJECTED`, `TIMEOUT`, `ERROR`,
  `OFFLINE`, or `RUN ENDED`;
- the last requested action and its probability;
- the action probability distribution when the model supplied one;
- `INFER` (model latency), `AGE` (observation age of the last answer), and
  `AGO` (time since the last answer, error, or timeout);
- `SEED`/`EP` (episode), simulation time, `DEC` (completed answers), `OK`
  (applied), `FAIL` (errors plus timeouts), and `TO` (timeouts);
- `STALE` and `REJ` counters on the latency line once a drop occurs.

The AI panel deliberately stays in the corner. The road camera and HUD keep
their normal layout.

## Telemetry semantics

Live mode adds these events to the existing telemetry:

| Event | Meaning |
| --- | --- |
| `live_decision_requested` | a request was sent; contains `request_seq`, `state_id`, tick, sim time |
| `external_action_received` | an answer matched the live request; contains `decision_age_s`, request start, model metadata |
| `action_decision` | the action was applied or rejected; now includes `decision_age_s` and request timing |
| `live_decision_dropped` | the answer was too old or otherwise unusable |
| `live_decision_timeout` | no answer arrived within the request timeout |
| `live_inference_error` | the controller reported a model error for this request |

`label_source` remains `original_model` or `fine_tuned_model` for model
actions. `controller_fallback` is never produced in live mode. Human actions
keep `label_source=human`; rule actions keep `rule_based`.

## What live mode does not prove

A long live episode proves that the integration is non-blocking and that the
model's answers are being applied with explicit timing. It does not prove that
Kev is driving well. A model that always answers `STAY` can complete an episode
on an open road. Use the action distribution, collision outcomes, and the
strict evaluation mode to judge the policy. Use the matched experiment runner
with held-out seeds for a comparison claim.

## Configuration

| Flag | Default | Meaning |
| --- | --- | --- |
| `--live` | off | enable continuous live mode |
| `--cycle-seeds` | off | use the next seed each episode instead of repeating a seed |
| `--decision-interval-s` | `1.0` | minimum simulated seconds between request starts |
| `--max-decision-age-s` | `5.0` | maximum observation age for applying an answer |
| `--request-timeout-ms` | `25000` | wall-clock timeout for one request |
| `--inference-cooldown-ms` | `250` | wall-clock gap after an answer before the next request |

The launcher passes these values and exposes them as PowerShell parameters.
`PLAY-KEV.bat -MaxDecisionAge 2 -DecisionInterval 2` is a tighter live run.
