# Project intent

LANESHIFT is a simulation testbed for a Jev/Kev-style decision model. The game
exists to produce structured state, actions and outcomes under controlled
conditions, not to be a graphics showcase.

## The original idea

The original design conversation proposed this chain:

```
phone gameplay -> ADB capture -> frames + touch + sensors -> perception
-> human decision dataset -> synthetic counterfactuals -> Kev fine-tuning
-> calibration and held-out comparison
```

That chain is the long-term research idea. This repository implements a smaller,
fully instrumented testbed: the Godot simulation supplies ground-truth state,
a Python controller records decisions, and a local ONNX Kev model can drive the
game. The phone capture and computer-vision layers are not implemented.

## What the testbed is for

- Compare a rule-based baseline, the released Kev checkpoint and, later, a
  fine-tuned checkpoint under identical seeds, duration, traffic density and
  simulation speed.
- Record every decision with its state, action, probability distribution,
  model identity and latency.
- Build a training export from approved decisions without silently mixing human,
  rule-based, model and synthetic labels.
- Measure score, survival, collisions, action distribution, inference latency
  and calibration on held-out scenario seeds.

## Rules that keep the experiment honest

1. The same seed, duration, traffic density and simulation speed apply to every
   policy in a comparison.
2. Splits are by scenario seed or session, never by adjacent frames.
3. A model run records a model id and a hash of the model file. A missing or
   swapped checkpoint must fail rather than inherit a label.
4. Observed, inferred and generated data stay distinct. The simulation's state
   is observed; a future perception pipeline's state is inferred; a future LLM
   augmentation is generated.
5. No claim that fine-tuning improves driving until a fine-tuned checkpoint has
   been compared on held-out seeds.

## Current scope

The current game is a three-lane top-down lane runner with discrete actions
(`LEFT`, `STAY`, `RIGHT`, `BRAKE`). The simulation is kinematic. That is enough
to test the decision loop and the data pipeline. It is not a driving simulator
in the physical sense.

The next research step is a small supervised baseline and then a Kev fine-tune.
The next engineering step is a larger matched evaluation, not more game features.
