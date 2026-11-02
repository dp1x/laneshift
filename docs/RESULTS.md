# Pilot results

This is a small matched pilot, not a finished study. It exists to prove the
measurement path and to establish a baseline before any fine-tuning.

These numbers predate the single-inference-worker fix in the controller and the
live-mode work. They remain a historical baseline; rerun the matched pilot with
the current controller before using the latency or score figures in a new
comparison.

## Conditions

| Field | Value |
| --- | --- |
| Policies | `rule_based`, `onnx_kev` |
| Seeds | 1, 2, 3, 4, 5 |
| Episodes per seed | 1 |
| Episode duration | 20 s simulated |
| Traffic density | 0.7 |
| Simulation speed | 1.0x |
| Decision cadence | 10 Hz |
| Model | `jaredpalmer/kev-0.8b@v1.0+onnx:L256` |
| Model SHA-256 | `031a03bd0d162bbbd48bf554bbd1878f29f870dfc3c28c69f5362f3eb5cc3129` |
| ONNX threads | 8 |
| Machine | Snapdragon X Elite, Windows on ARM64, CPU |
| Controller timeout | 10 s |
| Staleness bound | 6000 ticks (100 s) |

The runner gave both policies the same seeds, duration, density, simulation
speed, timeout and staleness bound. The raw summary is
[`pilot-5x20.csv`](pilot-5x20.csv); per-episode rows are in
[`pilot-5x20.episodes.jsonl`](pilot-5x20.episodes.jsonl).

## Results

| Policy | Episodes | Timeouts | Crashes | Mean score | Mean distance | Mean survival | Decisions | Action distribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rule_based` | 5 | 5 | 0 | 284.77 | 547.67 m | 20.00 s | 855 | STAY 843, BRAKE 8, LEFT 3, RIGHT 1 |
| `onnx_kev` | 5 | 2 | 3 | 199.61 | 404.64 m | 14.83 s | 30 | STAY 30 |

Inference latency for `onnx_kev`:

| Metric | Value |
| --- | ---: |
| Mean decision latency | 2119 ms |
| Median | 1637 ms |
| p95 | 4075 ms |
| Model mean | 2056 ms |

The controller received 68 states in an earlier 8-second single episode but
processed only 5 decisions; most states were superseded. At a 10 Hz decision
cadence, a 1.6-4.1 s model is a delayed controller.

## Reading the result

- The rule-based policy survives all five seeds and avoids every collision. It
  acts only 12 times in 855 decisions; the traffic density is low enough that
  staying in the centre lane is usually safe.
- The released Kev checkpoint chooses STAY for all 30 decisions. It survives
  when the centre lane stays clear and crashes when it does not. It never uses
  LEFT, RIGHT or BRAKE.
- The model's lower score is explained by three early crashes. On seeds 3 and 5,
  where the centre lane stays clear, it matches the rule-based score exactly.
- The comparison is descriptive. Five seeds and 20 seconds are enough to expose
  a broken policy and prove the pipeline; they are not enough for a confidence
  claim, a calibration claim, or a fine-tuning claim.

## What this does not show

- It does not show that Kev is a good driver.
- It does not show that fine-tuning improves anything.
- It does not measure calibration (Brier score, log loss, ECE).
- It does not test human play or a phone-captured dataset.
- It does not test generalization to held-out human sessions.

## Next experiment

1. Increase the sample to at least 20 seeds and 60 seconds, with density 1.0.
2. Add a stronger scripted baseline that changes to the emptiest lane and brakes
   early; the current rule-based policy is a floor.
3. Train a small MLP or GRU supervised baseline on approved decisions.
4. Fine-tune Kev, label the checkpoint `fine_tuned_model`, and record its hash.
5. Compare original, fine-tuned and scripted policies on held-out seeds.
6. Report calibration and action distributions alongside score and survival.
