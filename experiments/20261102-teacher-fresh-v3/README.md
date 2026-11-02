# Fresh-seed teacher validation (v3, clean label source)

This is the final 10-seed comparison of the old `rule_based` policy and the
improved `safe_teacher` policy. It is a **rule-generated teacher comparison**,
not a human demonstration and not a safety proof.

## Conditions

| Field | Value |
| --- | --- |
| Seeds | 101, 103, 107, 109, 113, 127, 131, 137, 139, 149 |
| Episodes | 1 per seed, 10 total |
| Duration | 30 s simulated |
| Traffic density | 1.5 |
| Simulation speed | 2.0x |
| Decision timeout | 60 s |
| Policies | `rule_based`, `safe_teacher` |
| Invalid commands | 0 for both |

## Results

| Policy | Collisions | Mean survival | Mean score | Mean distance | Actions | STAY | BRAKE | LEFT | RIGHT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rule_based` | 5/10 | 22.41 s | 351.36 | 517.92 m | 953 | 788 | 128 | 19 | 18 |
| `safe_teacher` | 1/10 | 27.87 s | 449.00 | 646.36 m | 1,188 | 1,004 | 141 | 20 | 23 |

The teacher ended 9 episodes by timeout and one by collision. The old rule
ended 5 by timeout and 5 by collision. Three seeds scored lower under the
teacher (103, 107, 131); seed 139 still crashed. The result supports the
teacher as a better label source, not as a solved driver.

## Artifacts

- `comparison.csv` — aggregate rows and action distributions.
- `comparison.episodes.jsonl` — per-episode outcomes.
- `comparison.rule_based.metrics.json` — controller metrics.
- `comparison.safe_teacher.metrics.json` — controller metrics.

The CSV and metrics are force-added because the repository ignores generic
`comparison.csv` and `*.metrics.json` outputs. The duplicate dev/v1/v2 runs are
local only.
