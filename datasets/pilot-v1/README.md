# LANESHIFT pilot decision dataset v1

**RULE-GENERATED — NOT HUMAN DEMONSTRATIONS.**

This is the canonical copy of the 1,348-row decision dataset produced by the
deterministic `rule_based` policy. Every label is a scripted rule action, not a
human choice and not an expert safety label. It is suitable for pipeline
validation, the first supervised baseline and a first Kev imitation fine-tune.
It is not suitable for a human-behaviour or safety claim.

## Files

| File | Contents |
| --- | --- |
| `decisions.jsonl` | Canonical 1,348 decision rows (Kev/MLP export source) |
| `decisions.csv` | CSV view of the same rows |
| `decisions.card.json` | Data card: conditions, counts, splits, provenance |
| `provenance.json` | Machine-readable source, label, split and file hashes |
| `pilot-15x30.csv` | Matched-policy aggregate summary |
| `pilot-15x30.episodes.jsonl` | Per-episode pilot rows |
| `pilot-15x30.rule_based.metrics.json` | Rule-policy pilot metrics |
| `SHA256SUMS` | SHA-256 for every file above except this checksum file |

## Conditions

- Policy: `rule_based` (deterministic scripted teacher)
- Label source: `rule_based`
- Rows / episodes / scenario seeds: 1,348 / 15 / 15
- Episode duration: 30 s simulated
- Traffic density: 1.5
- Simulation speed: 2.0x
- Decision / telemetry cadence: 10 Hz / 30 Hz
- Source session: `session_20261101_144044_173`
- Source code commit recorded in the data card: `630a9f9` (dirty at export)

## Action distribution

| Action | Rows | Share |
| --- | ---: | ---: |
| STAY | 1,192 | 88.43% |
| BRAKE | 111 | 8.23% |
| LEFT | 24 | 1.78% |
| RIGHT | 21 | 1.56% |

The imbalance is real: the teacher keeps the centre lane most of the time, and
the dataset contains only 45 turns. A plain cross-entropy fine-tune will learn
the STAY prior. Report macro-F1, per-action recall and unsafe-STAY rate
alongside accuracy.

## Splits

The data card contains the exact rows and seeds. Splits are by whole scenario
seed; adjacent decisions from one episode never cross a split.

- Train: seeds 11, 19, 23, 31, 37, 43, 47, 53, 67
- Validation: seeds 3, 41, 59
- Test: seeds 7, 29, 61

## Provenance and integrity

- The original recording lives in the local, gitignored
  `.portable-dev/recordings/session_20261101_144044_173`.
- `decisions.card.json` records the conditions, counts, split policy, code
  commit and dataset SHA-256.
- `provenance.json` records the label type, source session, seed splits and
  per-file hashes.
- `SHA256SUMS` verifies the files in this directory:

```powershell
Get-Content SHA256SUMS | ForEach-Object {
    $hash, $path = $_ -split '  ', 2
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $hash) {
        throw "checksum mismatch: $path"
    }
}
```

The `decisions.jsonl` SHA-256 is
`a4ff7096590b1e59ac635f086b546bf733eb9689d581517f51acce79d6903ce1`.

## Use

This directory is the dataset of record. Do not regenerate it in place from a
different build without bumping the dataset name and updating the card. Keep
the original recordings local; this directory is the portable, reviewable copy.
