"""Measure local Kev ONNX load, warm-up, encode, and steady-state latency.

This tool exercises the same OnnxKevPolicy used by the game controller. It
loads the model once per thread count, pays a warm-up, then reports median,
p95, and maximum latency. Use it on the Windows host to compare native ARM64
against the VM and to test whether an optional execution provider helps.

Example:
    python python/tools/benchmark_kev.py --threads 2,4,8 --runs 15
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLING_MODEL_DIR = ROOT.parent / "LANESHIFT2D-Kev-model"
DEFAULT_MODEL = SIBLING_MODEL_DIR / "kev07b_fp16_L256.onnx"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path(r"C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx")
DEFAULT_TOKENIZER = SIBLING_MODEL_DIR / "tokenizer.json"
if not DEFAULT_TOKENIZER.exists():
    DEFAULT_TOKENIZER = Path(r"C:\LANESHIFT2D-Kev-model\tokenizer.json")
sys.path.insert(0, str(ROOT / "python"))

from policies.onnx_kev import OnnxKevPolicy  # noqa: E402

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "qnn": ["QNNExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}


def representative_state() -> dict:
    """A four-vehicle scene from the observation audit's live run."""
    return {
        "episode_id": "bench",
        "sim_time": 1.0,
        "physics_tick": 60,
        "player": {
            "lane": 1,
            "target_lane": 1,
            "lane_change_in_progress": False,
            "lane_change_cooldown": 0.0,
            "speed": 23.0,
            "base_speed": 20.0,
            "max_speed": 28.0,
            "acceleration": 3.0,
            "braking": False,
            "position": {"x": 0.0, "y": 0.0, "z": 21.9},
            "current_action": "STAY",
            "previous_action": "STAY",
            "recent_actions": ["STAY"],
        },
        "legal_actions": ["LEFT", "STAY", "RIGHT", "BRAKE"],
        "traffic": [
            {"id": "veh_0", "lane": 0, "type": "truck", "length": 7.0, "width": 2.4,
             "speed": 13.3, "relative_distance_z": 25.1, "relative_speed_z": 9.7,
             "acceleration": 0.0, "on_screen": True},
            {"id": "veh_1", "lane": 2, "type": "truck", "length": 7.0, "width": 2.4,
             "speed": 14.1, "relative_distance_z": 43.4, "relative_speed_z": 8.9,
             "acceleration": 0.0, "on_screen": True},
            {"id": "veh_2", "lane": 2, "type": "truck", "length": 7.0, "width": 2.4,
             "speed": 17.7, "relative_distance_z": 64.9, "relative_speed_z": 5.3,
             "acceleration": 0.0, "on_screen": True},
            {"id": "veh_3", "lane": 2, "type": "car", "length": 4.5, "width": 1.8,
             "speed": 18.2, "relative_distance_z": 43.2, "relative_speed_z": 4.8,
             "acceleration": 0.0, "on_screen": True},
        ],
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def measure(args, threads: int, state: dict) -> dict:
    print(f"\n=== threads={threads} provider={args.provider} ===", flush=True)
    policy = OnnxKevPolicy(
        args.model,
        args.tokenizer,
        length=args.length,
        intra_threads=threads,
        arena=args.arena,
        model_id="benchmark",
        prompt_mode=args.prompt_mode,
        providers=PROVIDERS[args.provider],
    )
    started = time.perf_counter()
    import asyncio
    asyncio.run(policy.start())
    startup_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"load_ms={policy.load_ms:.0f} warmup_ms={policy.warmup_ms:.0f} "
        f"startup_ms={startup_ms:.0f}",
        flush=True,
    )

    for _ in range(args.warmups):
        policy._infer(state)
    totals: list[float] = []
    encodes: list[float] = []
    runs: list[float] = []
    last_action = ""
    last_probabilities: dict[str, float] = {}
    for _ in range(args.runs):
        action, probabilities, encoding, total_ms = policy._infer(state)
        totals.append(total_ms)
        encodes.append(float(encoding["encode_ms"]))
        runs.append(float(encoding["run_ms"]))
        last_action = action
        last_probabilities = probabilities

    result = {
        "threads": threads,
        "provider": args.provider,
        "runs": args.runs,
        "session_load_ms": round(policy.load_ms, 1),
        "warmup_ms": round(policy.warmup_ms, 1),
        "startup_ms": round(startup_ms, 1),
        "encode_ms_mean": round(statistics.fmean(encodes), 2),
        "run_ms_p50": round(percentile(runs, 0.50), 1),
        "run_ms_mean": round(statistics.fmean(runs), 1),
        "run_ms_p95": round(percentile(runs, 0.95), 1),
        "run_ms_max": round(max(runs), 1),
        "total_ms_p50": round(percentile(totals, 0.50), 1),
        "total_ms_mean": round(statistics.fmean(totals), 1),
        "total_ms_p95": round(percentile(totals, 0.95), 1),
        "total_ms_max": round(max(totals), 1),
        "state_tokens": policy.last_encoding.get("state_tokens"),
        "prompt_tokens": policy.last_encoding.get("total_tokens"),
        "last_action": last_action,
        "last_probabilities": {key: round(value, 4) for key, value in last_probabilities.items()},
    }
    print(json.dumps(result, indent=2), flush=True)
    policy.session = None
    policy.tok = None
    policy._blob = None
    del policy
    gc.collect()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--threads", default="4", help="comma-separated intra-op thread counts")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="cpu")
    parser.add_argument("--arena", action="store_true")
    parser.add_argument("--prompt-mode", choices=("compact", "lane", "scene"), default="scene")
    parser.add_argument("--state-json", type=Path, help="optional decision-state JSON file")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")
    if not Path(args.tokenizer).exists():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}")
    state = representative_state()
    if args.state_json:
        state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    print(
        json.dumps({
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "onnxruntime": __import__("onnxruntime").__version__,
            "available_providers": __import__("onnxruntime").get_available_providers(),
            "model": args.model,
            "tokenizer": args.tokenizer,
        }, indent=2),
        flush=True,
    )
    results = []
    for value in args.threads.split(","):
        threads = int(value.strip())
        if threads < 1:
            raise SystemExit("thread counts must be positive")
        results.append(measure(args, threads, state))
    payload = {"environment": {
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "onnxruntime": __import__("onnxruntime").__version__,
        "available_providers": __import__("onnxruntime").get_available_providers(),
    }, "results": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nBenchmark written to {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
