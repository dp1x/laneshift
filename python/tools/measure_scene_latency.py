"""Measure prompt size and local ONNX latency for the scene observation.

The graph is fixed at L256, so latency changes come mainly from encode/decode
overhead and padding, not context length. This tool records both so the claim
is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from policies.onnx_kev import OnnxKevPolicy  # noqa: E402


def load(path: Path) -> list[dict]:
    states = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "SCENARIO_JSON:"
        if marker in line:
            states.append(json.loads(line.split(marker, 1)[1]))
    return states


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--model", default=r"C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx")
    parser.add_argument("--tokenizer", default=r"C:\LANESHIFT2D-Kev-model\tokenizer.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--modes", default="scene,lane,compact")
    args = parser.parse_args()

    states = load(args.snapshots)
    rows = []
    for mode in args.modes.split(","):
        policy = OnnxKevPolicy(
            args.model, args.tokenizer, length=256,
            intra_threads=args.threads, prompt_mode=mode,
        )
        await policy.start()
        for state in states:
            for _ in range(args.repeat):
                started = time.perf_counter()
                decision = policy.choose_action(state)
                encoding = decision.response_payload["encoding"]
                rows.append({
                    "mode": mode,
                    "scenario": state.get("controlled_scenario", "?"),
                    "visible": sum(1 for vehicle in state["traffic"] if vehicle.get("on_screen")),
                    "state_tokens": encoding["state_tokens"],
                    "prompt_tokens": encoding["total_tokens"],
                    "action": decision.action,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                })
                print(json.dumps(rows[-1]), flush=True)
        # Two 0.8B sessions in one process exhaust memory on the 16 GB host.
        policy.session = None
        policy.tok = None
        policy._blob = None
        del policy
        gc.collect()
        await asyncio.sleep(0.5)

    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        latencies = sorted(row["latency_ms"] for row in subset)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(json.dumps({
            "summary": mode,
            "runs": len(subset),
            "mean_state_tokens": round(statistics.mean(row["state_tokens"] for row in subset), 1),
            "max_state_tokens": max(row["state_tokens"] for row in subset),
            "mean_prompt_tokens": round(statistics.mean(row["prompt_tokens"] for row in subset), 1),
            "mean_latency_ms": round(statistics.mean(latencies), 1),
            "p95_latency_ms": p95,
        }), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
