"""Probe Kev on controlled LANESHIFT decision states.

This is a diagnostic, not a benchmark. It compares the compact numeric prompt
with the lane-oriented natural-language prompt on the same states.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from policies.onnx_kev import OnnxKevPolicy  # noqa: E402


def state(player_lane: int, traffic: list[dict], legal=None) -> dict:
    return {
        "episode_id": "probe",
        "sim_time": 1.0,
        "player": {
            "lane": player_lane,
            "target_lane": player_lane,
            "lane_change_in_progress": False,
            "speed": 20.0,
            "braking": False,
        },
        "traffic": traffic,
        "legal_actions": legal or ["LEFT", "STAY", "RIGHT", "BRAKE"],
    }


def v(id_: str, lane: int, distance: float, relative_speed: float) -> dict:
    return {
        "id": id_,
        "lane": lane,
        "relative_distance_z": distance,
        "relative_speed_z": relative_speed,
    }


SCENARIOS = [
    ("current lane safe; both changes risky", state(1, [
        v("left", 0, 10, 8), v("right", 2, 9, 7),
    ]), "STAY"),
    ("left lane clearly safer", state(1, [
        v("center", 1, 14, 5), v("left", 0, 40, 1), v("right", 2, 9, 7),
    ]), "LEFT"),
    ("right lane clearly safer", state(1, [
        v("center", 1, 14, 5), v("left", 0, 8, 7), v("right", 2, 40, 1),
    ]), "RIGHT"),
    ("both adjacent lanes blocked; brake", state(1, [
        v("center", 1, 10, 6), v("left", 0, 7, 5), v("right", 2, 6, 4),
    ]), "BRAKE"),
    ("current lane unsafe; no clearly safe lane", state(1, [
        v("center", 1, 8, 7), v("left", 0, 12, 8), v("right", 2, 11, 6),
    ]), "BRAKE"),
    ("left lane has a fast car behind", state(1, [
        v("center", 1, 20, 2), v("left", 0, -10, -8), v("right", 2, 35, 1),
    ]), "STAY"),
    ("center front vehicle slowing; lanes clear", state(1, [
        v("center", 1, 20, 12), v("left", 0, 45, 1), v("right", 2, 40, 1),
    ]), "LEFT"),
    ("open road", state(1, []), "STAY"),
]


async def main() -> None:
    model = Path(r"C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx")
    tokenizer = Path(r"C:\LANESHIFT2D-Kev-model\tokenizer.json")
    policies = {
        "compact": OnnxKevPolicy(str(model), str(tokenizer), length=256, intra_threads=8, prompt_mode="compact"),
        "lane": OnnxKevPolicy(str(model), str(tokenizer), length=256, intra_threads=8, prompt_mode="lane"),
        "scene": OnnxKevPolicy(str(model), str(tokenizer), length=256, intra_threads=8, prompt_mode="scene"),
    }
    for name, policy in policies.items():
        await policy.start()
        print(f"\n=== {name} ===", flush=True)
        for label, probe_state, expected in SCENARIOS:
            decision = policy.choose_action(probe_state)
            probs = {k: round(v, 3) for k, v in decision.probabilities.items()}
            print(json.dumps({
                "scenario": label,
                "expected_safe_action": expected,
                "chosen": decision.action,
                "probabilities": probs,
                "state_tokens": decision.response_payload["encoding"]["state_tokens"],
                "latency_ms": round(decision.model_latency_ms, 1),
            }), flush=True)
        policy.session = None
        policy.tok = None
        policy._blob = None


if __name__ == "__main__":
    asyncio.run(main())
