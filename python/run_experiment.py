"""Run one policy across seeded episodes and print a result table.

Examples (from the repository root, development mode):

    python python/run_experiment.py --policy rule_based --seeds 1,2,3 --episodes 1 --duration 30
    python python/run_experiment.py --policy onnx_kev --seeds 11,12 --episodes 1 --duration 20 \
        --onnx-model C:\\Users\\Dhane\\dev\\kev-onnx-backup\\kev07b_fp16_L256.onnx \
        --onnx-tokenizer C:\\Users\\Dhane\\.cache\\huggingface\\hub\\models--Qwen--Qwen3.5-0.8B-Base\\snapshots\\dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68\\tokenizer.json

Packaged mode:

    python run_experiment.py --policy onnx_kev --game "R:\\LANESHIFT2D-Kev\\LANESHIFT.exe" \
        --controller "R:\\LANESHIFT2D-Kev\\runtime\\py\\python.exe R:\\LANESHIFT2D-Kev\\controller\\server.py" \
        --onnx-model "R:\\LANESHIFT2D-Kev\\runtime\\kev07b_fp16_L256.onnx" \
        --onnx-tokenizer "R:\\LANESHIFT2D-Kev\\runtime\\tokenizer.json"

The script starts the controller once, runs the game's built-in auto-run across the
requested seeds, collects LANESHIFT_EPISODE lines, and writes a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = ROOT / "python"


def _reader(pipe, sink: list[str], ready: threading.Event | None = None, ready_text: str = ""):
    for line in iter(pipe.readline, ""):
        sink.append(line)
        if ready is not None and ready_text and ready_text in line:
            ready.set()
    pipe.close()


def _start(cmd: list[str], cwd: Path) -> tuple[subprocess.Popen, list[str], threading.Event]:
    lines: list[str] = []
    ready = threading.Event()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_reader, args=(proc.stdout, lines, ready, "listening"), daemon=True).start()
    return proc, lines, ready


def _episodes(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        marker = "LANESHIFT_EPISODE "
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return out


def _summary(episodes: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = [float(e.get(key, 0.0)) for e in episodes]
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.fmean(values), 2),
            "std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "episodes": len(episodes),
        "score": stats("score"),
        "distance": stats("distance"),
        "survival_time": stats("survival_time"),
        "collisions": stats("collision_count"),
        "overtaken": stats("overtaken_count"),
        "crashes": sum(1 for e in episodes if e.get("termination_reason") == "crash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run seeded LANESHIFT episodes for one policy.")
    parser.add_argument("--policy", default="rule_based")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per seed")
    parser.add_argument("--duration", type=float, default=30.0, help="simulated seconds per episode")
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--countdown", type=float, default=0.0, help="0 for experiments")
    parser.add_argument("--out", default="experiments")
    parser.add_argument("--game", default="godot", help="game executable, or 'godot' for development")
    parser.add_argument("--controller", default="", help="controller command; default uses this repository")
    parser.add_argument("--onnx-model", default="")
    parser.add_argument("--onnx-tokenizer", default="")
    parser.add_argument("--onnx-length", type=int, default=256)
    parser.add_argument("--onnx-threads", type=int, default=8)
    parser.add_argument("--onnx-arena", action="store_true")
    parser.add_argument("--decision-timeout", type=float, default=10.0)
    parser.add_argument("--max-state-age", type=float, default=10.0)
    parser.add_argument("--controller-timeout", type=float, default=30.0)
    parser.add_argument("--game-timeout", type=float, default=0.0)
    args = parser.parse_args()

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    session_dir = Path(args.out) / f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{args.policy}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.controller:
        controller_cmd = shlex.split(args.controller, posix=False)
        if controller_cmd and controller_cmd[0].lower() in ("python", "python3"):
            controller_cmd[0] = sys.executable
        controller_cwd = Path(controller_cmd[0]).resolve().parent
    else:
        controller_cmd = [
            sys.executable, str(PYTHON_DIR / "server.py"), "--policy", args.policy,
            "--port", str(args.port),
            "--decision-timeout", str(args.decision_timeout),
            "--max-state-age", str(args.max_state_age),
        ]
        if args.policy == "onnx_kev":
            if not args.onnx_model or not args.onnx_tokenizer:
                parser.error("--policy onnx_kev requires --onnx-model and --onnx-tokenizer")
            controller_cmd += [
                "--onnx-model", args.onnx_model,
                "--onnx-tokenizer", args.onnx_tokenizer,
                "--onnx-length", str(args.onnx_length),
                "--onnx-threads", str(args.onnx_threads),
            ]
            if args.onnx_arena:
                controller_cmd.append("--onnx-arena")
        controller_cwd = PYTHON_DIR

    controller, controller_lines, ready = _start(controller_cmd, controller_cwd)
    try:
        if not ready.wait(args.controller_timeout):
            print("controller did not report listening; output follows:", file=sys.stderr)
            print("".join(controller_lines[-20:]), file=sys.stderr)
            return 2
        print(f"controller ready: {args.policy} on ws://127.0.0.1:{args.port}", flush=True)

        if args.game.lower() == "godot":
            game_cmd = ["godot", "--headless", "--path", str(ROOT), "--"]
        else:
            game_cmd = [args.game, "--headless", "--"]
        game_cmd += [
            "--auto-run",
            f"--seeds={','.join(str(s) for s in seeds)}",
            f"--episodes={args.episodes}",
            f"--episode-duration={args.duration}",
            f"--traffic-density={args.density}",
            f"--ws-port={args.port}",
            "--external-timeout-ms=12000",
            "--max-stale-ticks=6000",
            f"--countdown={args.countdown}",
            "--sim-speed-scale=0.5" if args.policy in ("onnx_kev", "kev") else "--sim-speed-scale=1.0",
        ]
        print("game:", " ".join(game_cmd), flush=True)
        game, game_lines, _ = _start(game_cmd, ROOT)
        try:
            game.wait(timeout=args.game_timeout if args.game_timeout > 0 else None)
        except subprocess.TimeoutExpired:
            game.terminate()

        episodes = _episodes(game_lines)
        summary = _summary(episodes)
        (session_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8")
        (session_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report = [
            f"policy: {args.policy}",
            f"seeds: {seeds}",
            f"episodes: {summary['episodes']}",
            f"score: {summary['score']}",
            f"distance: {summary['distance']}",
            f"survival: {summary['survival_time']}",
            f"collisions: {summary['collisions']}",
            f"overtaken: {summary['overtaken']}",
            f"crashes: {summary['crashes']}",
        ]
        (session_dir / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report), flush=True)
        if not episodes:
            print("no LANESHIFT_EPISODE lines were produced", file=sys.stderr)
            print("".join(game_lines[-30:]), file=sys.stderr)
            return 3
        return 0
    finally:
        controller.terminate()
        try:
            controller.wait(timeout=5)
        except subprocess.TimeoutExpired:
            controller.kill()


if __name__ == "__main__":
    raise SystemExit(main())
