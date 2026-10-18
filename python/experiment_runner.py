"""Run seeded LaneShift episodes under multiple Python policies."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def summarize_policy(policy, episodes, controller):
    count = len(episodes)
    mean = lambda key: sum(float(row.get(key, 0.0)) for row in episodes) / count if count else 0.0
    collisions = sum(
        row.get("termination_reason") == "crash" or int(row.get("collision_count", 0)) > 0
        for row in episodes
    )
    completed = sum(
        row.get("termination_reason") in {"timeout", "completed", "episode_complete"}
        for row in episodes
    )
    actions = controller.get("applied_actions", controller.get("actions", {}))
    actions_applied = sum(int(value) for value in actions.values())
    decisions_requested = int(controller.get("decisions", 0))
    decisions_rejected = max(
        int(controller.get("rejected_actions", 0)) + int(controller.get("protocol_rejections", 0)),
        decisions_requested - actions_applied,
    )
    return {
        "policy": policy,
        "scenario_seeds": ";".join(str(row.get("scenario_seed", "")) for row in episodes),
        "episode_count": count,
        "mean_score": mean("score"),
        "mean_distance": mean("distance"),
        "mean_survival_time": mean("survival_time"),
        "collision_frequency": collisions / count if count else 0.0,
        "completion_frequency": completed / count if count else 0.0,
        "runs_ending_outside_start_lane": sum(int(row.get("final_lane", 1)) != 1 for row in episodes),
        "final_lanes": ";".join(str(row.get("final_lane", "")) for row in episodes),
        "termination_reasons": json.dumps(_count_reasons(episodes), sort_keys=True),
        "actions_taken": actions_applied,
        "action_counts": json.dumps(actions, sort_keys=True),
        "action_decisions_requested": decisions_requested,
        "action_decisions_rejected": decisions_rejected,
        "decision_latency_ms_mean": float(controller.get("decision_latency_ms_mean", 0.0)),
        "decision_latency_ms_p50": float(controller.get("decision_latency_ms_p50", 0.0)),
        "decision_latency_ms_p95": float(controller.get("decision_latency_ms_p95", 0.0)),
        "decision_latency_ms_p99": float(controller.get("decision_latency_ms_p99", 0.0)),
        "ack_rtt_ms_mean": float(controller.get("ack_rtt_ms_mean", 0.0)),
        "controller_failures": int(controller.get("controller_failures", 0)),
        "invalid_commands": int(controller.get("invalid_actions", 0)) + int(controller.get("invalid_states", 0)) + decisions_rejected,
    }


def _count_reasons(episodes):
    counts = {}
    for episode in episodes:
        reason = episode.get("termination_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _wait_for_server(process, port, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("controller server exited before accepting connections")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"controller server did not open 127.0.0.1:{port}")


def _parse_episode_summaries(output):
    summaries = []
    for line in output.splitlines():
        prefix = "LANESHIFT_EPISODE "
        if line.startswith(prefix):
            try:
                summaries.append(json.loads(line[len(prefix):]))
            except json.JSONDecodeError:
                continue
    return summaries


def run_experiments(godot, project, policies, seeds, episodes_per_seed, duration, port, output_path):
    project = Path(project).resolve()
    output_path = Path(output_path).resolve()
    python_dir = project / "python"
    expected = len(seeds) * episodes_per_seed
    rows = []
    for policy in policies:
        metrics_path = output_path.with_name(f"{output_path.stem}.{policy}.metrics.json")
        server_command = [
            sys.executable, str(python_dir / "server.py"),
            "--policy", policy,
            "--port", str(port),
            "--expected-episodes", str(expected),
            "--metrics-out", str(metrics_path),
        ]
        server = subprocess.Popen(
            server_command,
            cwd=python_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_server(server, port)
            godot_command = [
                godot,
                "--headless",
                "--path", str(project),
                "--",
                "--auto-run",
                f"--seeds={','.join(map(str, seeds))}",
                f"--episodes={episodes_per_seed}",
                f"--episode-duration={duration}",
                f"--ws-port={port}",
            ]
            timeout = max(60.0, expected * duration * 2.5 + 30.0)
            run = subprocess.run(
                godot_command,
                cwd=project,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined_output = run.stdout + "\n" + run.stderr
            if run.returncode != 0:
                raise RuntimeError(f"Godot failed for {policy}:\n{combined_output[-5000:]}")
            episodes = _parse_episode_summaries(combined_output)
            if len(episodes) != expected:
                raise RuntimeError(
                    f"expected {expected} episode summaries for {policy}, got {len(episodes)}.\n"
                    f"Godot output tail:\n{combined_output[-3000:]}"
                )
            try:
                server.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=5.0)
            server_output = server.communicate(timeout=5.0)[0] or ""
            if server.returncode != 0:
                raise RuntimeError(f"controller server failed for {policy}:\n{server_output[-3000:]}")
            controller = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            if not controller:
                for line in server_output.splitlines():
                    if line.startswith("LANESHIFT_METRICS "):
                        controller = json.loads(line.removeprefix("LANESHIFT_METRICS "))
                        break
            if not controller or int(controller.get("decisions", 0)) == 0:
                raise RuntimeError(f"controller server recorded no decisions for {policy}:\n{server_output[-3000:]}")
            rows.append(summarize_policy(policy, episodes, controller))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default="godot", help="Godot executable path")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--policies", default="rule_based,dummy")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("comparison.csv"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not policies or not seeds or args.episodes_per_seed < 1 or args.duration <= 0:
        raise SystemExit("policies and seeds are required; durations and episode counts must be positive")
    rows = run_experiments(
        args.godot, args.project, policies, seeds, args.episodes_per_seed,
        args.duration, args.port, args.output,
    )
    print(json.dumps(rows, indent=2))
    print(f"Comparison written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
