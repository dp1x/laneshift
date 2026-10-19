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


BASELINE_POLICIES = {"dummy", "rule_based"}
MODEL_LABEL_SOURCES = {"original_model", "fine_tuned_model"}


def build_policy_spec(name, model_registry):
    """Describe how one controller policy must be launched.

    Baselines are self-describing. Model policies require an explicit registry
    entry that names the checkpoint, so no run can claim an identity that was
    never verified against a running service.
    """
    if name in BASELINE_POLICIES:
        return {"name": name, "kind": "baseline", "label_source": None}
    config = (model_registry or {}).get(name)
    if not isinstance(config, dict):
        raise ValueError(
            f"policy {name!r} is not a baseline and has no model registry entry; "
            "supply --model-registry with base_url, model, expected_run and label_source"
        )
    expected_run = str(config.get("expected_run", "")).strip()
    label_source = str(config.get("label_source", "")).strip()
    if not expected_run:
        raise ValueError(f"model policy {name!r} must declare expected_run so the loaded checkpoint can be verified")
    if label_source not in MODEL_LABEL_SOURCES:
        raise ValueError(
            f"model policy {name!r} must declare label_source as one of {sorted(MODEL_LABEL_SOURCES)}"
        )
    return {
        "name": name,
        "kind": "model",
        "label_source": label_source,
        "expected_run": expected_run,
        "base_url": str(config.get("base_url", "http://127.0.0.1:8009")),
        "model": str(config.get("model", "kev-latest")),
        "timeout": float(config.get("timeout", 0.4)),
        "api_key_env": str(config.get("api_key_env", "KEV_API_KEY")),
    }


def assert_models_loaded(rows):
    """Refuse to publish comparison results for checkpoints that never loaded."""
    unloaded = [
        row for row in rows
        if row.get("model_loaded") == "no"
    ]
    if unloaded:
        names = ", ".join(sorted({row.get("policy", "unknown") for row in unloaded}))
        raise RuntimeError(
            f"refusing to publish results: no checkpoint was verified as loaded for: {names}"
        )
    return True


def summarize_policy(policy, episodes, controller, split=None):
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
    model_id = str(controller.get("model_id", "") or "")
    model_alias = str(controller.get("model_alias", "") or "")
    policy_kind = str(controller.get("policy_kind", "") or "")
    if model_id:
        model_loaded = "yes"
    elif policy_kind == "model":
        model_loaded = "no"
    else:
        model_loaded = "n/a"
    return {
        "policy": policy,
        "split": split or "unsplit",
        "model_id": model_id,
        "model_alias": model_alias,
        "model_loaded": model_loaded,
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
        "model_latency_ms_mean": float(controller.get("model_latency_ms_mean", 0.0)),
        "model_latency_ms_p50": float(controller.get("model_latency_ms_p50", 0.0)),
        "model_latency_ms_p95": float(controller.get("model_latency_ms_p95", 0.0)),
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


def run_experiments(godot, project, policies, seeds, episodes_per_seed, duration, port, output_path,
                    model_registry=None, seed_splits=None):
    project = Path(project).resolve()
    output_path = Path(output_path).resolve()
    python_dir = project / "python"
    model_registry = model_registry or {}
    specs = [build_policy_spec(name, model_registry) for name in policies]
    if seed_splits:
        phases = [(name, list(values)) for name, values in seed_splits.items() if values]
    else:
        phases = [("unsplit", list(seeds))]
    expected = sum(len(values) for _name, values in phases) * episodes_per_seed
    rows = []
    episode_rows = []
    for spec in specs:
        policy = spec["name"]
        metrics_path = output_path.with_name(f"{output_path.stem}.{policy}.metrics.json")
        server_command = [
            sys.executable, str(python_dir / "server.py"),
            "--policy", policy,
            "--port", str(port),
            "--expected-episodes", str(expected),
            "--metrics-out", str(metrics_path),
        ]
        if spec["kind"] == "model":
            server_command += [
                "--kev-config", str(metrics_path.with_suffix(".registry.json")),
                "--decision-timeout", str(max(spec["timeout"] + 0.05, 0.5)),
            ]
            registry_path = metrics_path.with_suffix(".registry.json")
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps({
                policy: {
                    "base_url": spec["base_url"],
                    "model": spec["model"],
                    "expected_run": spec["expected_run"],
                    "label_source": spec["label_source"],
                    "timeout": spec["timeout"],
                    "api_key_env": spec["api_key_env"],
                }
            }, indent=2) + "\n", encoding="utf-8")
        server = subprocess.Popen(
            server_command,
            cwd=python_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_server(server, port)
            for split_name, split_seeds in phases:
                godot_command = [
                    godot,
                    "--headless",
                    "--path", str(project),
                    "--",
                    "--auto-run",
                    f"--seeds={','.join(map(str, split_seeds))}",
                    f"--episodes={episodes_per_seed}",
                    f"--episode-duration={duration}",
                    f"--ws-port={port}",
                ]
                timeout = max(60.0, len(split_seeds) * episodes_per_seed * duration * 2.5 + 30.0)
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
                expected_here = len(split_seeds) * episodes_per_seed
                if len(episodes) != expected_here:
                    raise RuntimeError(
                        f"expected {expected_here} episode summaries for {policy} split {split_name}, got {len(episodes)}.\n"
                        f"Godot output tail:\n{combined_output[-3000:]}"
                    )
                for episode in episodes:
                    episode_rows.append({"policy": policy, "split": split_name, **episode})
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
            for split_name, split_seeds in phases:
                split_episodes = [row for row in episode_rows if row["policy"] == policy and row["split"] == split_name]
                rows.append(summarize_policy(policy, split_episodes, controller, split_name))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    assert_models_loaded(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    episodes_path = output_path.with_name(output_path.stem + ".episodes.jsonl")
    with episodes_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in episode_rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default="godot", help="Godot executable path")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--policies", default="rule_based,dummy")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--dev-seeds", default="", help="exploratory seeds; results are labelled dev")
    parser.add_argument("--test-seeds", default="", help="held-out seeds; results are labelled test")
    parser.add_argument("--model-registry", type=Path, help="JSON registry of model policies with base_url, model, expected_run, label_source")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("comparison.csv"))
    return parser.parse_args(argv)


def _parse_seed_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv=None):
    args = parse_args(argv)
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    seeds = _parse_seed_list(args.seeds)
    dev_seeds = _parse_seed_list(args.dev_seeds)
    test_seeds = _parse_seed_list(args.test_seeds)
    if not policies or not seeds or args.episodes_per_seed < 1 or args.duration <= 0:
        raise SystemExit("policies and seeds are required; durations and episode counts must be positive")
    if not dev_seeds and not test_seeds:
        dev_seeds, test_seeds = seeds, []
    model_registry = {}
    if args.model_registry:
        try:
            model_registry = json.loads(Path(args.model_registry).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read model registry {args.model_registry}: {exc}")
        if isinstance(model_registry, dict) and isinstance(model_registry.get("models"), dict):
            model_registry = model_registry["models"]
    rows = run_experiments(
        args.godot, args.project, policies, seeds, args.episodes_per_seed,
        args.duration, args.port, args.output,
        model_registry=model_registry,
        seed_splits={"dev": dev_seeds, "test": test_seeds},
    )
    print(json.dumps(rows, indent=2))
    print(f"Comparison written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
