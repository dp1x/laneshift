"""Compare engine visibility, the old lane prompt, and the v2 scene prompt.

Input is the output of tests/controlled_observation_probe.gd: one JSON object
per SCENARIO_JSON line. The tool checks that every vehicle the renderer draws
appears exactly once in the scene observation, that off-screen vehicles are in
the separately labelled list, and reports the vehicles the lane prompt omits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from kev_state import (  # noqa: E402
    lane_observation,
    render_lane_state,
    render_scene_state,
    scene_observation,
)


def load(path: Path) -> list[dict]:
    states = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "SCENARIO_JSON:"
        if marker in line:
            states.append(json.loads(line.split(marker, 1)[1]))
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--examples", action="store_true", help="print before/after prompts")
    args = parser.parse_args()

    failures = 0
    for state in load(args.snapshots):
        name = state.get("controlled_scenario", "?")
        traffic = state["traffic"]
        visible = [vehicle for vehicle in traffic if vehicle.get("on_screen")]
        off_visible = [vehicle for vehicle in traffic if not vehicle.get("on_screen")]
        observation = scene_observation(state)
        scene_on = observation["on_screen"]
        scene_off = observation["off_screen"]
        ids_on = [row["vehicle_id"] for row in scene_on]
        ids_off = [row["vehicle_id"] for row in scene_off]
        visible_ids = {vehicle["id"] for vehicle in visible}
        failures += len(visible_ids.symmetric_difference(ids_on))
        failures += len(set(ids_on)) != len(ids_on)
        failures += len(set(ids_off)) != len(ids_off)
        failures += len(set(ids_on) & set(ids_off))
        missing_off = {vehicle["id"] for vehicle in off_visible
                       if -15.0 <= vehicle["relative_distance_z"] <= 160.0} - set(ids_off)
        failures += len(missing_off)

        lane = lane_observation(state)
        lane_records = sum(
            1 for row in lane["lanes"] for key in ("ahead", "behind") if row.get(key)
        )
        scene_text = render_scene_state(observation)
        lane_text = render_lane_state(lane)
        print(json.dumps({
            "scenario": name,
            "engine_active": len(traffic),
            "renderer_visible": len(visible),
            "engine_offscreen": len(off_visible),
            "kev_on_screen": len(scene_on),
            "kev_off_screen_relevant": len(scene_off),
            "kev_beyond_range": observation["visibility"]["off_screen_beyond_range_count"],
            "visible_ids": sorted(visible_ids),
            "kev_on_ids": sorted(ids_on),
            "lane_prompt_vehicle_records": lane_records,
            "lane_prompt_omits_at_least": max(0, len(visible) - lane_records),
            "lane_prompt_chars": len(lane_text),
            "scene_prompt_chars": len(scene_text),
        }), flush=True)
        if args.examples:
            print("--- LANE (before) ---")
            print(lane_text)
            print("--- SCENE (after) ---")
            print(scene_text)
            print()

    if failures:
        print(f"FAIL: {failures} visibility/identity mismatches", file=sys.stderr)
        raise SystemExit(1)
    print("SCENE_AUDIT_PASS")


if __name__ == "__main__":
    main()
