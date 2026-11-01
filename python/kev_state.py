"""The authoritative state projection sent to a Kev System One model.

One projection is used for both inference and training so the model is always
shown the same view of a game state, and so a training example matches what
the model saw when it was asked to decide.

The projection is intentionally small: a decision needs the player's lane and
speed, the nearby traffic with relative gaps and closing speeds, and the
actions the simulation currently allows. Absolute world coordinates are left
out because they do not change the decision.
"""

from __future__ import annotations

from typing import Any

ACTION_CRITERIA = {
    "LEFT": "Move one lane to the left.",
    "STAY": "Remain in the current lane.",
    "RIGHT": "Move one lane to the right.",
    "BRAKE": "Reduce speed using the brake.",
}

LANE_NAMES = ("LEFT", "CENTER", "RIGHT")

KEV_LANE_INSTRUCTIONS = (
    "Choose the safest legal action. Avoid collisions. Check front and rear "
    "clearance before changing lanes. Stay only when the current lane is safe. "
    "Brake when no lane change is safe. Choose exactly one legal option."
)

ACTION_CRITERIA_LANE = {
    "LEFT": "Move left only if the left lane has safe front and rear clearance.",
    "STAY": "Stay only if the current lane is safe and changing lanes adds risk.",
    "RIGHT": "Move right only if the right lane has safe front and rear clearance.",
    "BRAKE": "Slow down when no lane change is safe.",
}


def _movement(gap: float, relative_speed: float) -> tuple[str, float | None]:
    """Describe relative motion using the simulation's sign convention.

    relative_distance_z = vehicle.z - player.z: positive is ahead.
    relative_speed_z = player.speed - vehicle.speed: positive means the player
    is closing an ahead vehicle, and negative means a behind vehicle is
    approaching the player.
    """
    if gap > 0:
        if relative_speed > 0.01:
            return "closing", round(gap / relative_speed, 2)
        if relative_speed < -0.01:
            return "opening", None
        return "same speed", None
    if relative_speed < -0.01:
        return "approaching from behind", round(abs(gap) / abs(relative_speed), 2)
    if relative_speed > 0.01:
        return "falling behind", None
    return "same speed", None


def lane_observation(state: dict[str, Any]) -> dict[str, Any]:
    """A lane-oriented observation for a typed driving decision.

    Unlike compact_kev_state, this keeps the nearest front and rear vehicle in
    every lane, so a vehicle behind in a target lane cannot be discarded by a
    global nearest-three cap.
    """
    player = state.get("player")
    traffic = state.get("traffic")
    legal_actions = state.get("legal_actions")
    if not isinstance(player, dict) or not isinstance(traffic, list):
        raise ValueError("decision state is missing structured player or traffic data")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or "STAY" not in legal_actions
        or any(not isinstance(action, str) or action not in ACTION_CRITERIA for action in legal_actions)
        or len(set(legal_actions)) != len(legal_actions)
    ):
        raise ValueError("decision state has no valid authoritative legal_actions set")

    lanes = []
    for lane in range(3):
        ahead = None
        behind = None
        for vehicle in traffic:
            if not isinstance(vehicle, dict) or vehicle.get("lane") != lane:
                continue
            gap = vehicle.get("relative_distance_z")
            if not isinstance(gap, (int, float)) or isinstance(gap, bool):
                continue
            gap = float(gap)
            if gap > 0 and (ahead is None or gap < ahead[0]):
                ahead = (gap, vehicle)
            elif gap < 0 and (behind is None or gap > behind[0]):
                behind = (gap, vehicle)

        def describe(entry):
            if entry is None:
                return None
            gap, vehicle = entry
            relative_speed = vehicle.get("relative_speed_z")
            relative_speed = float(relative_speed) if isinstance(relative_speed, (int, float)) and not isinstance(relative_speed, bool) else 0.0
            movement, ttc = _movement(gap, relative_speed)
            return {
                "distance_m": round(abs(gap), 2),
                "relative_speed_mps": round(relative_speed, 2),
                "movement": movement,
                "time_to_contact_s": ttc,
            }

        lanes.append({
            "lane": LANE_NAMES[lane],
            "ahead": describe(ahead),
            "behind": describe(behind),
        })

    target_lane = player.get("target_lane", player.get("lane", 1))
    return {
        "player": {
            "lane": LANE_NAMES[int(player.get("lane", 1))] if player.get("lane") in (0, 1, 2) else "UNKNOWN",
            "target_lane": LANE_NAMES[int(target_lane)] if target_lane in (0, 1, 2) else "UNKNOWN",
            "lane_change_in_progress": bool(player.get("lane_change_in_progress", False)),
            "lane_change_cooldown_s": round(float(player.get("lane_change_cooldown", 0.0)), 2),
            "previous_action": str(player.get("current_action", "STAY")).upper(),
            "speed_mps": round(float(player.get("speed", 0.0)), 2),
            "braking": bool(player.get("braking", False)),
        },
        "lanes": lanes,
        "legal_actions": list(legal_actions),
    }


def render_lane_state(observation: dict[str, Any]) -> str:
    """Render lane_observation as a compact natural-language decision document."""
    player = observation["player"]
    lines = [
        "LANESHIFT DECISION STATE",
        f"Player: lane {player['lane']}, target {player['target_lane']}, speed {player['speed_mps']:.1f} m/s, braking {str(player['braking']).lower()}.",
        f"Lane change in progress: {str(player['lane_change_in_progress']).lower()}; "
        f"cooldown {player['lane_change_cooldown_s']:.1f} s; previous action: {player['previous_action']}.",
    ]
    for lane in observation["lanes"]:
        lines.append(f"{lane['lane']} LANE:")
        ahead = lane.get("ahead")
        behind = lane.get("behind")
        if ahead:
            ttc = f", time to contact {ahead['time_to_contact_s']:.1f} s" if ahead.get("time_to_contact_s") is not None else ""
            lines.append(
                f"  Nearest vehicle ahead: {ahead['distance_m']:.1f} m, {ahead['movement']} "
                f"(relative speed {ahead['relative_speed_mps']:+.1f} m/s){ttc}."
            )
        else:
            lines.append("  No vehicle ahead within observation range.")
        if behind:
            ttc = f", time to contact {behind['time_to_contact_s']:.1f} s" if behind.get("time_to_contact_s") is not None else ""
            lines.append(
                f"  Nearest vehicle behind: {behind['distance_m']:.1f} m, {behind['movement']} "
                f"(relative speed {behind['relative_speed_mps']:+.1f} m/s){ttc}."
            )
        else:
            lines.append("  No vehicle behind within observation range.")
    lines.append("Legal actions: " + ", ".join(observation["legal_actions"]) + ".")
    return "\n".join(lines)


def _round(value: Any, digits: int = 2) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def compact_kev_state(state: dict[str, Any], max_vehicles: int = 3) -> dict[str, Any]:
    """Project a game state snapshot into the shape the model receives.

    Traffic is ordered by absolute distance and capped so the nearest vehicle
    is always present, even when the prompt is truncated to fit a small graph.
    """
    player = state.get("player")
    traffic = state.get("traffic")
    legal_actions = state.get("legal_actions")
    if not isinstance(player, dict) or not isinstance(traffic, list):
        raise ValueError("decision state is missing structured player or traffic data")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or "STAY" not in legal_actions
        or any(not isinstance(action, str) or action not in ACTION_CRITERIA for action in legal_actions)
        or len(set(legal_actions)) != len(legal_actions)
    ):
        raise ValueError("decision state has no valid authoritative legal_actions set")

    candidates = []
    player_lane = player.get("lane")
    for vehicle in traffic:
        if not isinstance(vehicle, dict):
            continue
        distance = _round(vehicle.get("relative_distance_z"))
        closing = _round(vehicle.get("relative_speed_z"))
        same_lane = vehicle.get("lane") == player_lane
        ttc = None
        if same_lane and isinstance(distance, float) and isinstance(closing, float) and distance > 0.0 and closing > 0.0:
            ttc = round(distance / closing, 2)
        candidates.append({
            "id": vehicle.get("id"),
            "lane": vehicle.get("lane"),
            "relative_distance": distance,
            "relative_speed": closing,
            "in_player_lane": same_lane,
            "time_to_collision": ttc,
            "_sort_distance": abs(distance) if isinstance(distance, float) else 1_000_000.0,
        })
    candidates.sort(key=lambda item: item["_sort_distance"])
    vehicles = []
    for item in candidates[: max(1, int(max_vehicles))]:
        item.pop("_sort_distance", None)
        vehicles.append(item)

    return {
        "episode_id": state.get("episode_id"),
        "sim_time_s": _round(state.get("sim_time")),
        "player": {
            "lane": player.get("lane"),
            "target_lane": player.get("target_lane"),
            "lane_change_in_progress": bool(player.get("lane_change_in_progress", False)),
            "speed": _round(player.get("speed")),
        },
        "traffic": vehicles,
        "legal_actions": list(legal_actions),
    }
