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


def _round(value: Any, digits: int = 2) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def compact_kev_state(state: dict[str, Any]) -> dict[str, Any]:
    """Project a game state snapshot into the shape the model receives."""
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

    vehicles = []
    for vehicle in traffic:
        if not isinstance(vehicle, dict):
            continue
        vehicles.append({
            "id": vehicle.get("id"),
            "lane": vehicle.get("lane"),
            "relative_distance": _round(vehicle.get("relative_distance_z")),
            "relative_speed": _round(vehicle.get("relative_speed_z")),
        })

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
