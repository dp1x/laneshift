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

KEV_COMPACT_INSTRUCTIONS = (
    "Choose the safest useful next action from the currently legal choices. "
    "Only select an option present in criteria."
)

# The L256 graph leaves 154-168 tokens for the scene state, depending on the
# option branch. Nine vehicles plus the headers fit with margin; the renderer
# keeps the nearest ahead/behind vehicle in every lane and records how many
# relevant vehicles were omitted instead of silently dropping them.
SCENE_VEHICLE_LIMIT = 9

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


OBSERVATION_SCHEMA_VERSION = 2
# The renderer culls a vehicle when z_rel < -12 or z_rel > 110
# (scripts/main.gd: _draw_vehicle). These bounds are the exact draw gate.
VISIBLE_BEHIND_M = 12.0
VISIBLE_AHEAD_M = 110.0
# Active traffic is removed at z_rel < -15 (scripts/sim/simulation.gd: step),
# so -15 is the complete off-screen behind range. Off-screen ahead traffic is
# included out to a stated relevance horizon; beyond it a vehicle cannot
# affect the next manoeuvre.
RELEVANT_BEHIND_M = 15.0
RELEVANT_AHEAD_M = 160.0
LANE_CHANGE_SPEED_MPS = 15.0
PLAYER_ACCEL_MPS2 = 3.0
PLAYER_BRAKE_MPS2 = -14.0

KEV_SCENE_INSTRUCTIONS = (
    "Choose the safest legal action. Before changing lanes, check every vehicle "
    "in the destination lane, including traffic behind. Stay only when the "
    "current lane is safe. Brake when no lane change is safe. Choose exactly "
    "one legal option."
)

ACTION_CRITERIA_SCENE = {
    "LEFT": "Move left only with safe front and rear clearance.",
    "STAY": "Stay only if the current lane is safe.",
    "RIGHT": "Move right only with safe front and rear clearance.",
    "BRAKE": "Brake when no lane change is safe.",
}


def _lane_world_x(lane: int) -> float:
    # Simulation LANE_CENTERS = [4.0, 0.0, -4.0].
    return 4.0 - 4.0 * lane


def scene_observation(state: dict[str, Any]) -> dict[str, Any]:
    """A versioned, complete decision snapshot.

    Every vehicle the renderer draws at this snapshot is included. Active
    vehicles outside the draw gate are included separately when they are inside
    the stated relevance horizon, and counted when they are not. The game
    supplies `on_screen`; the fallback repeats the renderer's exact draw gate.
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

    player_speed = float(player.get("speed", 0.0))
    player_lane = int(player.get("lane", 1))
    target_lane = int(player.get("target_lane", player_lane))
    position = player.get("position") if isinstance(player.get("position"), dict) else {}
    player_x = float(position.get("x", _lane_world_x(player_lane)))
    player_z = float(position.get("z", 0.0))
    target_x = _lane_world_x(target_lane)
    remaining_m = abs(target_x - player_x)
    in_progress = bool(player.get("lane_change_in_progress", False))
    progress = max(0.0, min(1.0, 1.0 - remaining_m / 4.0)) if in_progress else 1.0
    remaining_s = remaining_m / LANE_CHANGE_SPEED_MPS if in_progress else 0.0
    direction = "NONE"
    if in_progress and target_lane != player_lane:
        direction = "LEFT" if target_lane < player_lane else "RIGHT"

    on_screen: list[dict[str, Any]] = []
    off_screen: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    excluded = 0
    invalid = 0
    for vehicle in traffic:
        if not isinstance(vehicle, dict):
            invalid += 1
            continue
        gap = vehicle.get("relative_distance_z")
        rel = vehicle.get("relative_speed_z")
        if not isinstance(gap, (int, float)) or isinstance(gap, bool):
            invalid += 1
            continue
        gap = float(gap)
        rel = float(rel) if isinstance(rel, (int, float)) and not isinstance(rel, bool) else 0.0
        visible = bool(vehicle.get("on_screen", -VISIBLE_BEHIND_M <= gap <= VISIBLE_AHEAD_M))
        if not visible and not (-RELEVANT_BEHIND_M <= gap <= RELEVANT_AHEAD_M):
            excluded += 1
            continue

        vehicle_type = str(vehicle.get("type", "car"))
        default_length, default_width = (7.0, 2.4) if vehicle_type == "truck" else (4.5, 1.8)
        length = vehicle.get("length", default_length)
        width = vehicle.get("width", default_width)
        length = float(length) if isinstance(length, (int, float)) and not isinstance(length, bool) else default_length
        width = float(width) if isinstance(width, (int, float)) and not isinstance(width, bool) else default_width
        speed = vehicle.get("speed")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool):
            absolute_speed = float(speed)
        else:
            absolute_speed = player_speed - rel
        # closing_speed_mps is the rate at which the absolute gap shrinks.
        # Positive means the vehicle is getting closer to the player.
        closing = rel if gap > 0.0 else -rel
        ttc = round(abs(gap) / closing, 2) if closing > 0.01 else None
        traffic_acceleration = vehicle.get("acceleration", 0.0)
        traffic_acceleration = (
            float(traffic_acceleration)
            if isinstance(traffic_acceleration, (int, float)) and not isinstance(traffic_acceleration, bool)
            else 0.0
        )
        record = {
            "label": "",
            "vehicle_id": str(vehicle.get("id", "")),
            "lane": LANE_NAMES[int(vehicle.get("lane", 1))] if vehicle.get("lane") in (0, 1, 2) else "UNKNOWN",
            "relative_distance_z": round(gap, 2),
            "relative_speed_z": round(rel, 2),
            "closing_speed_mps": round(closing, 2),
            "absolute_speed_mps": round(absolute_speed, 2),
            "acceleration_mps2": round(traffic_acceleration, 2),
            "type": vehicle_type,
            "length_m": round(length, 2),
            "width_m": round(width, 2),
            "on_screen": visible,
            "ahead": gap >= 0.0,
            "time_to_collision_s": ttc,
        }
        included.append(record)
        (on_screen if visible else off_screen).append(record)

    def order(record):
        return (LANE_NAMES.index(record["lane"]) if record["lane"] in LANE_NAMES else 9,
                record["relative_distance_z"])

    on_screen.sort(key=order)
    off_screen.sort(key=order)
    for index, record in enumerate(on_screen + off_screen, start=1):
        record["label"] = f"V{index}"

    lane_summaries = []
    for lane in range(3):
        records = [record for record in included if record["lane"] == LANE_NAMES[lane]]
        ahead = [record for record in records if record["relative_distance_z"] >= 0.0]
        behind = [record for record in records if record["relative_distance_z"] < 0.0]
        nearest_ahead = min(ahead, key=lambda record: record["relative_distance_z"]) if ahead else None
        nearest_behind = max(behind, key=lambda record: record["relative_distance_z"]) if behind else None

        def summary(record):
            if record is None:
                return None
            return {
                "vehicle_id": record["vehicle_id"],
                "relative_distance_z": record["relative_distance_z"],
                "closing_speed_mps": record["closing_speed_mps"],
                "on_screen": record["on_screen"],
                "time_to_collision_s": record["time_to_collision_s"],
            }

        lane_summaries.append({
            "lane": LANE_NAMES[lane],
            "vehicle_count": len(records),
            "ahead_count": len(ahead),
            "behind_count": len(behind),
            "nearest_ahead": summary(nearest_ahead),
            "nearest_behind": summary(nearest_behind),
        })

    acceleration = player.get("acceleration")
    if not isinstance(acceleration, (int, float)) or isinstance(acceleration, bool):
        acceleration = PLAYER_BRAKE_MPS2 if player.get("braking") else PLAYER_ACCEL_MPS2
    speed_limits = None
    base_speed = player.get("base_speed")
    max_speed = player.get("max_speed")
    if isinstance(base_speed, (int, float)) and isinstance(max_speed, (int, float)):
        speed_limits = {
            "brake_floor_mps": round(float(base_speed) * 0.45, 2),
            "max_mps": round(float(max_speed), 2),
        }
    recent = player.get("recent_actions", [])
    recent_actions = [
        str(action).upper() for action in recent
        if isinstance(action, str) and action.upper() in ACTION_CRITERIA
    ][-8:]
    previous_action = player.get("previous_action", player.get("current_action", "STAY"))
    previous_action = str(previous_action).upper()
    if previous_action not in ACTION_CRITERIA:
        previous_action = "STAY"

    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "player": {
            "lane": LANE_NAMES[player_lane] if player_lane in (0, 1, 2) else "UNKNOWN",
            "target_lane": LANE_NAMES[target_lane] if target_lane in (0, 1, 2) else "UNKNOWN",
            "position": {"x": round(player_x, 2), "z": round(player_z, 2)},
            "speed_mps": round(player_speed, 2),
            "acceleration_mps2": round(float(acceleration), 2),
            "speed_limits_mps": speed_limits,
            "braking": bool(player.get("braking", False)),
            "lane_change_in_progress": in_progress,
            "lane_change_direction": direction,
            "lane_change_progress": round(progress, 2),
            "lane_change_remaining_estimate_s": round(remaining_s, 2),
            "lane_change_cooldown_s": round(float(player.get("lane_change_cooldown", 0.0)), 2),
            "previous_action": previous_action,
            "recent_actions": recent_actions,
        },
        "road": {
            "lane_count": 3,
            "lane_width_m": 4.0,
            "at_left_boundary": player_lane <= 0,
            "at_right_boundary": player_lane >= 2,
        },
        "on_screen": on_screen,
        "off_screen": off_screen,
        "lane_summaries": lane_summaries,
        "legal_actions": list(legal_actions),
        "visibility": {
            "on_screen_rule": "renderer draw gate",
            "on_screen_z_min_m": -VISIBLE_BEHIND_M,
            "on_screen_z_max_m": VISIBLE_AHEAD_M,
            "off_screen_relevant_z_min_m": -RELEVANT_BEHIND_M,
            "off_screen_relevant_z_max_m": RELEVANT_AHEAD_M,
            "active_vehicle_count": len(included) + excluded,
            "on_screen_vehicle_count": len(on_screen),
            "off_screen_relevant_vehicle_count": len(off_screen),
            "off_screen_beyond_range_count": excluded,
            "invalid_vehicle_count": invalid,
        },
        "derived_fields": [
            "closing_speed_mps",
            "time_to_collision_s",
            "lane_change_progress",
            "lane_change_remaining_estimate_s",
            "lane_summaries",
        ],
    }


def select_scene_records(
    observation: dict[str, Any], vehicle_limit: int = SCENE_VEHICLE_LIMIT
) -> tuple[list[dict[str, Any]], int]:
    """Choose the safety-relevant vehicles that fit the fixed prompt.

    The structured observation keeps every vehicle. The prompt keeps the
    nearest ahead and behind vehicle in each lane, then fills remaining slots
    by on-screen status and absolute distance. The returned count says how
    many relevant vehicles were left out; the renderer prints it as ``X<n>``.
    """
    limit = max(1, int(vehicle_limit))
    records = list(observation["on_screen"]) + list(observation["off_screen"])
    if len(records) <= limit:
        return records, 0

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(record: dict[str, Any]) -> None:
        key = id(record)
        if key not in seen:
            seen.add(key)
            selected.append(record)

    for lane in ("LEFT", "CENTER", "RIGHT"):
        lane_records = [record for record in records if record.get("lane") == lane]
        ahead = [record for record in lane_records if record["relative_distance_z"] >= 0.0]
        behind = [record for record in lane_records if record["relative_distance_z"] < 0.0]
        if ahead:
            add(min(ahead, key=lambda record: record["relative_distance_z"]))
        if behind:
            add(max(behind, key=lambda record: record["relative_distance_z"]))

    for record in sorted(records, key=lambda record: (not record["on_screen"], abs(record["relative_distance_z"]))):
        if len(selected) >= limit:
            break
        add(record)

    selected = selected[:limit]
    order = {id(record): index for index, record in enumerate(records)}
    selected.sort(key=lambda record: order[id(record)])
    return selected, len(records) - len(selected)


def scene_prompt_metadata(
    observation: dict[str, Any], vehicle_limit: int = SCENE_VEHICLE_LIMIT
) -> dict[str, int]:
    selected, omitted = select_scene_records(observation, vehicle_limit)
    return {
        "vehicle_limit": int(vehicle_limit),
        "vehicles_rendered": len(selected),
        "vehicles_omitted": omitted,
    }


def render_scene_state(
    observation: dict[str, Any], vehicle_limit: int = SCENE_VEHICLE_LIMIT
) -> str:
    """Compact text for the fixed-size model context.

    The prompt is capped at ``vehicle_limit`` records. The full observation
    object retains every field and vehicle for audit and training metadata; the
    final ``X<n>`` field makes the omitted relevant vehicles explicit.
    """
    selected, omitted = select_scene_records(observation, vehicle_limit)
    on_screen = [record for record in selected if record["on_screen"]]
    off_screen = [record for record in selected if not record["on_screen"]]
    player = observation["player"]
    include_absolute = len(selected) <= 7

    def short(value: float) -> str:
        text = f"{float(value):.2f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"

    # P fields: current lane, target lane, speed m/s, acceleration m/s^2,
    # braking, lane change in progress, progress %, remaining estimate s,
    # cooldown s, previous action. The full names live in the observation.
    lines = [
        "v2",
        "P %s %s %d %d %d %d %d %s %s %s"
        % (
            player["lane"][0], player["target_lane"][0], round(player["speed_mps"]),
            round(player["acceleration_mps2"]), int(player["braking"]),
            int(player["lane_change_in_progress"]),
            round(player["lane_change_progress"] * 100.0),
            short(player["lane_change_remaining_estimate_s"]),
            short(player["lane_change_cooldown_s"]),
            player["previous_action"][0],
        ),
        "V laneDZ close abs type; +ahead -behind +closing; c car t truck"
        if include_absolute else
        "V laneDZ close type; +ahead -behind +closing; c car t truck",
        "ON",
    ]

    def compact_signed(value: float) -> str:
        # Rounding -0.4 with :+.0f prints "-0", which reads as a sign error.
        # Keep one decimal near zero so the ahead/behind sign stays truthful.
        if abs(value) < 1.0:
            return f"{float(value):+.1f}"
        return f"{int(round(float(value))):+d}"

    def vehicle_line(record):
        absolute = f" {round(record['absolute_speed_mps'])}" if include_absolute else ""
        return (
            f"{record['lane'][0]}{compact_signed(record['relative_distance_z'])} "
            f"{compact_signed(record['closing_speed_mps'])}{absolute} {record['type'][0]}"
        )

    if on_screen:
        lines.extend(vehicle_line(record) for record in on_screen)
    else:
        lines.append("none")
    lines.append("OFF")
    if off_screen:
        lines.extend(vehicle_line(record) for record in off_screen)
    else:
        lines.append("none")
    lane_fields = []
    for lane in observation["lane_summaries"]:
        lane_fields.append(f"{lane['lane'][0]}{lane['ahead_count']}/{lane['behind_count']}")
    lines.append(" ".join(lane_fields) + f" X{omitted}")
    return "\n".join(lines)


def render_compact_state(compact: dict[str, Any]) -> str:
    """Short numeric compact prompt that fits the L256 graph without slicing.

    The old YAML rendering of ``compact_kev_state`` was 190-212 tokens and was
    silently cut to the state budget. This rendering keeps the player and the
    nearest three vehicles and stays well under 100 tokens. The legal options
    live in the question branch, so repeating them in the state is redundant.
    """
    player = compact["player"]

    def signed(value: Any, digits: int = 0) -> str:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "-"
        # Preserve the sign of a small nonzero gap; "-0" is a sign error.
        if abs(float(value)) < 1.0:
            return f"{float(value):+.1f}"
        return f"{float(value):+.{digits}f}"

    lines = [
        "c1",
        "P %s %s %d %d"
        % (
            player.get("lane"),
            player.get("target_lane"),
            round(float(player.get("speed") or 0.0)),
            int(bool(player.get("lane_change_in_progress"))),
        ),
    ]
    traffic = compact.get("traffic") or []
    if not traffic:
        lines.append("V none")
    else:
        for vehicle in traffic:
            ttc = vehicle.get("time_to_collision")
            ttc_text = f"{float(ttc):.1f}" if isinstance(ttc, (int, float)) and not isinstance(ttc, bool) else "-"
            lines.append(
                "V %s %s %s %d %s"
                % (
                    vehicle.get("lane"),
                    signed(vehicle.get("relative_distance")),
                    signed(vehicle.get("relative_speed")),
                    int(bool(vehicle.get("in_player_lane"))),
                    ttc_text,
                )
            )
    return "\n".join(lines)


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
