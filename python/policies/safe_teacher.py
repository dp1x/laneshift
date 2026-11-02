"""Deterministic safety teacher for LANESHIFT lane decisions.

This is a label generator and baseline, not the decision model. It checks the
nearest vehicle ahead and behind in the current and adjacent lanes, requires a
front time-to-collision and a rear clearance before changing lanes, and brakes
when the current lane is unsafe and no adjacent lane is safe.

The teacher is intentionally conservative. It uses only the state the game
already gives the controller: lane, speed, relative gaps, relative speeds and
the authoritative legal-action set.
"""

from __future__ import annotations

import math
from typing import Any

from policy import BasePolicy


class SafeTeacherPolicy(BasePolicy):
    label_source = "safe_teacher"
    policy_name = "safe_teacher"

    # Front safety: keep a 20 m gap or a 3 s TTC, with a 10 m hard floor.
    FRONT_GAP_M = 20.0
    FRONT_TTC_S = 3.0
    FRONT_FLOOR_M = 10.0
    # Rear safety: a vehicle behind is acceptable when it is 20 m back or the
    # closing TTC is at least 2 s, with an 8 m hard floor.
    REAR_GAP_M = 20.0
    REAR_TTC_S = 2.0
    REAR_FLOOR_M = 8.0

    def choose_action(self, state: dict[str, Any]) -> str:
        player = state.get("player") or {}
        legal = list(state.get("legal_actions") or [])
        traffic = [v for v in (state.get("traffic") or []) if isinstance(v, dict)]
        lane = int(player.get("lane", 1))
        speed = float(player.get("speed", 0.0))

        current = self._lane_eval(traffic, lane, speed)
        if current["safe"]:
            return "STAY" if "STAY" in legal else self.legal_fallback(state)

        candidates = []
        for action, target in (("LEFT", lane - 1), ("RIGHT", lane + 1)):
            if action not in legal or target < 0 or target > 2:
                continue
            evaluation = self._lane_eval(traffic, target, speed)
            if evaluation["safe"]:
                candidates.append((evaluation["score"], evaluation["front_gap"], action))

        if candidates:
            # Highest score first; a tie prefers the larger front gap, then LEFT.
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            return candidates[0][2]
        if "BRAKE" in legal:
            return "BRAKE"
        return "STAY" if "STAY" in legal else self.legal_fallback(state)

    def _lane_eval(self, traffic: list[dict[str, Any]], lane: int, speed: float) -> dict[str, Any]:
        ahead: list[tuple[float, float]] = []
        behind: list[tuple[float, float]] = []
        for vehicle in traffic:
            if vehicle.get("lane") != lane:
                continue
            gap = vehicle.get("relative_distance_z")
            rel = vehicle.get("relative_speed_z", 0.0)
            if not isinstance(gap, (int, float)) or isinstance(gap, bool):
                continue
            if not isinstance(rel, (int, float)) or isinstance(rel, bool):
                rel = 0.0
            if gap >= 0.0:
                ahead.append((float(gap), float(rel)))
            else:
                behind.append((float(gap), float(rel)))

        front_gap, front_rel = min(ahead, key=lambda item: item[0]) if ahead else (math.inf, 0.0)
        rear_gap, rear_rel = max(behind, key=lambda item: item[0]) if behind else (-math.inf, 0.0)

        front_ttc = front_gap / front_rel if front_rel > 0.01 else math.inf
        rear_distance = abs(rear_gap) if math.isfinite(rear_gap) else math.inf
        rear_closing = -rear_rel if rear_rel < -0.01 else 0.0
        rear_ttc = rear_distance / rear_closing if rear_closing > 0.01 else math.inf

        front_safe = (
            front_gap >= self.FRONT_FLOOR_M
            and (front_gap >= self.FRONT_GAP_M or front_ttc >= self.FRONT_TTC_S)
        )
        rear_safe = (
            rear_distance >= self.REAR_FLOOR_M
            and (rear_distance >= self.REAR_GAP_M or rear_ttc >= self.REAR_TTC_S)
        )
        safe = front_safe and rear_safe
        score = min(
            front_ttc,
            rear_ttc,
            front_gap / 10.0 if math.isfinite(front_gap) else 100.0,
            rear_distance / 10.0 if math.isfinite(rear_distance) else 100.0,
        )
        if not safe:
            score -= 100.0
        return {
            "safe": safe,
            "score": score,
            "front_gap": front_gap,
            "front_ttc": front_ttc,
            "rear_gap": rear_distance,
            "rear_ttc": rear_ttc,
        }
