"""Simple rule-based traffic-avoidance baseline."""
from policy import BasePolicy


class RuleBasedPolicy(BasePolicy):
    label_source = "rule_based"

    def choose_action(self, state):
        player = state.get("player", {})
        lane = player.get("lane", 1)
        speed = player.get("speed", 20.0)
        traffic = state.get("traffic", [])
        # find nearest vehicle ahead in same lane
        nearest = None
        for v in traffic:
            if v.get("lane") != lane:
                continue
            gap = v.get("relative_distance_z", 999)
            if gap >= 0 and (nearest is None or gap < nearest.get("relative_distance_z", 999)):
                nearest = v
        if nearest is not None:
            gap = nearest.get("relative_distance_z", 999)
            rel = nearest.get("relative_speed_z", 0)
            if gap < 18.0:
                # try change lane to side with more room
                left_gap = self._lane_gap(traffic, lane - 1, player)
                right_gap = self._lane_gap(traffic, lane + 1, player)
                if lane > 0 and left_gap >= right_gap and left_gap > gap:
                    return "LEFT"
                if lane < 2 and right_gap > gap:
                    return "RIGHT"
                return "BRAKE"
        return "STAY"

    def _lane_gap(self, traffic, lane, player):
        if lane < 0 or lane > 2:
            return -1.0
        best = 1000.0
        for v in traffic:
            if v.get("lane") != lane:
                continue
            gap = v.get("relative_distance_z", 999)
            if gap >= 0 and gap < best:
                best = gap
        return best
