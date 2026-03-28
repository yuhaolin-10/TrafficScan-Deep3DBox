from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class CountEvent:
    track_id: str
    rule_name: str
    direction: str
    frame_index: Optional[int] = None
    timestamp_s: Optional[float] = None


@dataclass
class CountLineRule:
    name: str
    start: Point
    end: Point
    direction_mode: str = "any"


class TrafficCounter:
    def __init__(self, rule: CountLineRule):
        self.rule = rule
        self._counted_tracks: Dict[str, CountEvent] = {}

    @staticmethod
    def _orientation(a: Point, b: Point, c: Point) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))

    def _segment_cross_direction(self, previous_point: Point, current_point: Point) -> Optional[str]:
        line_a = self.rule.start
        line_b = self.rule.end
        prev_side = self._orientation(line_a, line_b, previous_point)
        curr_side = self._orientation(line_a, line_b, current_point)
        if prev_side == 0.0 and curr_side == 0.0:
            return None
        if prev_side * curr_side > 0.0:
            return None
        if prev_side < 0.0 <= curr_side:
            return "forward"
        if prev_side > 0.0 >= curr_side:
            return "backward"
        return "touch"

    def update(
        self,
        track_id: str,
        previous_point: Sequence[float],
        current_point: Sequence[float],
        *,
        frame_index: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> Optional[CountEvent]:
        if track_id in self._counted_tracks:
            return None
        prev = (float(previous_point[0]), float(previous_point[1]))
        curr = (float(current_point[0]), float(current_point[1]))
        direction = self._segment_cross_direction(prev, curr)
        if direction is None:
            return None
        if self.rule.direction_mode != "any" and direction != self.rule.direction_mode:
            return None
        event = CountEvent(
            track_id=str(track_id),
            rule_name=self.rule.name,
            direction=direction,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        self._counted_tracks[str(track_id)] = event
        return event

    def counted_total(self) -> int:
        return len(self._counted_tracks)

    def has_counted(self, track_id: str) -> bool:
        return str(track_id) in self._counted_tracks
