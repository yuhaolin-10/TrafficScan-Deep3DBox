from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]

RULE_EMERGENCY_LANE_OCCUPATION = "emergency_lane_occupation"
RULE_NO_PARKING = "no_parking"
RULE_NO_NON_MOTOR = "no_non_motor"
RULE_NO_WRONG_WAY = "no_wrong_way"

RULE_DISPLAY_NAMES = {
    RULE_EMERGENCY_LANE_OCCUPATION: "占用应急车道",
    RULE_NO_PARKING: "禁止停车",
    RULE_NO_NON_MOTOR: "禁止非机动车",
    RULE_NO_WRONG_WAY: "禁止逆行",
}


def _as_point(value) -> Optional[Point]:
    try:
        if len(value) != 2:
            return None
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _normalize_points(points) -> List[List[float]]:
    normalized = []
    for point in points or []:
        parsed = _as_point(point)
        if parsed is None:
            continue
        normalized.append([float(parsed[0]), float(parsed[1])])
    return normalized


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _normalize_vec(start: Point, end: Point) -> Optional[Tuple[float, float]]:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    return dx / length, dy / length


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return False
    return float(cv2.pointPolygonTest(polygon.astype(np.float32), point, False)) >= 0.0


def _polygon_overlap_ratio(subject_points, region_points) -> float:
    subject = np.asarray(subject_points or [], dtype=np.float32)
    region = np.asarray(region_points or [], dtype=np.float32)
    if subject.ndim != 2 or subject.shape[0] < 3 or subject.shape[1] != 2:
        return 0.0
    if region.ndim != 2 or region.shape[0] < 3 or region.shape[1] != 2:
        return 0.0

    min_x = int(math.floor(min(float(np.min(subject[:, 0])), float(np.min(region[:, 0]))))) - 1
    min_y = int(math.floor(min(float(np.min(subject[:, 1])), float(np.min(region[:, 1]))))) - 1
    max_x = int(math.ceil(max(float(np.max(subject[:, 0])), float(np.max(region[:, 0]))))) + 1
    max_y = int(math.ceil(max(float(np.max(subject[:, 1])), float(np.max(region[:, 1]))))) + 1

    width = max(2, max_x - min_x + 1)
    height = max(2, max_y - min_y + 1)
    offset = np.array([[-float(min_x), -float(min_y)]], dtype=np.float32)

    subject_mask = np.zeros((height, width), dtype=np.uint8)
    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(subject_mask, [np.round(subject + offset).astype(np.int32)], 255)
    cv2.fillPoly(region_mask, [np.round(region + offset).astype(np.int32)], 255)

    subject_area = int(np.count_nonzero(subject_mask))
    if subject_area <= 0:
        return 0.0
    overlap_area = int(np.count_nonzero(cv2.bitwise_and(subject_mask, region_mask)))
    return float(overlap_area) / float(subject_area)


def _polygon_centroid(points) -> Optional[Point]:
    polygon = np.asarray(points or [], dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] <= 0 or polygon.shape[1] != 2:
        return None
    return float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))


def _bbox_center(bbox) -> Optional[Point]:
    try:
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    except Exception:
        return None


def _detection_anchor_point(detection: dict) -> Optional[Point]:
    for candidate in (
        detection.get("track_anchor", []),
        detection.get("label_anchor", []),
        dict(detection.get("yaw_debug_vectors", {}) or {}).get("center", []),
    ):
        point = _as_point(candidate)
        if point is not None:
            return point
    centroid = _polygon_centroid(detection.get("footprint", []))
    if centroid is not None:
        return centroid
    return _bbox_center(detection.get("bbox", []))


def _detection_direction_unit(detection: dict) -> Optional[Tuple[float, float]]:
    vectors = dict(detection.get("yaw_debug_vectors", {}) or {})
    for candidate in (
        vectors.get("final_direction", []),
        vectors.get("perspective_direction", []),
        dict(vectors.get("lane_prior", {}) or {}).get("direction", []),
    ):
        point = _as_point(candidate)
        if point is None:
            continue
        direction = _normalize_vec((0.0, 0.0), point)
        if direction is not None:
            return direction
    return None


@dataclass
class RegionRuleSpec:
    region_id: str
    name: str
    points: List[List[float]]
    polygon: np.ndarray
    direction_line: List[List[float]]
    direction_unit: Optional[Tuple[float, float]]
    rule_bindings: List[dict]


class RegionRuleEngine:
    def __init__(self, scene_regions):
        self.regions = self._normalize_scene_regions(scene_regions)
        self._rule_states: Dict[tuple, dict] = {}
        self._track_last_seen: Dict[str, int] = {}

    def _normalize_scene_regions(self, scene_regions) -> List[RegionRuleSpec]:
        normalized = []
        for index, entry in enumerate(scene_regions or [], start=1):
            points = _normalize_points(dict(entry or {}).get("points", []))
            if len(points) < 3:
                continue
            polygon = np.asarray(points, dtype=np.float32)
            direction_line = _normalize_points(dict(entry or {}).get("direction_line", []))
            direction_unit = None
            if len(direction_line) == 2:
                direction_unit = _normalize_vec(tuple(direction_line[0]), tuple(direction_line[1]))
            bindings = []
            for binding in dict(entry or {}).get("rule_bindings", []):
                rule_type = str(dict(binding or {}).get("rule_type", "") or "").strip().lower()
                if not rule_type or not bool(dict(binding or {}).get("enabled", True)):
                    continue
                bindings.append(
                    {
                        "rule_type": rule_type,
                        "enabled": True,
                        "params": dict(dict(binding or {}).get("params", {}) or {}),
                    }
                )
            if not bindings:
                continue
            normalized.append(
                RegionRuleSpec(
                    region_id=str(dict(entry or {}).get("region_id", "") or f"region_{index:02d}"),
                    name=str(dict(entry or {}).get("name", "") or f"Region {index}"),
                    points=points,
                    polygon=polygon,
                    direction_line=direction_line if len(direction_line) == 2 else [],
                    direction_unit=direction_unit,
                    rule_bindings=bindings,
                )
            )
        return normalized

    def _state_key(self, track_id: str, region: RegionRuleSpec, rule_type: str):
        return str(track_id), str(region.region_id), str(rule_type)

    def _rule_state(self, track_id: str, region: RegionRuleSpec, rule_type: str) -> dict:
        return self._rule_states.setdefault(
            self._state_key(track_id, region, rule_type),
            {
                "triggered": False,
                "inside_frames": 0,
                "bad_frames": 0,
                "stationary_started_s": None,
                "entry_anchor": None,
                "last_anchor": None,
                "last_timestamp_s": None,
                "last_seen_frame": None,
            },
        )

    def _cleanup_states(self, frame_index: Optional[int]):
        if frame_index is None:
            return
        stale_track_ids = {
            track_id
            for track_id, last_seen in self._track_last_seen.items()
            if int(frame_index) - int(last_seen) > 60
        }
        if not stale_track_ids:
            return
        self._track_last_seen = {
            track_id: last_seen
            for track_id, last_seen in self._track_last_seen.items()
            if track_id not in stale_track_ids
        }
        self._rule_states = {
            key: value
            for key, value in self._rule_states.items()
            if key[0] not in stale_track_ids
        }

    def _movement_speed(self, previous_anchor, current_anchor, state: dict, timestamp_s) -> float:
        prev_point = _as_point(previous_anchor)
        curr_point = _as_point(current_anchor)
        if prev_point is None or curr_point is None:
            return 0.0
        distance_px = _distance(prev_point, curr_point)
        if timestamp_s is None or state.get("last_timestamp_s") is None:
            return float(distance_px)
        dt = float(timestamp_s) - float(state.get("last_timestamp_s"))
        if dt <= 1e-6:
            return float(distance_px)
        return float(distance_px) / dt

    def _ensure_violation_payload(self, detection: dict):
        payload = list(detection.get("rule_violations", []) or [])
        detection["rule_violations"] = payload
        return payload

    def _append_violation(self, detection: dict, region: RegionRuleSpec, rule_type: str):
        violations = self._ensure_violation_payload(detection)
        if any(
            str(item.get("region_id", "")) == str(region.region_id)
            and str(item.get("rule_type", "")) == str(rule_type)
            for item in violations
        ):
            return
        violations.append(
            {
                "region_id": str(region.region_id),
                "region_name": str(region.name),
                "rule_type": str(rule_type),
                "rule_label": RULE_DISPLAY_NAMES.get(str(rule_type), str(rule_type)),
            }
        )
        violation_labels = []
        existing_violation_type = str(detection.get("violation_type", "") or "").strip()
        if existing_violation_type and existing_violation_type.lower() != "none":
            violation_labels.extend(
                [
                    item.strip()
                    for item in existing_violation_type.split(",")
                    if item.strip()
                ]
            )
        for item in violations:
            label = str(item.get("rule_label", item.get("rule_type", "rule")) or "").strip()
            if label and label not in violation_labels:
                violation_labels.append(label)
        detection["is_violating"] = True
        detection["violation_type"] = ", ".join(violation_labels)
        detection["violation_ratio"] = max(float(detection.get("violation_ratio", 0.0) or 0.0), 1.0)

    def _new_event(self, detection: dict, region: RegionRuleSpec, rule_type: str, *, frame_index=None, timestamp_s=None):
        return {
            "track_id": str(detection.get("track_id", "") or ""),
            "vehicle_type": str(detection.get("vehicle_type", "vehicle") or "vehicle"),
            "region_id": str(region.region_id),
            "region_name": str(region.name),
            "rule_type": str(rule_type),
            "rule_label": RULE_DISPLAY_NAMES.get(str(rule_type), str(rule_type)),
            "frame_index": None if frame_index is None else int(frame_index),
            "timestamp_s": None if timestamp_s is None else float(timestamp_s),
        }

    def _apply_no_non_motor(self, detection: dict, region: RegionRuleSpec, binding: dict, state: dict, *, inside: bool, frame_index=None, timestamp_s=None):
        target_classes = {
            str(item).strip().lower()
            for item in dict(binding.get("params", {}) or {}).get("target_classes", ["bicycle", "motorcycle", "person"])
        }
        min_frames = max(1, int(dict(binding.get("params", {}) or {}).get("min_consecutive_frames", 2) or 2))
        min_hits = max(1, int(dict(binding.get("params", {}) or {}).get("min_confirmed_hits", 1) or 1))
        vehicle_type = str(detection.get("vehicle_type", "") or "").strip().lower()
        if not inside or vehicle_type not in target_classes or int(detection.get("track_hits", 0) or 0) < min_hits:
            state["inside_frames"] = 0
            return None
        state["inside_frames"] = int(state.get("inside_frames", 0) or 0) + 1
        if state["triggered"] or state["inside_frames"] < min_frames:
            return None
        state["triggered"] = True
        return self._new_event(detection, region, RULE_NO_NON_MOTOR, frame_index=frame_index, timestamp_s=timestamp_s)

    def _apply_no_parking(self, detection: dict, region: RegionRuleSpec, binding: dict, state: dict, *, inside: bool, frame_index=None, timestamp_s=None):
        params = dict(binding.get("params", {}) or {})
        target_classes = {
            str(item).strip().lower()
            for item in params.get("target_classes", ["car", "truck", "bus", "motorcycle"])
        }
        min_stop_seconds = max(1.0, float(params.get("min_stop_seconds", 5.0) or 5.0))
        max_speed = max(1.0, float(params.get("max_speed_px_per_s", 24.0) or 24.0))
        min_hits = max(1, int(params.get("min_confirmed_hits", 2) or 2))
        vehicle_type = str(detection.get("vehicle_type", "") or "").strip().lower()
        if not inside or vehicle_type not in target_classes or int(detection.get("track_hits", 0) or 0) < min_hits:
            state["stationary_started_s"] = None
            return None
        current_anchor = detection.get("track_anchor", [])
        speed = self._movement_speed(detection.get("track_previous_anchor", []), current_anchor, state, timestamp_s)
        if speed <= max_speed:
            if state.get("stationary_started_s") is None:
                start_value = timestamp_s if timestamp_s is not None else frame_index
                state["stationary_started_s"] = start_value
        else:
            state["stationary_started_s"] = None
            state["triggered"] = False
            return None

        started = state.get("stationary_started_s")
        if started is None:
            return None
        if timestamp_s is not None and isinstance(started, (int, float)):
            stationary_duration = float(timestamp_s) - float(started)
        elif frame_index is not None and isinstance(started, (int, float)):
            stationary_duration = float(frame_index) - float(started)
        else:
            stationary_duration = 0.0
        if state["triggered"] or stationary_duration < min_stop_seconds:
            return None
        state["triggered"] = True
        return self._new_event(detection, region, RULE_NO_PARKING, frame_index=frame_index, timestamp_s=timestamp_s)

    def _apply_no_wrong_way(self, detection: dict, region: RegionRuleSpec, binding: dict, state: dict, *, inside: bool, overlap_ratio=0.0, frame_index=None, timestamp_s=None):
        if region.direction_unit is None:
            state["bad_frames"] = 0
            state["entry_anchor"] = None
            return None
        params = dict(binding.get("params", {}) or {})
        target_classes = {
            str(item).strip().lower()
            for item in params.get("target_classes", ["car", "truck", "bus", "motorcycle", "bicycle"])
        }
        min_frames = max(1, int(params.get("min_consecutive_frames", 2) or 2))
        min_distance = max(8.0, float(params.get("min_direction_distance_px", 24.0) or 24.0))
        dot_threshold = float(params.get("wrong_way_dot_threshold", -0.10) or -0.10)
        min_overlap_ratio = min(1.0, max(0.0, float(params.get("min_roi_overlap_ratio", 0.20) or 0.20)))
        min_hits = max(1, int(params.get("min_confirmed_hits", 2) or 2))
        vehicle_type = str(detection.get("vehicle_type", "") or "").strip().lower()
        current_anchor = _as_point(detection.get("track_anchor", []))
        if (
            float(overlap_ratio) < min_overlap_ratio
            or vehicle_type not in target_classes
            or current_anchor is None
            or int(detection.get("track_hits", 0) or 0) < min_hits
        ):
            state["bad_frames"] = 0
            state["entry_anchor"] = None
            return None
        if state.get("entry_anchor") is None:
            state["entry_anchor"] = current_anchor
            state["bad_frames"] = 0
            return None

        entry_anchor = _as_point(state.get("entry_anchor"))
        if entry_anchor is None:
            state["entry_anchor"] = current_anchor
            return None
        direction_vec = _normalize_vec(entry_anchor, current_anchor)
        if direction_vec is None or _distance(entry_anchor, current_anchor) < min_distance:
            return None
        dot_value = _dot(direction_vec, region.direction_unit)
        if dot_value <= dot_threshold:
            state["bad_frames"] = int(state.get("bad_frames", 0) or 0) + 1
        else:
            state["bad_frames"] = 0
            state["entry_anchor"] = current_anchor
            state["triggered"] = False
            return None
        if state["triggered"] or state["bad_frames"] < min_frames:
            return None
        state["triggered"] = True
        return self._new_event(detection, region, RULE_NO_WRONG_WAY, frame_index=frame_index, timestamp_s=timestamp_s)

    def apply_image(self, detections, *, frame_index=None, timestamp_s=None):
        if not self.regions:
            return list(detections or []), [], {"no_parking": 0, "no_non_motor": 0, "no_wrong_way": 0}

        enriched = [dict(detection or {}) for detection in list(detections or [])]
        new_events = []
        summary_counts = {
            RULE_NO_PARKING: 0,
            RULE_NO_NON_MOTOR: 0,
            RULE_NO_WRONG_WAY: 0,
        }

        for detection in enriched:
            anchor = _detection_anchor_point(detection)
            if anchor is None:
                continue
            detection["track_anchor"] = [float(anchor[0]), float(anchor[1])]
            overlap_cache = {}
            direction_unit = _detection_direction_unit(detection)
            vehicle_type = str(detection.get("vehicle_type", "") or "").strip().lower()

            for region in self.regions:
                inside = _point_in_polygon(anchor, region.polygon)
                for binding in region.rule_bindings:
                    rule_type = str(binding.get("rule_type", "") or "").strip().lower()
                    if rule_type not in {RULE_NO_NON_MOTOR, RULE_NO_WRONG_WAY}:
                        continue

                    overlap_ratio = overlap_cache.get(region.region_id)
                    if overlap_ratio is None:
                        overlap_ratio = _polygon_overlap_ratio(detection.get("footprint", []), region.points)
                        overlap_cache[region.region_id] = float(overlap_ratio)

                    params = dict(binding.get("params", {}) or {})
                    if rule_type == RULE_NO_NON_MOTOR:
                        target_classes = {
                            str(item).strip().lower()
                            for item in params.get("target_classes", ["bicycle", "motorcycle", "person"])
                            if str(item).strip()
                        }
                        min_overlap_ratio = min(1.0, max(0.0, float(params.get("min_roi_overlap_ratio", 0.20) or 0.20)))
                        if vehicle_type not in target_classes:
                            continue
                        if not inside and float(overlap_ratio) < min_overlap_ratio:
                            continue
                        self._append_violation(detection, region, RULE_NO_NON_MOTOR)
                        new_events.append(self._new_event(detection, region, RULE_NO_NON_MOTOR, frame_index=frame_index, timestamp_s=timestamp_s))
                        summary_counts[RULE_NO_NON_MOTOR] += 1
                        continue

                    if region.direction_unit is None or direction_unit is None:
                        continue
                    target_classes = {
                        str(item).strip().lower()
                        for item in params.get("target_classes", ["car", "truck", "bus", "motorcycle", "bicycle"])
                        if str(item).strip()
                    }
                    min_overlap_ratio = min(1.0, max(0.0, float(params.get("min_roi_overlap_ratio", 0.20) or 0.20)))
                    dot_threshold = float(params.get("wrong_way_dot_threshold", -0.10) or -0.10)
                    if vehicle_type not in target_classes or float(overlap_ratio) < min_overlap_ratio:
                        continue
                    if _dot(direction_unit, region.direction_unit) > dot_threshold:
                        continue
                    self._append_violation(detection, region, RULE_NO_WRONG_WAY)
                    new_events.append(self._new_event(detection, region, RULE_NO_WRONG_WAY, frame_index=frame_index, timestamp_s=timestamp_s))
                    summary_counts[RULE_NO_WRONG_WAY] += 1

        return enriched, new_events, summary_counts

    def apply(self, detections, *, frame_index=None, timestamp_s=None):
        if not self.regions:
            return list(detections or []), [], {"no_parking": 0, "no_non_motor": 0, "no_wrong_way": 0}

        enriched = [dict(detection or {}) for detection in list(detections or [])]
        new_events = []
        summary_counts = {
            RULE_NO_PARKING: 0,
            RULE_NO_NON_MOTOR: 0,
            RULE_NO_WRONG_WAY: 0,
        }
        for detection in enriched:
            track_id = str(detection.get("track_id", "") or "").strip()
            anchor = _as_point(detection.get("track_anchor", []))
            if track_id and frame_index is not None:
                self._track_last_seen[track_id] = int(frame_index)
            if not track_id or anchor is None:
                continue
            for region in self.regions:
                inside = _point_in_polygon(anchor, region.polygon)
                for binding in region.rule_bindings:
                    rule_type = str(binding.get("rule_type", "") or "").strip().lower()
                    if not rule_type:
                        continue
                    state = self._rule_state(track_id, region, rule_type)
                    event = None
                    violation_gate = bool(inside)
                    if rule_type == RULE_NO_NON_MOTOR:
                        event = self._apply_no_non_motor(
                            detection,
                            region,
                            binding,
                            state,
                            inside=inside,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                        )
                    elif rule_type == RULE_NO_PARKING:
                        event = self._apply_no_parking(
                            detection,
                            region,
                            binding,
                            state,
                            inside=inside,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                        )
                    elif rule_type == RULE_NO_WRONG_WAY:
                        params = dict(binding.get("params", {}) or {})
                        overlap_ratio = _polygon_overlap_ratio(
                            detection.get("footprint", []),
                            region.points,
                        )
                        min_overlap_ratio = min(1.0, max(0.0, float(params.get("min_roi_overlap_ratio", 0.20) or 0.20)))
                        violation_gate = float(overlap_ratio) >= min_overlap_ratio
                        event = self._apply_no_wrong_way(
                            detection,
                            region,
                            binding,
                            state,
                            inside=inside,
                            overlap_ratio=overlap_ratio,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                        )

                    if state.get("triggered") and violation_gate:
                        self._append_violation(detection, region, rule_type)
                        summary_counts[rule_type] = summary_counts.get(rule_type, 0) + 1
                    if event is not None:
                        new_events.append(event)
                    state["last_anchor"] = anchor
                    state["last_timestamp_s"] = None if timestamp_s is None else float(timestamp_s)
                    state["last_seen_frame"] = None if frame_index is None else int(frame_index)

        self._cleanup_states(frame_index)
        return enriched, new_events, summary_counts
