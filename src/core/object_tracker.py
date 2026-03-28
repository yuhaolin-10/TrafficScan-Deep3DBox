from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _safe_bbox(value) -> BBox:
    try:
        x1, y1, x2, y2 = [float(item) for item in list(value or [])[:4]]
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_diagonal(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def _point_distance(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _normalize_point_list(points) -> List[Point]:
    normalized: List[Point] = []
    for point in points or []:
        try:
            if len(point) != 2:
                continue
            normalized.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    return normalized


MOTOR_VEHICLE_TYPES = {"car", "truck", "bus", "vehicle"}
LIGHT_ROAD_USER_TYPES = {"person", "bicycle", "motorcycle"}


def _vehicle_family(vehicle_type: str) -> str:
    key = str(vehicle_type or "").strip().lower()
    if not key:
        return ""
    if key in MOTOR_VEHICLE_TYPES:
        return "motor_vehicle"
    if key in LIGHT_ROAD_USER_TYPES:
        return "light_road_user"
    return f"class:{key}"


def _vehicle_types_compatible(track_vehicle_type: str, detected_vehicle_type: str) -> bool:
    track_family = _vehicle_family(track_vehicle_type)
    detected_family = _vehicle_family(detected_vehicle_type)
    if not track_family or not detected_family:
        return True
    return track_family == detected_family


def _vehicle_type_match_penalty(track_vehicle_type: str, detected_vehicle_type: str) -> float:
    track_key = str(track_vehicle_type or "").strip().lower()
    detected_key = str(detected_vehicle_type or "").strip().lower()
    if not track_key or not detected_key or track_key == detected_key:
        return 0.0
    if not _vehicle_types_compatible(track_key, detected_key):
        return float("inf")
    return 12.0


def _update_vehicle_type_votes(track, detected_vehicle_type: str) -> None:
    key = str(detected_vehicle_type or "").strip().lower()
    if not key:
        return
    track.vehicle_type_votes[key] = int(track.vehicle_type_votes.get(key, 0)) + 1
    current_key = str(track.vehicle_type or "").strip().lower()
    selected_key, _support = max(
        track.vehicle_type_votes.items(),
        key=lambda item: (
            int(item[1]),
            1 if item[0] == current_key else 0,
            1 if item[0] == key else 0,
            item[0],
        ),
    )
    track.vehicle_type = str(selected_key)


def detection_anchor_point(detection: dict) -> Point:
    footprint = _normalize_point_list(detection.get("footprint", []))
    if footprint:
        xs = [point[0] for point in footprint]
        ys = [point[1] for point in footprint]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return _bbox_center(_safe_bbox(detection.get("bbox", [])))


@dataclass
class TrackState:
    track_id: str
    vehicle_type: str
    anchor: Point
    bbox: BBox
    age: int = 1
    hits: int = 1
    miss_count: int = 0
    start_frame_index: Optional[int] = None
    last_frame_index: Optional[int] = None
    start_timestamp_s: Optional[float] = None
    last_timestamp_s: Optional[float] = None
    history: List[Point] = field(default_factory=list)
    vehicle_type_votes: Dict[str, int] = field(default_factory=dict)

    def displacement_px(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return _point_distance(self.history[0], self.history[-1])


class ObjectTracker:
    def __init__(
        self,
        *,
        max_match_distance: float = 110.0,
        max_missing_frames: int = 8,
        match_distance_ratio: float = 0.75,
        min_confirm_hits: int = 2,
    ):
        self.max_match_distance = float(max_match_distance)
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.match_distance_ratio = float(match_distance_ratio)
        self.min_confirm_hits = max(1, int(min_confirm_hits))
        self._next_track_index = 1
        self._active_tracks: Dict[str, TrackState] = {}
        self._all_tracks: Dict[str, TrackState] = {}

    def active_tracks(self) -> List[TrackState]:
        return list(self._active_tracks.values())

    def all_tracks(self) -> List[TrackState]:
        return list(self._all_tracks.values())

    def track_count(self, *, confirmed_only: bool = False) -> int:
        if not confirmed_only:
            return len(self._all_tracks)
        return sum(1 for track in self._all_tracks.values() if track.hits >= self.min_confirm_hits)

    def _new_track_id(self) -> str:
        track_id = f"T{self._next_track_index:04d}"
        self._next_track_index += 1
        return track_id

    def _match_limit(self, track: TrackState, det_bbox: BBox) -> float:
        scale = max(_bbox_diagonal(track.bbox), _bbox_diagonal(det_bbox), 40.0)
        return max(self.max_match_distance, scale * self.match_distance_ratio)

    def _create_track(self, det_meta: dict, *, frame_index=None, timestamp_s=None) -> TrackState:
        vehicle_type = str(det_meta["vehicle_type"] or "").strip().lower()
        track = TrackState(
            track_id=self._new_track_id(),
            vehicle_type=vehicle_type,
            anchor=det_meta["anchor"],
            bbox=det_meta["bbox"],
            start_frame_index=None if frame_index is None else int(frame_index),
            last_frame_index=None if frame_index is None else int(frame_index),
            start_timestamp_s=None if timestamp_s is None else float(timestamp_s),
            last_timestamp_s=None if timestamp_s is None else float(timestamp_s),
            history=[det_meta["anchor"]],
            vehicle_type_votes={vehicle_type: 1} if vehicle_type else {},
        )
        self._active_tracks[track.track_id] = track
        self._all_tracks[track.track_id] = track
        return track

    def _update_track(self, track: TrackState, det_meta: dict, *, frame_index=None, timestamp_s=None) -> Point:
        previous_anchor = track.anchor
        track.anchor = det_meta["anchor"]
        track.bbox = det_meta["bbox"]
        _update_vehicle_type_votes(track, det_meta["vehicle_type"])
        track.age += 1
        track.hits += 1
        track.miss_count = 0
        if frame_index is not None:
            track.last_frame_index = int(frame_index)
        if timestamp_s is not None:
            track.last_timestamp_s = float(timestamp_s)
        track.history.append(det_meta["anchor"])
        return previous_anchor

    def update(self, detections, *, frame_index=None, timestamp_s=None):
        enriched: List[dict] = []
        det_meta = []
        for detection in detections or []:
            bbox = _safe_bbox(detection.get("bbox", []))
            det_meta.append(
                {
                    "bbox": bbox,
                    "anchor": detection_anchor_point(detection),
                    "vehicle_type": str(detection.get("vehicle_type", "") or "").strip().lower(),
                }
            )

        assignments: Dict[int, str] = {}
        if self._active_tracks and det_meta:
            candidates = []
            for track_id, track in self._active_tracks.items():
                for det_index, meta in enumerate(det_meta):
                    distance = _point_distance(track.anchor, meta["anchor"])
                    if not _vehicle_types_compatible(track.vehicle_type, meta["vehicle_type"]):
                        continue
                    class_penalty = _vehicle_type_match_penalty(track.vehicle_type, meta["vehicle_type"])
                    limit = self._match_limit(track, meta["bbox"]) + class_penalty
                    if distance > limit:
                        continue
                    iou = _bbox_iou(track.bbox, meta["bbox"])
                    score = distance + class_penalty - (iou * 24.0)
                    candidates.append((score, -iou, distance, track_id, det_index))

            matched_tracks = set()
            matched_detections = set()
            for _score, _neg_iou, _distance, track_id, det_index in sorted(candidates):
                if track_id in matched_tracks or det_index in matched_detections:
                    continue
                assignments[det_index] = track_id
                matched_tracks.add(track_id)
                matched_detections.add(det_index)

        matched_track_ids = set(assignments.values())
        stale_track_ids = []
        for track_id, track in list(self._active_tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.miss_count += 1
            if track.miss_count > self.max_missing_frames:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self._active_tracks.pop(track_id, None)

        for det_index, detection in enumerate(detections or []):
            meta = det_meta[det_index]
            output = dict(detection)
            previous_anchor = None
            assigned_track_id = assignments.get(det_index)
            if assigned_track_id:
                track = self._active_tracks[assigned_track_id]
                previous_anchor = self._update_track(
                    track,
                    meta,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                )
            else:
                track = self._create_track(meta, frame_index=frame_index, timestamp_s=timestamp_s)

            raw_vehicle_type = str(output.get("vehicle_type", "") or "").strip().lower()
            if raw_vehicle_type:
                output["vehicle_type_raw"] = raw_vehicle_type
            stable_vehicle_type = str(track.vehicle_type or raw_vehicle_type)
            if stable_vehicle_type:
                output["vehicle_type"] = stable_vehicle_type
                output["track_vehicle_type"] = stable_vehicle_type
                output["track_vehicle_type_support"] = int(track.vehicle_type_votes.get(stable_vehicle_type, 0))
            output["track_id"] = track.track_id
            output["track_age"] = int(track.age)
            output["track_hits"] = int(track.hits)
            output["track_confirmed"] = bool(track.hits >= self.min_confirm_hits)
            output["track_anchor"] = [float(track.anchor[0]), float(track.anchor[1])]
            output["track_displacement_px"] = float(track.displacement_px())
            if previous_anchor is not None:
                output["track_previous_anchor"] = [float(previous_anchor[0]), float(previous_anchor[1])]
            enriched.append(output)

        return enriched
