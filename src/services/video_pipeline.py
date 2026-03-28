from datetime import datetime
import math
from pathlib import Path
import time

import cv2
import numpy as np

try:
    from .pipeline import process_frame
    from .region_rule_engine import RegionRuleEngine, RULE_NO_NON_MOTOR, RULE_NO_PARKING, RULE_NO_WRONG_WAY
    from .renderer import render_result
    from .video_reader import iter_video_frames, read_video_info
    from ..core.plate_recognizer import (
        PLATE_VEHICLE_TYPES,
        PlateCandidate,
        recognize_vehicle_plate,
        summarize_plate_candidates,
    )
except Exception:
    from services.pipeline import process_frame
    from services.region_rule_engine import RegionRuleEngine, RULE_NO_NON_MOTOR, RULE_NO_PARKING, RULE_NO_WRONG_WAY
    from services.renderer import render_result
    from services.video_reader import iter_video_frames, read_video_info
    from core.plate_recognizer import (
        PLATE_VEHICLE_TYPES,
        PlateCandidate,
        recognize_vehicle_plate,
        summarize_plate_candidates,
    )

from core.object_tracker import ObjectTracker
from core.traffic_counter import CountLineRule, TrafficCounter


DEFAULT_VIOLATION_PLATE_WINDOW_S = 1.0
DEFAULT_TRACK_PLATE_MIN_HITS = 3
DEFAULT_TRACK_PLATE_INTERVAL_FRAMES = 6
DEFAULT_TRACK_PLATE_VIOLATION_INTERVAL_FRAMES = 3
DEFAULT_TRACK_PLATE_STABLE_INTERVAL_FRAMES = 18
DEFAULT_TRACK_PLATE_MIN_REGION_OVERLAP_RATIO = 0.05


def _get_value(item, name, default=None):
    if item is None:
        return default
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_count_rules(count_lines):
    rules = []
    for index, item in enumerate(count_lines or [], start=1):
        enabled = bool(_get_value(item, "enabled", True))
        if not enabled:
            continue
        start = _get_value(item, "start", [])
        end = _get_value(item, "end", [])
        try:
            if len(start) != 2 or len(end) != 2:
                continue
            start_pt = (float(start[0]), float(start[1]))
            end_pt = (float(end[0]), float(end[1]))
        except Exception:
            continue
        direction_mode = str(_get_value(item, "direction_mode", "any") or "any").strip().lower()
        if direction_mode not in {"any", "forward", "backward"}:
            direction_mode = "any"
        name = str(_get_value(item, "name", f"count_line_{index:02d}") or f"count_line_{index:02d}")
        rules.append(
            CountLineRule(
                name=name,
                start=start_pt,
                end=end_pt,
                direction_mode=direction_mode,
            )
        )
    return rules


def _auto_count_rule(lane_polygons, image_width, image_height):
    polygons = []
    for polygon in lane_polygons or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            continue
        polygons.append(pts)

    if polygons:
        merged = np.concatenate(polygons, axis=0)
        min_x = float(np.min(merged[:, 0]))
        max_x = float(np.max(merged[:, 0]))
        min_y = float(np.min(merged[:, 1]))
        max_y = float(np.max(merged[:, 1]))
    else:
        width = max(1.0, float(image_width or 0.0))
        height = max(1.0, float(image_height or 0.0))
        margin_x = max(12.0, width * 0.10)
        margin_y = max(12.0, height * 0.10)
        min_x = margin_x
        max_x = max(min_x + 2.0, width - margin_x)
        min_y = margin_y
        max_y = max(min_y + 2.0, height - margin_y)

    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 2.0 and span_y <= 2.0:
        return None

    if span_y >= span_x:
        center_y = (min_y + max_y) * 0.5
        start = (min_x, center_y)
        end = (max_x, center_y)
    else:
        center_x = (min_x + max_x) * 0.5
        start = (center_x, min_y)
        end = (center_x, max_y)

    return CountLineRule(name="auto_center_line", start=start, end=end, direction_mode="any")


def _int_point(point):
    return tuple(int(round(float(value))) for value in point)


def _build_render_lane_mask(image_shape, lane_polygons):
    height, width = image_shape[:2]
    lane_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for polygon in lane_polygons or []:
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        cv2.fillPoly(lane_mask, [pts], 255)
    return lane_mask


def _draw_text_panel(frame, lines, *, top_left=(12, 12), fg_color=(240, 240, 240), bg_color=(12, 16, 24)):
    if not lines:
        return
    x, y = int(top_left[0]), int(top_left[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54
    thickness = 1
    padding = 8
    line_gap = 6
    text_sizes = [cv2.getTextSize(str(line), font, font_scale, thickness)[0] for line in lines]
    width = max(size[0] for size in text_sizes) + (padding * 2)
    height = sum(size[1] for size in text_sizes) + (line_gap * (len(lines) - 1)) + (padding * 2)
    x = max(4, min(x, max(4, frame.shape[1] - width - 4)))
    y = max(height + 4, min(y + height, max(height + 4, frame.shape[0] - 4)))
    top = (x, y - height)
    bottom = (x + width, y)
    overlay = frame.copy()
    cv2.rectangle(overlay, top, bottom, bg_color, -1)
    frame[:] = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    cv2.rectangle(frame, top, bottom, (70, 82, 98), 1)

    cursor_y = top[1] + padding
    for line, size in zip(lines, text_sizes):
        cursor_y += size[1]
        cv2.putText(
            frame,
            str(line),
            (top[0] + padding, cursor_y),
            font,
            font_scale,
            fg_color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_gap


def _annotate_rendered_frame(
    frame,
    detections,
    count_rules,
    counters,
    *,
    labels_enabled=False,
    confirmed_track_count=0,
    count_total=0,
    forward_total=0,
    backward_total=0,
    current_violation_count=0,
):
    for rule, counter in zip(count_rules or [], counters or []):
        if str(rule.name or "") == "auto_center_line":
            continue
        start = _int_point(rule.start)
        end = _int_point(rule.end)
        cv2.line(frame, start, end, (255, 210, 0), 3, cv2.LINE_AA)
        mid = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        _draw_text_panel(
            frame,
            [f"{rule.name}: {counter.counted_total()}"],
            top_left=(int(mid[0]) + 8, int(mid[1]) - 8),
            fg_color=(255, 248, 220),
            bg_color=(42, 34, 12),
        )

    if labels_enabled:
        for detection in detections or []:
            anchor = detection.get("track_anchor", [])
            if len(anchor) != 2:
                continue
            point = _int_point(anchor)
            cv2.circle(frame, point, 4, (245, 245, 245), -1)
            label = str(detection.get("track_id", "")).strip()
            if not label:
                continue
            _draw_text_panel(
                frame,
                [label],
                top_left=(point[0] + 8, point[1] - 12),
                fg_color=(245, 245, 245),
                bg_color=(18, 18, 18),
            )

    _draw_text_panel(
        frame,
        [
            f"Tracks {int(confirmed_track_count)}",
            f"Count {int(count_total)} | F {int(forward_total)} | B {int(backward_total)}",
            f"Violations {int(current_violation_count)}",
        ],
        top_left=(12, 12),
        fg_color=(240, 240, 240),
        bg_color=(12, 16, 24),
    )


def _safe_progress_callback(callback, payload):
    if callback is None:
        return
    try:
        callback(dict(payload))
    except Exception:
        return


def _estimate_processed_frame_total(frame_count, frame_stride, max_frames):
    if int(frame_count or 0) <= 0:
        return 0
    estimated = int(math.ceil(float(frame_count) / max(1, int(frame_stride))))
    if max_frames is not None:
        estimated = min(estimated, int(max_frames))
    return max(0, estimated)


def _violation_plate_window_frames(fps: float, *, window_s: float = DEFAULT_VIOLATION_PLATE_WINDOW_S) -> int:
    fps_value = float(fps or 0.0)
    if fps_value <= 1e-6:
        fps_value = 6.0
    return max(1, int(round(fps_value * float(window_s))))


def _as_point(value):
    try:
        if len(value) != 2:
            return None
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _point_in_polygon(point, polygon) -> bool:
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return False
    return float(cv2.pointPolygonTest(pts, point, False)) >= 0.0


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


def _detection_is_region_relevant(detection, region_specs) -> bool:
    if bool(detection.get("is_violating", False)) or bool(list(detection.get("rule_violations", []) or [])):
        return True
    if not region_specs:
        return True

    anchor = _as_point(detection.get("track_anchor", []))
    if anchor is None:
        anchor = _as_point(detection.get("label_anchor", []))

    for region in list(region_specs or []):
        if anchor is not None and _point_in_polygon(anchor, getattr(region, "polygon", [])):
            return True
        overlap_ratio = _polygon_overlap_ratio(
            detection.get("footprint", []),
            getattr(region, "points", []),
        )
        if float(overlap_ratio) >= float(DEFAULT_TRACK_PLATE_MIN_REGION_OVERLAP_RATIO):
            return True
    return False


def _new_track_plate_ocr_state() -> dict:
    return {
        "last_attempt_frame": None,
        "attempt_count": 0,
        "last_success_frame": None,
        "success_count": 0,
        "region_relevant": False,
        "last_region_relevant_frame": None,
        "plate_text": "",
        "plate_confidence": 0.0,
        "plate_support_count": 0,
        "plate_stable": False,
    }


def _should_run_track_plate_ocr(
    detection,
    *,
    frame_index: int,
    ocr_state: dict,
    region_specs,
    recognizer,
) -> bool:
    if recognizer is None or not bool(getattr(recognizer, "enabled", False)):
        return False

    vehicle_type = str(detection.get("vehicle_type", "") or "").strip().lower()
    if vehicle_type and vehicle_type not in PLATE_VEHICLE_TYPES:
        return False
    if int(detection.get("track_hits", 0) or 0) < int(DEFAULT_TRACK_PLATE_MIN_HITS):
        return False

    if _detection_is_region_relevant(detection, region_specs):
        ocr_state["region_relevant"] = True
        ocr_state["last_region_relevant_frame"] = int(frame_index)
    elif not bool(ocr_state.get("region_relevant", False)):
        return False

    has_stable_plate = bool(ocr_state.get("plate_stable", False))
    interval_frames = (
        DEFAULT_TRACK_PLATE_STABLE_INTERVAL_FRAMES
        if has_stable_plate
        else DEFAULT_TRACK_PLATE_INTERVAL_FRAMES
    )
    if bool(detection.get("is_violating", False)) and not has_stable_plate:
        interval_frames = min(interval_frames, DEFAULT_TRACK_PLATE_VIOLATION_INTERVAL_FRAMES)

    last_attempt_frame = ocr_state.get("last_attempt_frame")
    if last_attempt_frame is not None and (int(frame_index) - int(last_attempt_frame)) < int(interval_frames):
        return False
    return True


def _append_track_plate_candidate(track_plate_candidates, track_id: str, detection: dict, *, frame_index: int) -> bool:
    plate_text = str(detection.get("plate_text", "") or "").strip()
    if not track_id or not plate_text:
        return False
    track_plate_candidates.setdefault(track_id, []).append(
        PlateCandidate(
            text=plate_text,
            confidence=float(detection.get("plate_confidence", 0.0) or 0.0),
            frame_index=int(frame_index),
            crop_path="",
            plate_type=str(detection.get("plate_type", "") or ""),
            plate_type_id=int(detection.get("plate_type_id", -1) or -1),
            detect_confidence=float(detection.get("plate_detect_confidence", 0.0) or 0.0),
            crop_strategy=str(detection.get("plate_crop_strategy", "") or ""),
            box=[int(v) for v in list(detection.get("plate_box", []))[:4]],
        )
    )
    return True


def _update_track_plate_ocr_state(ocr_state: dict, summary, *, min_confidence: float = 0.0) -> None:
    if summary is None:
        return
    ocr_state["plate_text"] = str(summary.get("plate_text", "") or "")
    ocr_state["plate_confidence"] = float(summary.get("plate_confidence", 0.0) or 0.0)
    ocr_state["plate_support_count"] = int(summary.get("plate_support_count", 0) or 0)
    ocr_state["plate_stable"] = bool(
        ocr_state["plate_text"]
        and int(ocr_state["plate_support_count"]) >= 2
        and float(ocr_state["plate_confidence"]) >= float(min_confidence)
    )


def _filter_plate_candidates_by_frame_window(
    candidates,
    *,
    frame_start=None,
    frame_end=None,
    window_frames=0,
):
    left = None if frame_start is None else int(frame_start) - max(0, int(window_frames))
    right = None if frame_end is None else int(frame_end) + max(0, int(window_frames))
    filtered = []
    for item in list(candidates or []):
        if item.frame_index is None:
            continue
        frame_index = int(item.frame_index)
        if left is not None and frame_index < left:
            continue
        if right is not None and frame_index > right:
            continue
        filtered.append(item)
    return filtered


def _summarize_violation_plate_candidates(
    candidates,
    *,
    frame_start=None,
    frame_end=None,
    window_frames=0,
    min_confidence: float = 0.0,
):
    filtered = _filter_plate_candidates_by_frame_window(
        candidates,
        frame_start=frame_start,
        frame_end=frame_end,
        window_frames=window_frames,
    )
    if not filtered:
        return None
    return summarize_plate_candidates(filtered, min_confidence=min_confidence)


def process_video(
    video_path,
    lane_detector,
    vehicle_detector,
    violation_checker,
    plate_recognizer,
    output_dir,
    *,
    preview_dir=None,
    layers=None,
    lane_override_polygons=None,
    scene_regions=None,
    count_lines=None,
    frame_stride=1,
    max_frames=None,
    stop_requested=None,
    progress_callback=None,
):
    video_path = Path(video_path)
    info = read_video_info(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is None:
        preview_dir = output_dir
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_video_path = output_dir / f"{timestamp_token}_{video_path.stem}_processed.mp4"
    output_preview_path = preview_dir / f"{timestamp_token}_{video_path.stem}_preview.jpg"

    input_fps = float(info.fps) if float(info.fps) > 1e-6 else 10.0
    violation_plate_window_frames = _violation_plate_window_frames(info.fps)
    frame_stride = max(1, int(frame_stride))
    output_fps = max(1.0, input_fps / frame_stride)
    expected_processed_frames = _estimate_processed_frame_total(info.frame_count, frame_stride, max_frames)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (int(info.width), int(info.height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to create output video writer: {output_video_path}")

    tracker = ObjectTracker(max_missing_frames=15, match_distance_ratio=0.9)
    region_rule_engine = RegionRuleEngine(scene_regions)
    count_rules = _normalize_count_rules(count_lines)
    count_line_source = "scene_profile" if count_rules else ""
    counters = [TrafficCounter(rule) for rule in count_rules]
    count_events = []
    region_rule_events = []
    frames_with_count_event = set()
    frames_with_region_rule_violation = set()

    processed_frame_count = 0
    total_vehicle_instances = 0
    total_violation_instances = 0
    frames_with_violation = 0
    max_vehicle_count = 0
    track_plate_candidates: dict[str, list[PlateCandidate]] = {}
    track_plate_ocr_state: dict[str, dict] = {}
    violating_track_meta: dict[str, dict] = {}
    preview_written = False
    lane_sources = set()
    run_started_at = time.monotonic()
    plate_min_confidence = float(getattr(plate_recognizer, "min_confidence", 0.0) or 0.0)
    total_plate_ocr_attempt_count = 0
    total_plate_ocr_success_count = 0

    try:
        for frame_data in iter_video_frames(video_path, stride=frame_stride, max_frames=max_frames):
            if callable(stop_requested) and stop_requested():
                raise InterruptedError(f"Video processing interrupted: {video_path}")
            frame_result, _ = process_frame(
                frame_data.frame,
                lane_detector,
                vehicle_detector,
                violation_checker,
                plate_recognizer=None,
                plate_mode="disabled",
                layers=layers,
                lane_override_polygons=lane_override_polygons,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_index=frame_data.frame_index,
                timestamp_s=frame_data.timestamp_s,
                render_output=False,
            )
            tracked_detections = tracker.update(
                frame_result.get("detections", []),
                frame_index=frame_data.frame_index,
                timestamp_s=frame_data.timestamp_s,
            )
            tracked_detections, frame_rule_events, frame_rule_summary = region_rule_engine.apply(
                tracked_detections,
                frame_index=frame_data.frame_index,
                timestamp_s=frame_data.timestamp_s,
            )
            frame_result["detections"] = tracked_detections
            frame_rule_events = list(frame_rule_events or [])
            if frame_rule_events:
                region_rule_events.extend(frame_rule_events)
                frames_with_region_rule_violation.add(int(frame_data.frame_index))

            if not counters:
                auto_rule = _auto_count_rule(
                    frame_result.get("lane_polygons", []),
                    frame_result.get("image_width", info.width),
                    frame_result.get("image_height", info.height),
                )
                if auto_rule is not None:
                    count_rules = [auto_rule]
                    counters = [TrafficCounter(auto_rule)]
                    count_line_source = "auto_center_line"
                else:
                    count_line_source = "none"

            for detection in tracked_detections:
                track_id = str(detection.get("track_id", "") or "")
                previous_anchor = detection.get("track_previous_anchor", [])
                current_anchor = detection.get("track_anchor", [])
                if len(previous_anchor) != 2 or len(current_anchor) != 2 or not track_id:
                    continue
                for counter in counters:
                    event = counter.update(
                        track_id,
                        previous_anchor,
                        current_anchor,
                        frame_index=frame_data.frame_index,
                        timestamp_s=frame_data.timestamp_s,
                    )
                    if event is None:
                        continue
                    count_events.append(
                        {
                            "track_id": str(event.track_id),
                            "rule_name": str(event.rule_name),
                            "direction": str(event.direction),
                            "frame_index": None if event.frame_index is None else int(event.frame_index),
                            "timestamp_s": None if event.timestamp_s is None else float(event.timestamp_s),
                        }
                    )
                    frames_with_count_event.add(int(frame_data.frame_index))

            frame_rule_violation_count = int(sum(int(value or 0) for value in dict(frame_rule_summary or {}).values()))
            frame_violation_count = int(
                sum(1 for detection in tracked_detections if bool(detection.get("is_violating", False)))
            )
            frame_result["violation_count"] = frame_violation_count
            frame_result["any_violation"] = bool(frame_violation_count > 0)
            frame_result["rule_violation_count"] = frame_rule_violation_count

            frame_plate_ocr_attempt_count = 0
            frame_plate_ocr_success_count = 0
            for detection in tracked_detections:
                track_id = str(detection.get("track_id", "") or "").strip()
                if track_id:
                    ocr_state = track_plate_ocr_state.setdefault(track_id, _new_track_plate_ocr_state())
                    if _should_run_track_plate_ocr(
                        detection,
                        frame_index=int(frame_data.frame_index),
                        ocr_state=ocr_state,
                        region_specs=region_rule_engine.regions,
                        recognizer=plate_recognizer,
                    ):
                        ocr_state["last_attempt_frame"] = int(frame_data.frame_index)
                        ocr_state["attempt_count"] = int(ocr_state.get("attempt_count", 0) or 0) + 1
                        frame_plate_ocr_attempt_count += 1
                        total_plate_ocr_attempt_count += 1
                        try:
                            plate_data = recognize_vehicle_plate(
                                frame_data.frame,
                                detection.get("bbox", []),
                                plate_recognizer,
                                vehicle_type=detection.get("vehicle_type", ""),
                                frame_index=int(frame_data.frame_index),
                            )
                        except Exception:
                            plate_data = None
                        if plate_data is not None:
                            detection.update(plate_data)
                            detection["plate_status"] = "detected"
                            _append_track_plate_candidate(
                                track_plate_candidates,
                                track_id,
                                detection,
                                frame_index=int(frame_data.frame_index),
                            )
                            ocr_state["last_success_frame"] = int(frame_data.frame_index)
                            ocr_state["success_count"] = int(ocr_state.get("success_count", 0) or 0) + 1
                            frame_plate_ocr_success_count += 1
                            total_plate_ocr_success_count += 1
                        else:
                            detection["plate_status"] = "not_found"

                if not bool(detection.get("is_violating", False)) or not track_id:
                    continue
                track_meta = violating_track_meta.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "vehicle_type": str(detection.get("vehicle_type", "vehicle")),
                        "max_violation_ratio": 0.0,
                        "first_violation_frame": None,
                        "last_violation_frame": None,
                    },
                )
                track_meta["vehicle_type"] = str(detection.get("vehicle_type", track_meta["vehicle_type"]))
                track_meta["max_violation_ratio"] = max(
                    float(track_meta.get("max_violation_ratio", 0.0)),
                    float(detection.get("violation_ratio", 0.0) or 0.0),
                )
                if track_meta["first_violation_frame"] is None:
                    track_meta["first_violation_frame"] = int(frame_data.frame_index)
                track_meta["last_violation_frame"] = int(frame_data.frame_index)

            for detection in tracked_detections:
                track_id = str(detection.get("track_id", "") or "").strip()
                if not track_id:
                    continue
                summary = summarize_plate_candidates(track_plate_candidates.get(track_id, []), min_confidence=0.0)
                _update_track_plate_ocr_state(
                    track_plate_ocr_state.setdefault(track_id, _new_track_plate_ocr_state()),
                    summary,
                    min_confidence=plate_min_confidence,
                )
                if summary is None:
                    continue
                detection["track_plate_text"] = str(summary.get("plate_text", "") or "")
                detection["track_plate_confidence"] = float(summary.get("plate_confidence", 0.0) or 0.0)
                detection["track_plate_support_count"] = int(summary.get("plate_support_count", 0) or 0)
                detection["track_plate_type"] = str(summary.get("plate_type", "") or "")
                detection["track_plate_source_frame_indices"] = [
                    int(v) for v in list(summary.get("plate_source_frame_indices", []))[:12]
                ]

                if bool(detection.get("is_violating", False)) and not str(detection.get("plate_text", "") or "").strip():
                    local_summary = _summarize_violation_plate_candidates(
                        track_plate_candidates.get(track_id, []),
                        frame_start=frame_data.frame_index,
                        frame_end=frame_data.frame_index,
                        window_frames=violation_plate_window_frames,
                        min_confidence=0.0,
                    )
                    if local_summary is None:
                        continue
                    detection["plate_text"] = str(local_summary.get("plate_text", "") or "")
                    detection["plate_confidence"] = float(local_summary.get("plate_confidence", 0.0) or 0.0)
                    detection["plate_support_count"] = int(local_summary.get("plate_support_count", 0) or 0)
                    detection["plate_type"] = str(local_summary.get("plate_type", "") or "")
                    detection["plate_type_id"] = int(local_summary.get("plate_type_id", -1) or -1)
                    detection["plate_detect_confidence"] = float(local_summary.get("plate_detect_confidence", 0.0) or 0.0)
                    detection["plate_source_frame_indices"] = [
                        int(v) for v in list(local_summary.get("plate_source_frame_indices", []))[:12]
                    ]
                    detection["plate_crop_strategy"] = "track_fused_windowed"
                    detection["plate_status"] = "track_fused"

            current_detected_plate_count = int(
                sum(
                    1
                    for detection in tracked_detections
                    if str(detection.get("plate_text", "") or detection.get("track_plate_text", "")).strip()
                )
            )
            lane_mask_for_render = None
            if frame_result.get("lane_polygons"):
                lane_mask_for_render = _build_render_lane_mask(
                    frame_data.frame.shape,
                    frame_result.get("lane_polygons", []),
                )
            rendered = render_result(
                frame_data.frame,
                lane_mask_for_render,
                tracked_detections,
                layers=layers,
            )

            vehicle_count = int(frame_result.get("vehicle_count", 0))
            violation_count = int(frame_violation_count)
            total_vehicle_instances += vehicle_count
            total_violation_instances += violation_count
            max_vehicle_count = max(max_vehicle_count, vehicle_count)
            if violation_count > 0:
                frames_with_violation += 1

            forward_total = sum(1 for event in count_events if event.get("direction") == "forward")
            backward_total = sum(1 for event in count_events if event.get("direction") == "backward")
            _annotate_rendered_frame(
                rendered,
                tracked_detections,
                count_rules,
                counters,
                labels_enabled=bool((layers or {}).get("labels", False)),
                confirmed_track_count=tracker.track_count(confirmed_only=True),
                count_total=len(count_events),
                forward_total=forward_total,
                backward_total=backward_total,
                current_violation_count=violation_count,
            )

            writer.write(rendered)
            processed_frame_count += 1
            lane_sources.add(str(frame_result.get("lane_source", "auto")))

            should_write_preview = (
                not preview_written
                or int(frame_data.frame_index) == int(info.preview_frame_index)
            )
            if should_write_preview and cv2.imwrite(str(output_preview_path), rendered):
                preview_written = True

            elapsed_s = max(1e-6, time.monotonic() - run_started_at)
            processed_frames_per_s = float(processed_frame_count) / elapsed_s
            if expected_processed_frames > 0:
                progress_percent = min(100.0, (float(processed_frame_count) / float(expected_processed_frames)) * 100.0)
                remaining_frames = max(0, expected_processed_frames - processed_frame_count)
                estimated_remaining_s = remaining_frames / max(processed_frames_per_s, 1e-6)
            else:
                progress_percent = 0.0
                estimated_remaining_s = 0.0

            _safe_progress_callback(
                progress_callback,
                {
                    "media_type": "video",
                    "status": "running",
                    "original_path": str(video_path),
                    "processed_video_path": str(output_video_path),
                    "preview_frame_path": str(output_preview_path) if output_preview_path.exists() else "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_count": int(info.frame_count),
                    "processed_frame_count": int(processed_frame_count),
                    "expected_processed_frame_count": int(expected_processed_frames),
                    "fps": float(info.fps),
                    "output_fps": float(output_fps),
                    "duration_s": float(info.duration_s),
                    "image_width": int(info.width),
                    "image_height": int(info.height),
                    "preview_frame_index": int(info.preview_frame_index),
                    "codec": str(info.codec),
                    "frame_stride": int(frame_stride),
                    "violation_plate_window_frames": int(violation_plate_window_frames),
                    "violation_plate_window_s": float(DEFAULT_VIOLATION_PLATE_WINDOW_S),
                    "current_frame_index": int(frame_data.frame_index),
                    "progress_percent": float(progress_percent),
                    "processed_frames_per_s": float(processed_frames_per_s),
                    "elapsed_processing_s": float(elapsed_s),
                    "estimated_remaining_s": float(estimated_remaining_s),
                    "lane_source": (
                        "none"
                        if not lane_sources or lane_sources == {"none"}
                        else ("manual" if lane_sources == {"manual"} else ("auto" if lane_sources == {"auto"} else "mixed"))
                    ),
                    "count_line_source": count_line_source or "none",
                    "count_line_count": int(len(count_rules)),
                    "count_line_names": [rule.name for rule in count_rules],
                    "scene_region_count": int(len(scene_regions or [])),
                    "region_rule_event_count": int(len(region_rule_events)),
                    "region_rule_no_parking_count": int(sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_PARKING)),
                    "region_rule_no_non_motor_count": int(sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_NON_MOTOR)),
                    "region_rule_no_wrong_way_count": int(sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_WRONG_WAY)),
                    "current_vehicle_count": int(vehicle_count),
                    "current_violation_count": int(violation_count),
                    "current_region_rule_violation_count": int(frame_rule_violation_count),
                    "current_detected_plate_count": int(current_detected_plate_count),
                    "current_plate_ocr_attempt_count": int(frame_plate_ocr_attempt_count),
                    "current_plate_ocr_success_count": int(frame_plate_ocr_success_count),
                    "total_vehicle_instances": int(total_vehicle_instances),
                    "max_vehicle_count": int(max_vehicle_count),
                    "frames_with_violation": int(frames_with_violation),
                    "frames_with_region_rule_violation": int(len(frames_with_region_rule_violation)),
                    "total_violation_instances": int(total_violation_instances),
                    "total_plate_ocr_attempt_count": int(total_plate_ocr_attempt_count),
                    "total_plate_ocr_success_count": int(total_plate_ocr_success_count),
                    "track_count": int(len(tracker.all_tracks())),
                    "confirmed_track_count": int(tracker.track_count(confirmed_only=True)),
                    "moving_track_count": int(sum(1 for track in tracker.all_tracks() if track.hits >= tracker.min_confirm_hits and track.displacement_px() >= 12.0)),
                    "traffic_count_total": int(len(count_events)),
                    "traffic_count_forward": int(forward_total),
                    "traffic_count_backward": int(backward_total),
                    "frames_with_count_event": int(len(frames_with_count_event)),
                    "any_violation": bool(frames_with_violation > 0),
                },
            )
            if callable(stop_requested) and stop_requested():
                raise InterruptedError(f"Video processing interrupted: {video_path}")
        writer.release()
    except InterruptedError:
        writer.release()
        raise
    except Exception:
        writer.release()
        raise

    if processed_frame_count == 0:
        raise RuntimeError(f"No frames were processed for video: {video_path}")

    all_tracks = tracker.all_tracks()
    confirmed_tracks = [track for track in all_tracks if track.hits >= tracker.min_confirm_hits]
    moving_tracks = [track for track in confirmed_tracks if track.displacement_px() >= 12.0]
    forward_total = sum(1 for event in count_events if event.get("direction") == "forward")
    backward_total = sum(1 for event in count_events if event.get("direction") == "backward")
    no_parking_total = sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_PARKING)
    no_non_motor_total = sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_NON_MOTOR)
    no_wrong_way_total = sum(1 for event in region_rule_events if event.get("rule_type") == RULE_NO_WRONG_WAY)

    if not lane_sources or lane_sources == {"none"}:
        lane_source = "none"
    elif lane_sources == {"manual"}:
        lane_source = "manual"
    elif lane_sources == {"auto"}:
        lane_source = "auto"
    else:
        lane_source = "mixed"

    violating_track_plates = []
    for track_id, meta in violating_track_meta.items():
        summary = _summarize_violation_plate_candidates(
            track_plate_candidates.get(track_id, []),
            frame_start=meta.get("first_violation_frame"),
            frame_end=meta.get("last_violation_frame"),
            window_frames=violation_plate_window_frames,
            min_confidence=0.0,
        )
        if summary is None:
            continue
        violating_track_plates.append(
            {
                "track_id": str(track_id),
                "vehicle_type": str(meta.get("vehicle_type", "vehicle")),
                "plate_text": str(summary.get("plate_text", "")),
                "plate_confidence": float(summary.get("plate_confidence", 0.0) or 0.0),
                "plate_support_count": int(summary.get("plate_support_count", 0) or 0),
                "plate_type": str(summary.get("plate_type", "") or ""),
                "source_frame_indices": [int(v) for v in list(summary.get("plate_source_frame_indices", []))],
                "max_violation_ratio": float(meta.get("max_violation_ratio", 0.0) or 0.0),
                "first_violation_frame": None
                if meta.get("first_violation_frame") is None
                else int(meta.get("first_violation_frame")),
                "last_violation_frame": None
                if meta.get("last_violation_frame") is None
                else int(meta.get("last_violation_frame")),
            }
        )
    violating_track_plates.sort(
        key=lambda item: (
            int(item.get("plate_support_count", 0)),
            float(item.get("plate_confidence", 0.0)),
            float(item.get("max_violation_ratio", 0.0)),
        ),
        reverse=True,
    )
    unread_violating_track_count = max(0, len(violating_track_meta) - len(violating_track_plates))

    return {
        "media_type": "video",
        "original_path": str(video_path),
        "processed_video_path": str(output_video_path),
        "preview_frame_path": str(output_preview_path) if output_preview_path.exists() else "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_count": int(info.frame_count),
        "processed_frame_count": int(processed_frame_count),
        "expected_processed_frame_count": int(expected_processed_frames),
        "fps": float(info.fps),
        "output_fps": float(output_fps),
        "duration_s": float(info.duration_s),
        "image_width": int(info.width),
        "image_height": int(info.height),
        "preview_frame_index": int(info.preview_frame_index),
        "codec": str(info.codec),
        "frame_stride": int(frame_stride),
        "violation_plate_window_frames": int(violation_plate_window_frames),
        "violation_plate_window_s": float(DEFAULT_VIOLATION_PLATE_WINDOW_S),
        "lane_source": lane_source,
        "count_line_source": count_line_source or "none",
        "count_line_count": int(len(count_rules)),
        "count_line_names": [rule.name for rule in count_rules],
        "scene_region_count": int(len(scene_regions or [])),
        "region_rule_event_count": int(len(region_rule_events)),
        "region_rule_no_parking_count": int(no_parking_total),
        "region_rule_no_non_motor_count": int(no_non_motor_total),
        "region_rule_no_wrong_way_count": int(no_wrong_way_total),
        "frames_with_region_rule_violation": int(len(frames_with_region_rule_violation)),
        "region_rule_events": region_rule_events,
        "total_vehicle_instances": int(total_vehicle_instances),
        "max_vehicle_count": int(max_vehicle_count),
        "frames_with_violation": int(frames_with_violation),
        "total_violation_instances": int(total_violation_instances),
        "total_plate_ocr_attempt_count": int(total_plate_ocr_attempt_count),
        "total_plate_ocr_success_count": int(total_plate_ocr_success_count),
        "violating_track_count": int(len(violating_track_meta)),
        "violating_track_plate_count": int(len(violating_track_plates)),
        "unread_violating_track_count": int(unread_violating_track_count),
        "violating_track_plates": violating_track_plates,
        "track_count": int(len(all_tracks)),
        "confirmed_track_count": int(len(confirmed_tracks)),
        "moving_track_count": int(len(moving_tracks)),
        "traffic_count_total": int(len(count_events)),
        "traffic_count_forward": int(forward_total),
        "traffic_count_backward": int(backward_total),
        "frames_with_count_event": int(len(frames_with_count_event)),
        "count_events": count_events,
        "any_violation": bool(frames_with_violation > 0),
        "status": "processed",
    }
