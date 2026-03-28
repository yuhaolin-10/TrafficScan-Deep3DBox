from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from .renderer import render_result
    from ..core.plate_recognizer import PLATE_VEHICLE_TYPES, recognize_vehicle_plate
except Exception:
    from services.renderer import render_result
    from core.plate_recognizer import PLATE_VEHICLE_TYPES, recognize_vehicle_plate


def _to_int_points(points):
    arr = np.array(points, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return []
    return arr.tolist()


def _to_float_point(point):
    arr = np.array(point, dtype=np.float32).reshape(-1)
    if arr.size != 2:
        return []
    return [float(arr[0]), float(arr[1])]


def _normalize_lane_polygons(polygons):
    valid = []
    for polygon in polygons or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            continue
        valid.append(np.round(pts).astype(np.int32))
    return valid


def _build_lane_mask(image_shape, lane_polygons):
    height, width = image_shape[:2]
    lane_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in _normalize_lane_polygons(lane_polygons):
        cv2.fillPoly(lane_mask, [polygon], 255)
    return lane_mask


def _normalize_plate_mode(plate_mode):
    mode = str(plate_mode or "violating_only").strip().lower()
    if mode not in {"violating_only", "eligible_vehicle", "disabled"}:
        return "violating_only"
    return mode


def _should_attempt_plate_recognition(vehicle_type, is_violating, plate_mode):
    if plate_mode == "disabled":
        return False
    if plate_mode == "violating_only" and not bool(is_violating):
        return False
    vehicle_key = str(vehicle_type or "").strip().lower()
    if vehicle_key and vehicle_key not in PLATE_VEHICLE_TYPES:
        return False
    return True


def _summarize_violating_plates(detections):
    plates = []
    missing_count = 0
    for det in detections or []:
        if not bool(det.get("is_violating", False)):
            continue
        plate_text = str(det.get("plate_text", "") or "").strip()
        if not plate_text:
            missing_count += 1
            continue
        plates.append(
            {
                "index": int(det.get("index", 0) or 0),
                "vehicle_type": str(det.get("vehicle_type", "vehicle")),
                "plate_text": plate_text,
                "plate_confidence": float(det.get("plate_confidence", 0.0) or 0.0),
                "plate_support_count": int(det.get("plate_support_count", 0) or 0),
                "plate_type": str(det.get("plate_type", "") or ""),
                "violation_ratio": float(det.get("violation_ratio", 0.0) or 0.0),
            }
        )
    plates.sort(
        key=lambda item: (
            float(item.get("plate_confidence", 0.0)),
            float(item.get("violation_ratio", 0.0)),
        ),
        reverse=True,
    )
    return plates, missing_count


def process_frame(
    frame,
    lane_detector,
    vehicle_detector,
    violation_checker,
    *,
    plate_recognizer=None,
    plate_mode="violating_only",
    layers=None,
    lane_override_polygons=None,
    timestamp=None,
    frame_index=None,
    timestamp_s=None,
    render_output=True,
):
    plate_mode = _normalize_plate_mode(plate_mode)
    manual_lane_polygons = _normalize_lane_polygons(lane_override_polygons)
    if manual_lane_polygons:
        lane_polygons = manual_lane_polygons
        lane_mask = _build_lane_mask(frame.shape, lane_polygons)
        lane_source = "manual"
    else:
        lane_mask = None
        lane_polygons = []
        lane_source = "none"

    vehicles = vehicle_detector.detect(frame, lane_polygons=lane_polygons)
    lane_area_px = int(np.count_nonzero(lane_mask)) if lane_mask is not None else 0

    detections = []
    violation_count = 0

    for idx, veh in enumerate(vehicles, start=1):
        footprint = np.array(veh.get("footprint_2d", []), dtype=np.int32)
        corners_2d = np.array(veh.get("corners_2d", []), dtype=np.int32)
        bbox = [float(v) for v in veh.get("bbox", [0, 0, 0, 0])]
        vehicle_type = str(veh.get("type", "vehicle"))

        if footprint.ndim != 2 or footprint.shape[0] < 3:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            footprint = np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]], dtype=np.int32)

        is_violating, ratio = violation_checker.check(footprint, lane_mask)
        if is_violating:
            violation_count += 1

        plate_data = None
        plate_attempted = False
        if plate_recognizer is not None and bool(getattr(plate_recognizer, "enabled", False)):
            plate_attempted = _should_attempt_plate_recognition(vehicle_type, is_violating, plate_mode)
        if plate_attempted:
            try:
                plate_data = recognize_vehicle_plate(
                    frame,
                    bbox,
                    plate_recognizer,
                    vehicle_type=vehicle_type,
                    frame_index=frame_index,
                )
            except Exception:
                plate_data = None

        footprint_area_px = float(abs(cv2.contourArea(footprint))) if footprint.shape[0] >= 3 else 0.0
        depth_hint = float(np.max(footprint[:, 1])) if footprint.shape[0] > 0 else 0.0
        anchor = footprint[0].tolist() if footprint.shape[0] > 0 else None
        detection = {
            "index": idx,
            "vehicle_type": vehicle_type,
            "confidence": float(veh.get("conf", 0.0)),
            "bbox": bbox,
            "footprint": _to_int_points(footprint),
            "corners_2d": _to_int_points(corners_2d),
            "mask_polygon": _to_int_points(veh.get("mask_polygon_2d", [])),
            "is_violating": bool(is_violating),
            "violation_ratio": float(ratio),
            "violation_type": "Emergency Lane Occupation" if is_violating else "",
            "footprint_area_px": footprint_area_px,
            "depth_hint": depth_hint,
            "label_anchor": anchor,
            "yaw": float(veh.get("yaw", 0.0)),
            "plate_status": (
                "detected"
                if plate_data is not None
                else ("not_found" if plate_attempted else "skipped")
            ),
            "plate_text": "",
            "plate_confidence": 0.0,
            "plate_support_count": 0,
            "plate_source_frame_indices": [],
            "plate_type": "",
            "plate_type_id": -1,
            "plate_detect_confidence": 0.0,
            "plate_box": [],
            "plate_crop_strategy": "",
            "plate_candidates": [],
            "yaw_debug": {
                "magnitude": float(veh.get("yaw_debug", {}).get("magnitude", 0.0)),
                "perspective_magnitude": float(veh.get("yaw_debug", {}).get("perspective_magnitude", 0.0) or 0.0),
                "lane_prior_yaw_abs": float(veh.get("yaw_debug", {}).get("lane_prior_yaw_abs", 0.0) or 0.0),
                "lane_weight": float(veh.get("yaw_debug", {}).get("lane_weight", 0.0)),
                "lane_prior_confidence": float(veh.get("yaw_debug", {}).get("lane_prior_confidence", 0.0)),
                "preferred_sign": float(veh.get("yaw_debug", {}).get("preferred_sign", 0.0)),
                "visual_sign": float(veh.get("yaw_debug", {}).get("visual_sign", 0.0)),
                "vanishing_point_source": str(veh.get("yaw_debug", {}).get("vanishing_point_source", "none")),
                "vanishing_point_confidence": float(veh.get("yaw_debug", {}).get("vanishing_point_confidence", 0.0)),
            },
            "yaw_debug_vectors": {
                "center": _to_float_point(veh.get("geometry_debug", {}).get("vehicle_center_2d", [])),
                "vanishing_point": _to_float_point(veh.get("yaw_debug", {}).get("vanishing_point", [])),
                "perspective_reference": {
                    "start": _to_float_point(veh.get("geometry_debug", {}).get("vehicle_center_2d", [])),
                    "end": _to_float_point(veh.get("yaw_debug", {}).get("vanishing_point", [])),
                },
                "lane_prior": {
                    "direction": _to_float_point(veh.get("lane_prior", {}).get("direction", [])),
                    "confidence": float(veh.get("lane_prior", {}).get("confidence", 0.0) or 0.0),
                },
                "perspective_direction": _to_float_point(veh.get("yaw_debug", {}).get("perspective_direction", [])),
                "final_direction": _to_float_point(veh.get("yaw_debug", {}).get("final_direction", [])),
            },
        }
        if plate_data is not None:
            detection.update(plate_data)
        detections.append(detection)

    violating_plates, missing_plate_count = _summarize_violating_plates(detections)

    rendered = render_result(frame, lane_mask, detections, layers=layers) if bool(render_output) else None
    result = {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "lane_area_px": lane_area_px,
        "lane_source": lane_source,
        "vehicle_count": len(detections),
        "violation_count": violation_count,
        "any_violation": violation_count > 0,
        "violating_plate_count": len(violating_plates),
        "violating_plate_missing_count": int(missing_plate_count),
        "violating_plates": violating_plates,
        "lane_polygons": [polygon.tolist() for polygon in lane_polygons],
        "detections": detections,
    }
    if frame_index is not None:
        result["frame_index"] = int(frame_index)
    if timestamp_s is not None:
        result["timestamp_s"] = float(timestamp_s)
    return result, rendered


def process_image(
    image_path,
    lane_detector,
    vehicle_detector,
    violation_checker,
    output_dir,
    plate_recognizer=None,
    layers=None,
    lane_override_polygons=None,
):
    """
    Run full processing for one image and return a structured result package.
    """
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    frame_result, rendered = process_frame(
        frame,
        lane_detector,
        vehicle_detector,
        violation_checker,
        plate_recognizer=plate_recognizer,
        layers=layers,
        lane_override_polygons=lane_override_polygons,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = f"{ts}_{image_path.stem}.jpg"
    save_path = output_dir / save_name
    if not cv2.imwrite(str(save_path), rendered):
        raise RuntimeError(f"Failed to write processed image: {save_path}")

    return {
        "original_path": str(image_path),
        "processed_path": str(save_path),
        **frame_result,
    }
