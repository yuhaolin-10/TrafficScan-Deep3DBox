from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MIN_BOUNDARY_POINTS = 6
MIN_BOUNDARY_WIDTH_PX = 6
ROW_SAMPLE_STEP = 4
BOUNDARY_SMOOTH_WINDOW = 9
LINE_RESIDUAL_THRESHOLD_PX = 3.5
BOUNDARY_ENDPOINT_TRIM_RATIO = 0.08
BOUNDARY_ENDPOINT_TRIM_MIN_PX = 12.0
BOUNDARY_SELECTION_MAX_DIST_RATIO = 0.08
CURVE_MERGE_MAX_GAP_RATIO = 0.24
CURVE_MERGE_MAX_X_GAP_RATIO = 0.07
CURVE_MERGE_MAX_SLOPE_DELTA = 0.22
SIDE_PREFERENCE_MIN_QUALITY = 0.46
SIDE_PREFERENCE_MIN_GAP = 0.10
SIDE_PREFERENCE_MAX_BONUS = 0.24
SIDE_PREFERENCE_MAX_PENALTY = 0.12
LANE_ARROW_LEN_RATIO = 0.18
LANE_ARROW_LEN_MIN = 48.0
LANE_ARROW_LEN_MAX = 132.0
ANCHOR_LOCAL_DIRECTION_WEIGHT = 0.80
ANCHOR_SAME_LANE_DIRECTION_WEIGHT = 1.00
ANCHOR_NEIGHBOR_LANE_DIRECTION_WEIGHT = 0.42
VP_MIN_CURVE_QUALITY = 0.40
VP_MIN_SLOPE_DELTA = 0.05
VP_MAX_X_MARGIN_RATIO = 0.75
VP_MAX_Y_RATIO = 0.70
VP_MIN_Y_RATIO = -0.80


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 2:
        return values.astype(np.float32)

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, int(values.size) if int(values.size) % 2 == 1 else int(values.size) - 1)
    if window < 3:
        return values.astype(np.float32)

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _direction_from_slope(slope: float) -> np.ndarray:
    direction = _normalize(np.array([-float(slope), -1.0], dtype=np.float32))
    if direction[1] > 0:
        direction = -direction
    return direction


def _valid_polygon(poly) -> Optional[np.ndarray]:
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None
    return pts


def _build_mask_from_polygons(
    lane_polygons: Sequence[Sequence[Sequence[float]]],
    image_shape,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    valid_polygons: List[np.ndarray] = []
    for poly in lane_polygons:
        pts = _valid_polygon(poly)
        if pts is None:
            continue
        pts_int = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_int], 255)
        valid_polygons.append(pts_int)
    return mask, valid_polygons


def _prepare_boundary_curve(raw_points, image_shape, side: str) -> Optional[Dict[str, object]]:
    pts = np.asarray(raw_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < MIN_BOUNDARY_POINTS or pts.shape[1] != 2:
        return None

    order = np.argsort(pts[:, 1])
    pts = pts[order]
    ys = pts[:, 1]
    xs = pts[:, 0]

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if (y_max - y_min) < 12.0:
        return None

    y_grid = np.arange(y_min, y_max + ROW_SAMPLE_STEP, ROW_SAMPLE_STEP, dtype=np.float32)
    if y_grid.size < MIN_BOUNDARY_POINTS:
        y_grid = np.linspace(y_min, y_max, num=max(MIN_BOUNDARY_POINTS, int(pts.shape[0])), dtype=np.float32)

    x_interp = np.interp(y_grid, ys, xs).astype(np.float32)
    window = min(BOUNDARY_SMOOTH_WINDOW, int(y_grid.size) if int(y_grid.size) % 2 == 1 else int(y_grid.size) - 1)
    x_smooth = _moving_average(x_interp, max(3, window))
    smooth_points = np.column_stack([x_smooth, y_grid]).astype(np.float32)

    trim_margin = max(BOUNDARY_ENDPOINT_TRIM_MIN_PX, (y_max - y_min) * BOUNDARY_ENDPOINT_TRIM_RATIO)
    fit_mask = (smooth_points[:, 1] >= (y_min + trim_margin)) & (smooth_points[:, 1] <= (y_max - trim_margin))
    fit_points = smooth_points[fit_mask]
    if fit_points.shape[0] < MIN_BOUNDARY_POINTS:
        fit_points = smooth_points

    line_coeff = np.polyfit(fit_points[:, 1], fit_points[:, 0], deg=1)
    x_line = np.polyval(line_coeff, fit_points[:, 1]).astype(np.float32)
    line_residual = float(np.sqrt(np.mean((fit_points[:, 0] - x_line) ** 2)))
    mode = "line" if line_residual <= LINE_RESIDUAL_THRESHOLD_PX else "curve"

    return {
        "side": side,
        "points": smooth_points,
        "fit_points": fit_points,
        "line_coeff": [float(line_coeff[0]), float(line_coeff[1])],
        "line_residual": line_residual,
        "mode": mode,
        "support_span_y": float(y_max - y_min),
        "point_count": int(smooth_points.shape[0]),
        "y_min": y_min,
        "y_max": y_max,
        "fit_y_min": float(np.min(fit_points[:, 1])),
        "fit_y_max": float(np.max(fit_points[:, 1])),
    }


def _extract_polygon_boundary_curves(poly: np.ndarray, image_shape) -> List[Dict[str, object]]:
    h, w = image_shape[:2]
    pts = _valid_polygon(poly)
    if pts is None:
        return []

    x1 = int(np.clip(np.floor(np.min(pts[:, 0])), 0, w - 1))
    x2 = int(np.clip(np.ceil(np.max(pts[:, 0])), 0, w - 1))
    y1 = int(np.clip(np.floor(np.min(pts[:, 1])), 0, h - 1))
    y2 = int(np.clip(np.ceil(np.max(pts[:, 1])), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return []

    roi_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    shifted = np.round(pts).astype(np.int32)
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(roi_mask, [shifted], 255)

    row_step = max(2, int(round(max(1.0, float(h) / 160.0))))
    left_points: List[List[float]] = []
    right_points: List[List[float]] = []
    for y_local in range(0, roi_mask.shape[0], row_step):
        xs = np.flatnonzero(roi_mask[y_local] > 0)
        if xs.size < MIN_BOUNDARY_WIDTH_PX:
            continue
        y_global = float(y1 + y_local)
        left_points.append([float(x1 + xs[0]), y_global])
        right_points.append([float(x1 + xs[-1]), y_global])

    curves = []
    for side, raw_points in (("left", left_points), ("right", right_points)):
        curve = _prepare_boundary_curve(raw_points, image_shape=image_shape, side=side)
        if curve is not None:
            curves.append(curve)
    return curves


def _curve_anchor_distance(curve: Dict[str, object], anchor: np.ndarray) -> Tuple[float, float, float]:
    pts = np.asarray(curve["points"], dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    anchor_y = float(anchor[1])
    clamp_min = float(curve.get("fit_y_min", float(np.min(ys))))
    clamp_max = float(curve.get("fit_y_max", float(np.max(ys))))
    clamped_y = float(np.clip(anchor_y, clamp_min, clamp_max))
    x_at_y = float(np.interp(clamped_y, ys, xs))
    dx = abs(float(anchor[0]) - x_at_y)
    dy = abs(anchor_y - clamped_y)
    return dx + (0.25 * dy), x_at_y, clamped_y


def _curve_direction_near_anchor(
    curve: Dict[str, object],
    anchor: np.ndarray,
    image_shape,
) -> Dict[str, object]:
    h, w = image_shape[:2]
    pts = np.asarray(curve["points"], dtype=np.float32)
    fit_pts = np.asarray(curve.get("fit_points", pts), dtype=np.float32)
    ys = fit_pts[:, 1]

    anchor_distance, x_at_y, y_ref = _curve_anchor_distance(curve, anchor)
    if curve["mode"] == "line":
        slope = float(curve["line_coeff"][0])
        residual = float(curve["line_residual"])
        local_points = fit_pts
    else:
        idx = int(np.argmin(np.abs(ys - y_ref)))
        half = min(5, max(2, fit_pts.shape[0] // 4))
        local_points = fit_pts[max(0, idx - half) : min(fit_pts.shape[0], idx + half + 1)]
        if local_points.shape[0] < 3:
            local_points = fit_pts
        coeff = np.polyfit(local_points[:, 1], local_points[:, 0], deg=1)
        pred = np.polyval(coeff, local_points[:, 1]).astype(np.float32)
        slope = float(coeff[0])
        residual = float(np.sqrt(np.mean((local_points[:, 0] - pred) ** 2)))

    direction = _direction_from_slope(slope)

    support_score = float(
        np.clip(float(curve["support_span_y"]) / max(48.0, float(h) * 0.45), 0.0, 1.0)
    )
    residual_score = float(np.clip(1.0 - (residual / 10.0), 0.0, 1.0))
    distance_score = float(np.clip(1.0 - (anchor_distance / max(28.0, float(w) * 0.08)), 0.0, 1.0))
    confidence = 0.42 * support_score + 0.38 * residual_score + 0.20 * distance_score

    return {
        "direction": direction,
        "reference_point": [float(x_at_y), float(y_ref)],
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "fit_mode": curve["mode"],
        "fit_residual": residual,
        "anchor_distance": float(anchor_distance),
        "curve_side": curve["side"],
    }


def _curve_global_direction(curve: Dict[str, object], image_shape) -> Dict[str, object]:
    direction = _direction_from_slope(float(curve["line_coeff"][0]))
    quality = float(curve.get("quality_score", _curve_quality_score(curve, image_shape)))
    support_score = float(
        np.clip(float(curve["support_span_y"]) / max(48.0, float(image_shape[0]) * 0.45), 0.0, 1.0)
    )
    confidence = float(np.clip(0.65 * quality + 0.35 * support_score, 0.0, 1.0))
    return {
        "direction": direction,
        "confidence": confidence,
        "quality": quality,
        "curve_side": str(curve["side"]),
        "fit_mode": str(curve["mode"]),
        "fit_residual": float(curve["line_residual"]),
    }


def _curve_quality_score(curve: Dict[str, object], image_shape) -> float:
    h, _ = image_shape[:2]
    support_score = float(np.clip(float(curve["support_span_y"]) / max(48.0, float(h) * 0.45), 0.0, 1.0))
    residual = float(curve.get("line_residual", 10.0))
    point_count = int(np.asarray(curve.get("fit_points", curve["points"]), dtype=np.float32).shape[0])
    residual_score = float(np.clip(1.0 - (residual / 9.0), 0.0, 1.0))
    density_score = float(np.clip(float(point_count) / 28.0, 0.0, 1.0))
    mode_bonus = 1.0 if str(curve.get("mode", "curve")) == "line" else 0.85
    quality = 0.44 * residual_score + 0.34 * support_score + 0.12 * density_score + 0.10 * mode_bonus
    return float(np.clip(quality, 0.0, 1.0))


def _curve_endpoints(curve: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    fit_pts = np.asarray(curve.get("fit_points", curve["points"]), dtype=np.float32)
    order = np.argsort(fit_pts[:, 1])
    fit_pts = fit_pts[order]
    return fit_pts[0], fit_pts[-1]


def _line_intersection_x_of_y(
    coeff_a: Sequence[float],
    coeff_b: Sequence[float],
) -> Optional[np.ndarray]:
    a1, b1 = float(coeff_a[0]), float(coeff_a[1])
    a2, b2 = float(coeff_b[0]), float(coeff_b[1])
    if abs(a1 - a2) <= 1e-6:
        return None
    y = (b2 - b1) / (a1 - a2)
    x = a1 * y + b1
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([float(x), float(y)], dtype=np.float32)


def _curve_merge_compatible(curve_a: Dict[str, object], curve_b: Dict[str, object], image_shape) -> bool:
    if curve_a["side"] != curve_b["side"]:
        return False

    h, w = image_shape[:2]
    a_top, a_bottom = _curve_endpoints(curve_a)
    b_top, b_bottom = _curve_endpoints(curve_b)
    if a_top[1] > b_top[1]:
        curve_a, curve_b = curve_b, curve_a
        a_top, a_bottom, b_top, b_bottom = b_top, b_bottom, a_top, a_bottom

    y_gap = float(b_top[1] - a_bottom[1])
    if y_gap < 0.0 or y_gap > (float(h) * CURVE_MERGE_MAX_GAP_RATIO):
        return False

    slope_a = float(curve_a["line_coeff"][0])
    slope_b = float(curve_b["line_coeff"][0])
    if abs(slope_a - slope_b) > CURVE_MERGE_MAX_SLOPE_DELTA:
        return False

    x_pred = float(np.polyval(np.asarray(curve_a["line_coeff"], dtype=np.float32), b_top[1]))
    x_gap = abs(x_pred - float(b_top[0]))
    if x_gap > max(18.0, float(w) * CURVE_MERGE_MAX_X_GAP_RATIO):
        return False
    return True


def _merge_curve_group(curves: List[Dict[str, object]], image_shape) -> List[Dict[str, object]]:
    if len(curves) <= 1:
        return curves

    merged = list(curves)
    changed = True
    while changed:
        changed = False
        merged.sort(key=lambda curve: float(curve.get("fit_y_min", curve["y_min"])))
        new_curves: List[Dict[str, object]] = []
        skip_next = False
        for idx in range(len(merged)):
            if skip_next:
                skip_next = False
                continue

            current = merged[idx]
            if idx < (len(merged) - 1):
                candidate = merged[idx + 1]
                if _curve_merge_compatible(current, candidate, image_shape):
                    merged_points = np.vstack(
                        [
                            np.asarray(current["points"], dtype=np.float32),
                            np.asarray(candidate["points"], dtype=np.float32),
                        ]
                    )
                    polygon_index = int(min(current.get("polygon_index", 0), candidate.get("polygon_index", 0)))
                    rebuilt = _prepare_boundary_curve(merged_points, image_shape=image_shape, side=current["side"])
                    if rebuilt is not None:
                        rebuilt["polygon_index"] = polygon_index
                        new_curves.append(rebuilt)
                        skip_next = True
                        changed = True
                        continue

            new_curves.append(current)
        merged = new_curves
    return merged


def _build_polygon_side_preferences(
    curves: List[Dict[str, object]],
    image_shape,
) -> Dict[int, Dict[str, object]]:
    grouped: Dict[int, Dict[str, Dict[str, object]]] = {}
    for curve in curves:
        polygon_index = int(curve.get("polygon_index", -1))
        if polygon_index < 0:
            continue
        grouped.setdefault(polygon_index, {})
        side = str(curve["side"])
        quality = _curve_quality_score(curve, image_shape)
        curve["quality_score"] = quality
        prev = grouped[polygon_index].get(side)
        if prev is None or quality > float(prev["quality_score"]):
            grouped[polygon_index][side] = {
                "curve": curve,
                "quality_score": quality,
            }

    preferences: Dict[int, Dict[str, object]] = {}
    for polygon_index, side_map in grouped.items():
        left = side_map.get("left")
        right = side_map.get("right")
        preferred_side = None
        preferred_quality = 0.0
        quality_gap = 0.0

        if left and right:
            left_q = float(left["quality_score"])
            right_q = float(right["quality_score"])
            if left_q >= right_q:
                preferred_side = "left"
                preferred_quality = left_q
                quality_gap = left_q - right_q
            else:
                preferred_side = "right"
                preferred_quality = right_q
                quality_gap = right_q - left_q

            if preferred_quality < SIDE_PREFERENCE_MIN_QUALITY or quality_gap < SIDE_PREFERENCE_MIN_GAP:
                preferred_side = None
        elif left or right:
            only = left or right
            preferred_side = str(only["curve"]["side"])
            preferred_quality = float(only["quality_score"])
            quality_gap = preferred_quality
            if preferred_quality < SIDE_PREFERENCE_MIN_QUALITY:
                preferred_side = None

        preferences[polygon_index] = {
            "preferred_side": preferred_side,
            "preferred_quality": float(preferred_quality),
            "quality_gap": float(max(0.0, quality_gap)),
            "side_scores": {
                side: float(info["quality_score"])
                for side, info in side_map.items()
            },
        }
    return preferences


def _selection_score_for_anchor(
    curve: Dict[str, object],
    anchor: np.ndarray,
    image_shape,
    polygon_preference: Optional[Dict[str, object]] = None,
) -> Tuple[float, Dict[str, object]]:
    h, w = image_shape[:2]
    info = _curve_direction_near_anchor(curve, anchor, image_shape)
    anchor_distance = float(info["anchor_distance"])
    distance_score = float(np.clip(1.0 - (anchor_distance / max(24.0, float(w) * BOUNDARY_SELECTION_MAX_DIST_RATIO)), 0.0, 1.0))

    fit_y_min = float(curve.get("fit_y_min", curve["y_min"]))
    fit_y_max = float(curve.get("fit_y_max", curve["y_max"]))
    endpoint_margin = min(abs(float(info["reference_point"][1]) - fit_y_min), abs(fit_y_max - float(info["reference_point"][1])))
    endpoint_score = float(np.clip(endpoint_margin / max(18.0, float(h) * 0.08), 0.0, 1.0))

    support_score = float(np.clip(float(curve["support_span_y"]) / max(48.0, float(h) * 0.45), 0.0, 1.0))
    total = 0.45 * distance_score + 0.30 * float(info["confidence"]) + 0.15 * endpoint_score + 0.10 * support_score

    side_preference_bonus = 0.0
    side_quality_score = float(curve.get("quality_score", _curve_quality_score(curve, image_shape)))
    preferred_side = None
    if polygon_preference:
        preferred_side = polygon_preference.get("preferred_side")
        quality_gap = float(polygon_preference.get("quality_gap", 0.0))
        if preferred_side:
            strength = float(np.clip(quality_gap / 0.30, 0.0, 1.0))
            if str(curve["side"]) == str(preferred_side):
                side_preference_bonus = SIDE_PREFERENCE_MAX_BONUS * strength
            else:
                side_preference_bonus = -SIDE_PREFERENCE_MAX_PENALTY * strength
            total += side_preference_bonus

    info["side_preference_bonus"] = float(side_preference_bonus)
    info["preferred_side"] = preferred_side
    info["side_quality_score"] = side_quality_score
    return float(total), info


def _collect_boundary_curves(
    lane_polygons: Sequence[Sequence[Sequence[float]]],
    image_shape,
) -> List[Dict[str, object]]:
    curves: List[Dict[str, object]] = []
    for polygon_index, poly in enumerate(lane_polygons):
        for curve in _extract_polygon_boundary_curves(np.asarray(poly, dtype=np.float32), image_shape):
            curve["polygon_index"] = int(polygon_index)
            curves.append(curve)
    by_side: Dict[str, List[Dict[str, object]]] = {"left": [], "right": []}
    for curve in curves:
        by_side.setdefault(str(curve["side"]), []).append(curve)

    merged_curves: List[Dict[str, object]] = []
    for side_curves in by_side.values():
        merged_curves.extend(_merge_curve_group(side_curves, image_shape))
    return merged_curves


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return np.zeros(2, dtype=np.float32)
    moments = cv2.moments(pts.astype(np.float32))
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return np.array(
            [
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            ],
            dtype=np.float32,
        )
    return np.mean(pts, axis=0).astype(np.float32)


def _build_lane_direction_entries(
    valid_polygons: Sequence[np.ndarray],
    boundary_curves: List[Dict[str, object]],
    image_shape,
    polygon_preferences: Dict[int, Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for curve in boundary_curves:
        polygon_index = int(curve.get("polygon_index", -1))
        if polygon_index < 0:
            continue
        grouped.setdefault(polygon_index, []).append(curve)

    h, w = image_shape[:2]
    arrow_len = float(np.clip(LANE_ARROW_LEN_RATIO * max(h, w), LANE_ARROW_LEN_MIN, LANE_ARROW_LEN_MAX))
    entries: List[Dict[str, object]] = []

    for polygon_index, poly in enumerate(valid_polygons):
        polygon_curves = grouped.get(int(polygon_index), [])
        if not polygon_curves:
            continue

        polygon_preference = polygon_preferences.get(int(polygon_index), {})
        preferred_side = polygon_preference.get("preferred_side")

        vectors = []
        confidences = []
        qualities = []
        source_sides = []
        best_curve = None
        best_confidence = -1.0
        for curve in polygon_curves:
            info = _curve_global_direction(curve, image_shape)
            curve_confidence = float(info["confidence"])
            direction = np.asarray(info["direction"], dtype=np.float32)
            if direction.shape != (2,) or curve_confidence <= 1e-4:
                continue

            weight = curve_confidence
            if preferred_side:
                if str(curve["side"]) == str(preferred_side):
                    weight *= 1.0
                else:
                    weight *= 0.35

            vectors.append(direction * weight)
            confidences.append(curve_confidence)
            qualities.append(float(info["quality"]))
            source_sides.append(str(curve["side"]))
            if curve_confidence > best_confidence:
                best_curve = curve
                best_confidence = curve_confidence

        if not vectors:
            continue

        vector_sum = np.sum(np.stack(vectors, axis=0), axis=0)
        direction = _normalize(vector_sum)
        if np.linalg.norm(direction) <= 1e-6 and best_curve is not None:
            direction = np.asarray(_curve_global_direction(best_curve, image_shape)["direction"], dtype=np.float32)

        total_weight = float(np.sum(np.linalg.norm(np.stack(vectors, axis=0), axis=1)))
        agreement = float(np.clip(np.linalg.norm(vector_sum) / max(total_weight, 1e-6), 0.0, 1.0))
        base_confidence = float(max(confidences))
        confidence = float(np.clip(base_confidence * (0.76 + 0.24 * agreement), 0.0, 1.0))

        anchor = _polygon_centroid(np.asarray(poly, dtype=np.float32))
        start = anchor
        end = anchor + direction * arrow_len
        pts = np.asarray(poly, dtype=np.float32)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))

        entries.append(
            {
                "polygon_index": int(polygon_index),
                "anchor": [float(anchor[0]), float(anchor[1])],
                "start": [float(start[0]), float(start[1])],
                "end": [float(end[0]), float(end[1])],
                "direction": [float(direction[0]), float(direction[1])],
                "confidence": confidence,
                "preferred_side": preferred_side,
                "source_sides": source_sides,
                "side_scores": dict(polygon_preference.get("side_scores", {})),
                "quality": float(np.mean(qualities)) if qualities else confidence,
                "bbox": [x_min, y_min, x_max, y_max],
                "sort_key_x": float(anchor[0]),
                "left_neighbor_polygon_index": None,
                "right_neighbor_polygon_index": None,
                "neighbor_polygon_indices": [],
            }
        )

    entries.sort(key=lambda entry: float(entry["sort_key_x"]))
    for idx, entry in enumerate(entries):
        neighbors = []
        if idx > 0:
            left_idx = int(entries[idx - 1]["polygon_index"])
            entry["left_neighbor_polygon_index"] = left_idx
            neighbors.append(left_idx)
        if idx < (len(entries) - 1):
            right_idx = int(entries[idx + 1]["polygon_index"])
            entry["right_neighbor_polygon_index"] = right_idx
            neighbors.append(right_idx)
        entry["neighbor_polygon_indices"] = neighbors
        entry["lane_rank"] = int(idx)
    return entries


def _build_anchor_prior(
    *,
    anchor_index: int,
    anchor: np.ndarray,
    best_curve: Dict[str, object],
    best_info: Dict[str, object],
    lane_entries_by_polygon: Dict[int, Dict[str, object]],
) -> Dict[str, object]:
    polygon_index = int(best_curve.get("polygon_index", -1))
    lane_entry = lane_entries_by_polygon.get(polygon_index)

    components = []

    def _append_component(direction, confidence: float, weight: float, source: str, src_polygon_index=None):
        vec = np.asarray(direction, dtype=np.float32).reshape(-1)
        if vec.shape[0] != 2 or not np.all(np.isfinite(vec)):
            return
        vec = _normalize(vec)
        if np.linalg.norm(vec) <= 1e-6:
            return
        comp_weight = float(max(0.0, confidence) * max(0.0, weight))
        if comp_weight <= 1e-6:
            return
        components.append(
            {
                "vector": vec,
                "direction": [float(vec[0]), float(vec[1])],
                "weight": comp_weight,
                "source": source,
                "polygon_index": None if src_polygon_index is None else int(src_polygon_index),
                "confidence": float(confidence),
            }
        )

    local_direction = np.asarray(best_info["direction"], dtype=np.float32)
    local_confidence = float(best_info["confidence"])
    _append_component(
        local_direction,
        confidence=local_confidence,
        weight=ANCHOR_LOCAL_DIRECTION_WEIGHT,
        source="local_anchor",
        src_polygon_index=polygon_index,
    )

    lane_direction = None
    lane_confidence = 0.0
    neighbor_polygon_indices: List[int] = []
    if lane_entry is not None:
        lane_direction = np.asarray(lane_entry["direction"], dtype=np.float32)
        lane_confidence = float(lane_entry["confidence"])
        _append_component(
            lane_direction,
            confidence=lane_confidence,
            weight=ANCHOR_SAME_LANE_DIRECTION_WEIGHT,
            source="same_lane",
            src_polygon_index=polygon_index,
        )
        for key in ("left_neighbor_polygon_index", "right_neighbor_polygon_index"):
            neighbor_idx = lane_entry.get(key)
            if neighbor_idx is None:
                continue
            neighbor_idx = int(neighbor_idx)
            neighbor_entry = lane_entries_by_polygon.get(neighbor_idx)
            if neighbor_entry is None:
                continue
            neighbor_polygon_indices.append(neighbor_idx)
            _append_component(
                neighbor_entry["direction"],
                confidence=float(neighbor_entry["confidence"]),
                weight=ANCHOR_NEIGHBOR_LANE_DIRECTION_WEIGHT,
                source="adjacent_lane",
                src_polygon_index=neighbor_idx,
            )

    if components:
        weighted = np.stack([comp["vector"] * float(comp["weight"]) for comp in components], axis=0)
        total_vector = np.sum(weighted, axis=0)
        direction = _normalize(total_vector)
        total_weight = float(np.sum([comp["weight"] for comp in components]))
        agreement = float(np.clip(np.linalg.norm(total_vector) / max(total_weight, 1e-6), 0.0, 1.0))
        base_confidence = max(
            local_confidence,
            lane_confidence,
            max(float(comp["confidence"]) for comp in components),
        )
        confidence = float(np.clip(base_confidence * (0.78 + 0.22 * agreement), 0.0, 1.0))
    else:
        direction = _normalize(local_direction)
        confidence = float(np.clip(local_confidence, 0.0, 1.0))

    return {
        "anchor_index": int(anchor_index),
        "anchor": [float(anchor[0]), float(anchor[1])],
        "polygon_index": polygon_index,
        "direction": [float(direction[0]), float(direction[1])],
        "confidence": confidence,
        "local_direction": [float(local_direction[0]), float(local_direction[1])],
        "local_confidence": local_confidence,
        "lane_direction": (
            [float(lane_direction[0]), float(lane_direction[1])] if lane_direction is not None else None
        ),
        "lane_confidence": lane_confidence,
        "neighbor_polygon_indices": neighbor_polygon_indices,
        "component_count": int(len(components)),
        "components": [
            {
                "source": str(comp["source"]),
                "polygon_index": comp["polygon_index"],
                "direction": list(comp["direction"]),
                "weight": float(comp["weight"]),
                "confidence": float(comp["confidence"]),
            }
            for comp in components
        ],
    }


def _estimate_lane_vanishing_point(
    boundary_curves: List[Dict[str, object]],
    image_shape,
) -> Dict[str, object]:
    h, w = image_shape[:2]
    candidates: List[Dict[str, object]] = []

    for idx_a in range(len(boundary_curves)):
        curve_a = boundary_curves[idx_a]
        quality_a = float(curve_a.get("quality_score", _curve_quality_score(curve_a, image_shape)))
        if quality_a < VP_MIN_CURVE_QUALITY:
            continue

        for idx_b in range(idx_a + 1, len(boundary_curves)):
            curve_b = boundary_curves[idx_b]
            quality_b = float(curve_b.get("quality_score", _curve_quality_score(curve_b, image_shape)))
            if quality_b < VP_MIN_CURVE_QUALITY:
                continue

            slope_a = float(curve_a["line_coeff"][0])
            slope_b = float(curve_b["line_coeff"][0])
            slope_delta = abs(slope_a - slope_b)
            if slope_delta < VP_MIN_SLOPE_DELTA:
                continue

            intersection = _line_intersection_x_of_y(curve_a["line_coeff"], curve_b["line_coeff"])
            if intersection is None:
                continue

            x, y = float(intersection[0]), float(intersection[1])
            if x < (-float(w) * VP_MAX_X_MARGIN_RATIO) or x > (float(w) * (1.0 + VP_MAX_X_MARGIN_RATIO)):
                continue
            if y < (float(h) * VP_MIN_Y_RATIO) or y > (float(h) * VP_MAX_Y_RATIO):
                continue

            support_a = float(np.clip(float(curve_a["support_span_y"]) / max(48.0, float(h) * 0.45), 0.0, 1.0))
            support_b = float(np.clip(float(curve_b["support_span_y"]) / max(48.0, float(h) * 0.45), 0.0, 1.0))
            angle_strength = float(np.clip((slope_delta - VP_MIN_SLOPE_DELTA) / 0.28, 0.0, 1.0))
            cross_side_bonus = 1.0 if str(curve_a["side"]) != str(curve_b["side"]) else 0.82
            pair_weight = quality_a * quality_b * (0.55 + 0.45 * angle_strength) * ((support_a + support_b) * 0.5) * cross_side_bonus
            if pair_weight <= 1e-6:
                continue

            candidates.append(
                {
                    "point": [x, y],
                    "weight": float(pair_weight),
                    "quality": float((quality_a + quality_b) * 0.5),
                    "curve_indices": [int(idx_a), int(idx_b)],
                    "curve_sides": [str(curve_a["side"]), str(curve_b["side"])],
                    "polygon_indices": [int(curve_a.get("polygon_index", -1)), int(curve_b.get("polygon_index", -1))],
                }
            )

    if not candidates:
        return {
            "point": None,
            "confidence": 0.0,
            "candidate_count": 0,
            "inlier_count": 0,
            "source": "default",
            "candidates": [],
        }

    points = np.asarray([item["point"] for item in candidates], dtype=np.float32)
    weights = np.asarray([float(item["weight"]) for item in candidates], dtype=np.float32)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-6:
        return {
            "point": None,
            "confidence": 0.0,
            "candidate_count": int(len(candidates)),
            "inlier_count": 0,
            "source": "default",
            "candidates": candidates,
        }

    weighted_center = np.sum(points * weights[:, None], axis=0) / weight_sum
    residuals = np.linalg.norm(points - weighted_center[None, :], axis=1)
    tol = max(float(w) * 0.18, float(h) * 0.16, 36.0)
    inlier_mask = residuals <= tol
    if not np.any(inlier_mask):
        inlier_mask = residuals <= float(np.max(residuals))

    inlier_points = points[inlier_mask]
    inlier_weights = weights[inlier_mask]
    inlier_weight_sum = float(np.sum(inlier_weights))
    if inlier_weight_sum <= 1e-6:
        return {
            "point": None,
            "confidence": 0.0,
            "candidate_count": int(len(candidates)),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "source": "default",
            "candidates": candidates,
        }

    vp = np.sum(inlier_points * inlier_weights[:, None], axis=0) / inlier_weight_sum
    spread = float(np.mean(np.linalg.norm(inlier_points - vp[None, :], axis=1))) if inlier_points.shape[0] > 0 else float("inf")
    spread_score = float(np.clip(1.0 - (spread / max(float(w) * 0.22, 44.0)), 0.0, 1.0))
    inlier_ratio = float(np.clip(float(np.count_nonzero(inlier_mask)) / max(1, len(candidates)), 0.0, 1.0))
    count_score = float(np.clip((len(candidates) - 1) / 5.0, 0.0, 1.0))
    quality_score = float(np.clip(np.mean([float(item["quality"]) for item in candidates]), 0.0, 1.0))
    confidence = float(np.clip((0.34 * quality_score) + (0.28 * spread_score) + (0.22 * inlier_ratio) + (0.16 * count_score), 0.0, 1.0))

    return {
        "point": [float(vp[0]), float(vp[1])],
        "confidence": confidence,
        "candidate_count": int(len(candidates)),
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "source": "lane_geometry",
        "candidates": candidates,
    }


def _summarize_global_direction(curves: List[Dict[str, object]], image_shape) -> Tuple[Optional[List[float]], float]:
    if not curves:
        return None, 0.0

    directions = []
    weights = []
    for curve in curves:
        pts = np.asarray(curve["points"], dtype=np.float32)
        mid_idx = int(pts.shape[0] // 2)
        info = _curve_direction_near_anchor(curve, pts[mid_idx], image_shape)
        direction = np.asarray(info["direction"], dtype=np.float32)
        weight = float(info["confidence"])
        if direction.shape != (2,) or weight <= 1e-4:
            continue
        directions.append(direction * weight)
        weights.append(weight)

    if not directions:
        return None, 0.0

    global_direction = _normalize(np.sum(np.stack(directions, axis=0), axis=0))
    confidence = float(np.clip(np.mean(weights), 0.0, 1.0))
    return [float(global_direction[0]), float(global_direction[1])], confidence


def estimate_lane_direction(
    lane_polygons: Sequence[Sequence[Sequence[float]]],
    image_shape,
    anchors: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, object]:
    """
    Estimate local lane directions from lane boundaries rather than raw polygon point clouds.
    """
    _, valid_polygons = _build_mask_from_polygons(lane_polygons, image_shape)
    boundary_curves = _collect_boundary_curves(valid_polygons, image_shape)
    if not boundary_curves:
        return {
            "available": False,
            "global_direction": None,
            "global_arrow": None,
            "global_confidence": 0.0,
            "vanishing_point": None,
            "vanishing_point_confidence": 0.0,
            "vanishing_point_source": "default",
            "vanishing_point_candidate_count": 0,
            "lane_arrows": [],
            "anchor_arrows": [],
            "anchor_priors": [],
            "boundary_count": 0,
        }

    h, w = image_shape[:2]
    arrow_len = float(np.clip(0.12 * max(h, w), 36.0, 96.0))
    anchor_arrows: List[Dict[str, object]] = []
    polygon_preferences = _build_polygon_side_preferences(boundary_curves, image_shape)
    lane_arrows = _build_lane_direction_entries(valid_polygons, boundary_curves, image_shape, polygon_preferences)
    lane_entries_by_polygon = {
        int(entry["polygon_index"]): entry for entry in lane_arrows
    }
    anchor_priors: List[Dict[str, object]] = []
    for anchor_index, anchor in enumerate(anchors or []):
        if anchor is None:
            continue
        a = np.asarray(anchor, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2 or not np.all(np.isfinite(a)):
            continue

        best_curve = None
        best_info = None
        best_score = None
        for curve in boundary_curves:
            polygon_preference = polygon_preferences.get(int(curve.get("polygon_index", -1)))
            score, info = _selection_score_for_anchor(curve, a, image_shape, polygon_preference=polygon_preference)
            if best_score is None or score > best_score:
                best_score = score
                best_curve = curve
                best_info = info

        if best_curve is None or best_info is None:
            continue

        direction = np.asarray(best_info["direction"], dtype=np.float32)
        start = a
        end = a + direction * arrow_len
        anchor_arrows.append(
            {
                "anchor_index": int(anchor_index),
                "anchor": [float(a[0]), float(a[1])],
                "start": [float(start[0]), float(start[1])],
                "end": [float(end[0]), float(end[1])],
                "direction": [float(direction[0]), float(direction[1])],
                "confidence": float(best_info["confidence"]),
                "fit_mode": str(best_info["fit_mode"]),
                "fit_residual": float(best_info["fit_residual"]),
                "curve_side": str(best_info["curve_side"]),
                "reference_point": [float(v) for v in best_info["reference_point"]],
                "anchor_distance": float(best_info["anchor_distance"]),
                "selection_score": float(best_score),
                "side_quality_score": float(best_info.get("side_quality_score", 0.0)),
                "preferred_side": best_info.get("preferred_side"),
                "side_preference_bonus": float(best_info.get("side_preference_bonus", 0.0)),
                "polygon_index": int(best_curve["polygon_index"]),
            }
        )
        anchor_priors.append(
            _build_anchor_prior(
                anchor_index=anchor_index,
                anchor=a,
                best_curve=best_curve,
                best_info=best_info,
                lane_entries_by_polygon=lane_entries_by_polygon,
            )
        )

    global_direction, global_confidence = _summarize_global_direction(boundary_curves, image_shape)
    vp_info = _estimate_lane_vanishing_point(boundary_curves, image_shape)
    return {
        "available": True,
        "global_direction": global_direction,
        "global_arrow": None,
        "global_confidence": global_confidence,
        "vanishing_point": vp_info["point"],
        "vanishing_point_confidence": float(vp_info["confidence"]),
        "vanishing_point_source": str(vp_info["source"]),
        "vanishing_point_candidate_count": int(vp_info["candidate_count"]),
        "lane_arrows": lane_arrows,
        "anchor_arrows": anchor_arrows,
        "anchor_priors": anchor_priors,
        "boundary_count": int(len(boundary_curves)),
        "image_size": [int(w), int(h)],
    }
