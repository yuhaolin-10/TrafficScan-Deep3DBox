import cv2
import numpy as np


DEFAULT_LAYERS = {
    "lane_mask": True,
    "vehicle_mask": False,
    "footprint": True,
    "boxes_3d": True,
    "labels": False,
    "yaw_debug_overlay": False,
}


def _normalize_layers(layers):
    state = dict(DEFAULT_LAYERS)
    if layers:
        state.update(layers)
    return state


def _point2(value):
    arr = np.array(value, dtype=np.float32).reshape(-1)
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        return None
    return np.array([float(arr[0]), float(arr[1])], dtype=np.float32)


def _draw_arrow(result, start, direction, length, color, thickness=2, tip_length=0.18):
    start_pt = _point2(start)
    dir_vec = _point2(direction)
    if start_pt is None or dir_vec is None:
        return
    norm = float(np.linalg.norm(dir_vec))
    if norm <= 1e-6:
        return
    end_pt = start_pt + (dir_vec / norm) * float(length)
    cv2.arrowedLine(
        result,
        tuple(np.round(start_pt).astype(np.int32).tolist()),
        tuple(np.round(end_pt).astype(np.int32).tolist()),
        color,
        thickness,
        tipLength=tip_length,
    )


def _draw_text_block(result, origin, lines, fg_color=(230, 230, 230), bg_color=(18, 18, 18)):
    anchor = _point2(origin)
    if anchor is None or not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    padding = 6
    line_gap = 4
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    width = max(size[0] for size in line_sizes) + (padding * 2)
    height = sum(size[1] for size in line_sizes) + (line_gap * (len(lines) - 1)) + (padding * 2)

    x = int(round(float(anchor[0]) + 8.0))
    y = int(round(float(anchor[1]) - 8.0))
    x = max(4, min(x, result.shape[1] - width - 4))
    y = max(height + 4, min(y, result.shape[0] - 4))

    top_left = (x, y - height)
    bottom_right = (x + width, y)
    overlay = result.copy()
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    result[:] = cv2.addWeighted(overlay, 0.72, result, 0.28, 0)
    cv2.rectangle(result, top_left, bottom_right, (70, 70, 70), 1)

    text_y = top_left[1] + padding
    for line, size in zip(lines, line_sizes):
        text_y += size[1]
        cv2.putText(
            result,
            line,
            (top_left[0] + padding, text_y),
            font,
            font_scale,
            fg_color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_gap


def render_result(frame, lane_mask, detections, layers=None, selected_idx=None):
    """
    Render visualization overlays onto a frame.

    Args:
        frame: Original BGR image.
        lane_mask: Binary lane mask (0/255) or None.
        detections: List of detection dicts from pipeline.
        layers: Optional layer switches.
        selected_idx: Optional detection index to highlight.
    """
    state = _normalize_layers(layers)
    # Keep the preview clean: do not render vehicle segmentation fills/outlines.
    state["vehicle_mask"] = False
    result = frame.copy()

    if state["lane_mask"] and lane_mask is not None:
        colored_lane = np.zeros_like(result)
        colored_lane[lane_mask > 0] = [255, 0, 0]  # BGR blue
        result = cv2.addWeighted(result, 1.0, colored_lane, 0.3, 0)

    if state["yaw_debug_overlay"]:
        global_vp = None
        global_vp_source = "none"
        global_vp_conf = 0.0
        for det in detections:
            yaw_debug = det.get("yaw_debug", {}) or {}
            vp = _point2((det.get("yaw_debug_vectors", {}) or {}).get("vanishing_point", []))
            if vp is None:
                continue
            global_vp = vp
            global_vp_source = str(yaw_debug.get("vanishing_point_source", "none"))
            global_vp_conf = float(yaw_debug.get("vanishing_point_confidence", 0.0))
            break

        if global_vp is not None:
            marker_color = {
                "lane": (0, 200, 255),
                "mixed": (120, 220, 255),
                "ground_default": (220, 220, 220),
                "default": (200, 200, 200),
            }.get(global_vp_source, (200, 200, 200))
            vp_pt = tuple(np.round(global_vp).astype(np.int32).tolist())
            cv2.drawMarker(
                result,
                vp_pt,
                marker_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )
            if selected_idx is not None:
                _draw_text_block(
                    result,
                    np.array([global_vp[0] + 4.0, global_vp[1] + 26.0], dtype=np.float32),
                    [
                        f"VP {global_vp_source}",
                        f"conf={global_vp_conf:.2f}",
                        f"({global_vp[0]:.0f},{global_vp[1]:.0f})",
                    ],
                    fg_color=(245, 245, 245),
                    bg_color=(14, 18, 28),
                )

    for idx, det in enumerate(detections):
        is_selected = selected_idx is not None and idx == selected_idx
        is_violating = bool(det.get("is_violating", False))
        ratio = float(det.get("violation_ratio", 0.0))
        vehicle_type = det.get("vehicle_type", "vehicle")
        conf = float(det.get("confidence", 0.0))

        base_color = (0, 0, 255) if is_violating else (0, 255, 0)
        color = (255, 0, 0) if is_selected else base_color
        mask_outline_thickness = 4 if is_selected else 3
        footprint_thickness = 4 if is_selected else (5 if is_violating else 4)
        box_thickness = 3 if is_selected else 2

        if state["vehicle_mask"]:
            mask_polygon = np.array(det.get("mask_polygon", []), dtype=np.int32)
            if mask_polygon.size >= 6:
                overlay = np.zeros_like(result)
                cv2.fillPoly(overlay, [mask_polygon], color)
                result = cv2.addWeighted(result, 1.0, overlay, 0.16, 0)
                cv2.polylines(result, [mask_polygon], True, color, mask_outline_thickness, cv2.LINE_AA)

        if state["footprint"]:
            footprint = np.array(det.get("footprint", []), dtype=np.int32)
            if footprint.size >= 8:
                cv2.polylines(result, [footprint], True, color, footprint_thickness, cv2.LINE_AA)

        if state["boxes_3d"]:
            corners = np.array(det.get("corners_2d", []), dtype=np.int32)
            if corners.shape[0] >= 8:
                top_face = corners[[0, 3, 7, 4]]
                cv2.polylines(result, [top_face], True, color, box_thickness, cv2.LINE_AA)
                for i, j in [(0, 1), (3, 2), (7, 6), (4, 5)]:
                    cv2.line(result, tuple(corners[i]), tuple(corners[j]), color, box_thickness, cv2.LINE_AA)

        if is_selected:
            plate_text = str(det.get("plate_text", "") or "").strip()
            if not plate_text:
                plate_text = str(det.get("track_plate_text", "") or "").strip()
            anchor = det.get("label_anchor")
            if not anchor:
                anchor = det.get("track_anchor", [])
            if not anchor:
                bbox = np.array(det.get("bbox", []), dtype=np.float32).reshape(-1)
                if bbox.size == 4:
                    anchor = [float(bbox[0]), float(bbox[1])]
            info_lines = [
                f"{vehicle_type}  conf={conf:.2f}",
                ("Violation" if is_violating else "Normal") + f"  ratio={ratio:.2f}",
            ]
            if plate_text:
                info_lines.append(f"Plate: {plate_text}")
            _draw_text_block(
                result,
                anchor,
                info_lines,
                fg_color=(255, 255, 255),
                bg_color=(14, 18, 28),
            )

        if state["yaw_debug_overlay"]:
            vectors = det.get("yaw_debug_vectors", {}) or {}
            yaw_debug = det.get("yaw_debug", {}) or {}
            center = _point2(vectors.get("center", []))
            bbox = np.array(det.get("bbox", []), dtype=np.float32).reshape(-1)
            if center is None and bbox.size == 4:
                center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

            if center is not None:
                vp = _point2(vectors.get("vanishing_point", []))
                if vp is not None:
                    cv2.arrowedLine(
                        result,
                        tuple(np.round(center).astype(np.int32).tolist()),
                        tuple(np.round(vp).astype(np.int32).tolist()),
                        (245, 245, 245),
                        1,
                        tipLength=0.04,
                    )

                lane_prior = vectors.get("lane_prior", {}) or {}
                lane_dir = lane_prior.get("direction", [])
                lane_conf = float(lane_prior.get("confidence", 0.0))
                if lane_conf > 1e-4:
                    lane_len = 34.0 + (26.0 * np.clip(lane_conf, 0.0, 1.0))
                    _draw_arrow(result, center, lane_dir, lane_len, (255, 160, 60), thickness=2)

                yaw_value = float(det.get("yaw", 0.0))
                final_dir = _point2(vectors.get("final_direction", []))
                if final_dir is None:
                    final_dir = np.array([np.sin(yaw_value), -np.cos(yaw_value)], dtype=np.float32)
                final_len = 38.0 + (22.0 * np.clip(abs(yaw_value) / np.radians(55.0), 0.0, 1.0))
                _draw_arrow(result, center, final_dir, final_len, (70, 70, 255), thickness=2)
                cv2.circle(result, tuple(np.round(center).astype(np.int32).tolist()), 3, (240, 240, 240), -1)

                sign_label = "L" if float(yaw_debug.get("preferred_sign", 0.0)) > 0 else "R"
                vp_source = str(yaw_debug.get("vanishing_point_source", "none"))
                vp_conf = float(yaw_debug.get("vanishing_point_confidence", 0.0))
                if is_selected:
                    lines = [
                        f"yaw={np.degrees(yaw_value):.1f}deg {sign_label}",
                        f"persp={np.degrees(float(yaw_debug.get('perspective_magnitude', 0.0))):.1f}",
                        f"lane={np.degrees(float(yaw_debug.get('lane_prior_yaw_abs', 0.0))):.1f}",
                        f"w_lane={float(yaw_debug.get('lane_weight', 0.0)):.2f}",
                        f"c_lane={float(yaw_debug.get('lane_prior_confidence', 0.0)):.2f}",
                        f"vp={vp_source} {vp_conf:.2f}",
                    ]
                    _draw_text_block(result, center, lines)

    return result
