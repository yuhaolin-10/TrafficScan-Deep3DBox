import math
from pathlib import Path

try:
    from .qt import QtGui, QtWidgets
except Exception:
    try:
        from gui.qt import QtGui, QtWidgets
    except Exception:
        from qt import QtGui, QtWidgets


class ViolationsTable(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background:#0b1220;border-left:1px solid #243041;")

        header = QtWidgets.QFrame()
        header.setStyleSheet("background:#111827;border-bottom:1px solid #243041;")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(14, 14, 14, 12)
        header_layout.setSpacing(4)

        title = QtWidgets.QLabel("Details")
        title.setStyleSheet("color:#f8fafc;font-size:16px;font-weight:600;")
        subtitle = QtWidgets.QLabel("Select an image or video to inspect metadata, detections, and summary results.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#94a3b8;font-size:12px;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        self.details = QtWidgets.QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setStyleSheet(
            "QPlainTextEdit{background:#0b1220;border:0px;color:#d1d5db;font-size:12px;padding:12px;}"
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header, 0)
        root.addWidget(self.details, 1)

        self.clear_results()

    def clear_results(self):
        self.details.setPlainText(
            "No file is selected yet.\n\n"
            "Choose an image or video from the workspace to inspect it here."
        )

    def show_file_details(self, image_path: str, *, status: str = "", failure_reason: str = "", result: dict | None = None):
        self.details.setPlainText(
            self._format_file_details(
                image_path=image_path,
                status=status,
                failure_reason=failure_reason,
                result=result or {},
            )
        )

    def show_video_details(self, video_path: str, *, status: str = "", failure_reason: str = "", video_info=None, result: dict | None = None):
        self.details.setPlainText(
            self._format_video_details(
                video_path=video_path,
                status=status,
                failure_reason=failure_reason,
                video_info=video_info,
                result=result or {},
            )
        )

    def append_results_for_file(self, image_path: str, result: dict):
        self.show_file_details(image_path, status="done", result=result)

    def focus_detection(self, image_path: str, result: dict, detection: dict, *, candidate_count=1, point=None):
        self.details.setPlainText(
            self._format_detection_details(
                image_path=image_path,
                result=result,
                detection=detection,
                candidate_count=candidate_count,
                point=point,
            )
        )

    def show_lane_details(self, image_path: str, result: dict, lane_index: int, lane_polygon, *, point=None):
        violating = [det for det in result.get("detections", []) if bool(det.get("is_violating", False))]
        lines = [
            "Current object: lane region",
            f"File name: {Path(image_path).name}",
            f"File path: {self._display_path(image_path)}",
            f"Record id: {result.get('record_id', '-')}",
            f"Clicked point: {point if point is not None else '-'}",
            f"Lane index: {lane_index}",
            f"Polygon points: {len(lane_polygon) if lane_polygon else 0}",
            f"Lane area: {int(result.get('lane_area_px', 0))}",
            f"Detected vehicles: {int(result.get('vehicle_count', 0))}",
            f"Violations: {int(result.get('violation_count', 0))}",
            f"Any violation: {'yes' if bool(result.get('any_violation', False)) else 'no'}",
        ]
        if violating:
            lines.extend(["", "Top violating detections:"])
            for det in sorted(violating, key=lambda item: float(item.get("violation_ratio", 0.0)), reverse=True)[:5]:
                plate_text = str(det.get("plate_text", "") or "").strip()
                plate_suffix = f" | plate {plate_text}" if plate_text else ""
                lines.append(
                    f"- Vehicle #{det.get('index', '?')} | {self._vehicle_type_text(det.get('vehicle_type'))} | "
                    f"ratio {float(det.get('violation_ratio', 0.0)):.1%}{plate_suffix}"
                )
        else:
            lines.extend(["", "No violating vehicle is currently associated with this lane region."])
        self.details.setPlainText("\n".join(lines))

    def _format_file_details(self, image_path, status, failure_reason, result):
        path = Path(str(image_path))
        image_info = self._read_image_info(path, result)
        timestamp = result.get("timestamp") or "-"
        processed_path = result.get("processed_path") or "-"
        lines = [
            "Current object: image file",
            f"File name: {path.name}",
            f"File path: {self._display_path(path)}",
            f"Status: {status or 'unknown'}",
            f"Exists: {'yes' if image_info['exists'] else 'no'}",
            f"File size: {self._format_file_size(image_info['size_bytes'])}",
            f"Resolution: {self._format_resolution(image_info['width'], image_info['height'])}",
        ]

        if result:
            lines.extend(
                [
                    f"Record id: {result.get('record_id', '-')}",
                    f"Processed at: {timestamp}",
                    f"Lane regions: {len(result.get('lane_polygons', []))}",
                    f"Detected vehicles: {int(result.get('vehicle_count', 0))}",
                    f"Violations: {int(result.get('violation_count', 0))}",
                    f"Recognized violating plates: {int(result.get('violating_plate_count', 0) or 0)}",
                    f"Any violation: {'yes' if bool(result.get('any_violation', False)) else 'no'}",
                    f"Result image: {processed_path}",
                ]
            )
            violating_plates = list(result.get("violating_plates", []) or [])
            if violating_plates:
                lines.extend(["", "Violating plates:"])
                for item in violating_plates[:8]:
                    lines.append(self._format_plate_summary_line(item, include_track=False))
            elif int(result.get("violation_count", 0) or 0) > 0:
                missing = int(result.get("violating_plate_missing_count", 0) or 0)
                suffix = f" ({missing} vehicle(s) unread)" if missing > 0 else ""
                lines.extend(["", f"Violating plates: no stable plate read{suffix}."])
        else:
            lines.extend([
                "Result: not processed yet.",
                "Hint: run the detector, then click an object in the preview for more details.",
            ])

        if failure_reason:
            lines.extend(["", "Failure reason:", str(failure_reason)])

        return "\n".join(lines)

    def _format_video_details(self, video_path, status, failure_reason, video_info, result):
        path = Path(str(video_path))
        info = video_info or {}
        frame_count = int(self._value(info, 'frame_count', result.get('frame_count', 0)) or 0)
        fps = float(self._value(info, 'fps', result.get('fps', 0.0)) or 0.0)
        duration_s = float(self._value(info, 'duration_s', result.get('duration_s', 0.0)) or 0.0)
        width = int(self._value(info, 'width', result.get('image_width', 0)) or 0)
        height = int(self._value(info, 'height', result.get('image_height', 0)) or 0)
        preview_frame_index = int(result.get('preview_frame_index', self._value(info, 'preview_frame_index', 0)) or 0)
        codec = str(self._value(info, 'codec', result.get('codec', '')) or '-')
        preview_frame_path = result.get('preview_frame_path') or '-'

        lines = [
            "Current object: video file",
            f"File name: {path.name}",
            f"File path: {self._display_path(path)}",
            f"Status: {status or 'unknown'}",
            f"Exists: {'yes' if path.exists() else 'no'}",
            f"File size: {self._format_file_size(path.stat().st_size if path.exists() else None)}",
            f"Resolution: {self._format_resolution(width, height)}",
            f"Frames: {frame_count}",
            f"FPS: {fps:.2f}" if fps > 0 else "FPS: unknown",
            f"Duration: {self._format_duration(duration_s)}",
            f"Codec: {codec}",
            f"Representative frame index: {preview_frame_index}",
        ]

        if result:
            count_line_names = [
                str(name)
                for name in (result.get('count_line_names', []) or [])
                if str(name) != 'auto_center_line'
            ]
            count_line_text = ', '.join(str(name) for name in count_line_names) if count_line_names else '-'
            count_line_source = self._display_count_line_source(result.get('count_line_source', '-'))
            processed_frames = int(result.get('processed_frame_count', 0))
            expected_processed_frames = int(result.get('expected_processed_frame_count', 0) or 0)
            progress_percent = float(result.get('progress_percent', 0.0) or 0.0)
            throughput = float(result.get('processed_frames_per_s', 0.0) or 0.0)
            eta_seconds = float(result.get('estimated_remaining_s', 0.0) or 0.0)
            current_frame_index = int(result.get('current_frame_index', 0) or 0)
            current_vehicle_count = int(result.get('current_vehicle_count', 0) or 0)
            current_violation_count = int(result.get('current_violation_count', 0) or 0)
            progress_lines = []
            if str(result.get('status', '')) == 'running' or str(status or '').lower() == 'running':
                current_detected_plate_count = int(result.get("current_detected_plate_count", 0) or 0)
                current_region_rule_count = int(result.get("current_region_rule_violation_count", 0) or 0)
                progress_lines.extend(
                    [
                        f"Run progress: {progress_percent:.1f}%",
                        f"Processed frames: {processed_frames}/{expected_processed_frames}" if expected_processed_frames > 0 else f"Processed frames: {processed_frames}",
                        f"Current source frame index: {current_frame_index}",
                        f"Throughput: {throughput:.2f} fps" if throughput > 0 else "Throughput: warming up",
                        f"ETA: {self._format_duration(eta_seconds)}" if eta_seconds > 0 else "ETA: estimating",
                        f"Current vehicles in frame: {current_vehicle_count}",
                        f"Current violations in frame: {current_violation_count}",
                        f"Current region-rule hits in frame: {current_region_rule_count}",
                        f"Current detected plates in frame: {current_detected_plate_count}",
                    ]
                )
            lines.extend(progress_lines)
            lines.extend(
                [
                    f"Video status: {result.get('status', 'ready')}",
                    f"Preview frame path: {preview_frame_path}",
                    f"Processed video path: {result.get('processed_video_path', '-')}",
                    f"Processed frames: {processed_frames}",
                    f"Frame stride: {int(result.get('frame_stride', 1) or 1)}",
                    f"Output FPS: {float(result.get('output_fps', 0.0)):.2f}" if float(result.get('output_fps', 0.0)) > 0 else "Output FPS: unknown",
                    f"Vehicle instances: {int(result.get('total_vehicle_instances', 0))}",
                    f"Max vehicles in one frame: {int(result.get('max_vehicle_count', 0))}",
                    f"Tracks discovered: {int(result.get('track_count', 0))}",
                    f"Confirmed tracks: {int(result.get('confirmed_track_count', 0))}",
                    f"Moving tracks: {int(result.get('moving_track_count', 0))}",
                    f"Traffic count total: {int(result.get('traffic_count_total', 0))}",
                    f"Traffic forward: {int(result.get('traffic_count_forward', 0))}",
                    f"Traffic backward: {int(result.get('traffic_count_backward', 0))}",
                    f"Frames with count event: {int(result.get('frames_with_count_event', 0))}",
                    f"Count line source: {count_line_source}",
                    f"Count lines: {count_line_text}",
                    f"Scene regions: {int(result.get('scene_region_count', 0) or 0)}",
                    f"Region rule events: {int(result.get('region_rule_event_count', 0) or 0)}",
                    f"No parking events: {int(result.get('region_rule_no_parking_count', 0) or 0)}",
                    f"No non-motor events: {int(result.get('region_rule_no_non_motor_count', 0) or 0)}",
                    f"Wrong-way events: {int(result.get('region_rule_no_wrong_way_count', 0) or 0)}",
                    f"Frames with region-rule violation: {int(result.get('frames_with_region_rule_violation', 0) or 0)}",
                    f"Frames with violation: {int(result.get('frames_with_violation', 0))}",
                    f"Violation instances: {int(result.get('total_violation_instances', 0))}",
                    f"Plate OCR attempts: {int(result.get('total_plate_ocr_attempt_count', 0) or 0)}",
                    f"Plate OCR successes: {int(result.get('total_plate_ocr_success_count', 0) or 0)}",
                    f"Violating tracks: {int(result.get('violating_track_count', 0) or 0)}",
                    f"Tracks with stable plate: {int(result.get('violating_track_plate_count', 0) or 0)}",
                    f"Any violation: {'yes' if bool(result.get('any_violation', False)) else 'no'}",
                ]
            )
            region_rule_events = list(result.get("region_rule_events", []) or [])
            if region_rule_events:
                lines.extend(["", "Recent region-rule events:"])
                for item in region_rule_events[-10:]:
                    lines.append(self._format_region_rule_event_line(item))
            elif int(result.get("scene_region_count", 0) or 0) > 0:
                suffix = " yet" if str(result.get("status", "")).strip().lower() == "running" else ""
                lines.extend(["", f"Region-rule events: none triggered{suffix}."])
            violating_track_plates = list(result.get("violating_track_plates", []) or [])
            if violating_track_plates:
                lines.extend(["", "Violating plates (fused by track):"])
                for item in violating_track_plates[:10]:
                    lines.append(self._format_plate_summary_line(item, include_track=True))
            elif int(result.get("frames_with_violation", 0) or 0) > 0:
                unread = int(result.get("unread_violating_track_count", 0) or 0)
                suffix = f" ({unread} violating track(s) unread)" if unread > 0 else ""
                lines.extend(["", f"Violating plates: no stable track-level plate read{suffix}."])
        else:
            lines.extend([
                "Result: metadata only, not processed yet.",
                "Hint: run the video pipeline to get tracking, traffic count, and violation summary.",
            ])

        if failure_reason:
            lines.extend(["", "Failure reason:", str(failure_reason)])

        return "\n".join(lines)

    def _format_detection_details(self, image_path, result, detection, candidate_count, point):
        bbox = detection.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        yaw_deg = math.degrees(float(detection.get("yaw", 0.0)))
        lines = [
            "Current object: vehicle detection",
            f"File name: {Path(str(image_path)).name}",
            f"File path: {self._display_path(image_path)}",
            f"Record id: {result.get('record_id', '-')}",
            f"Vehicle index: {detection.get('index', '-')}",
            f"Vehicle type: {self._vehicle_type_text(detection.get('vehicle_type'))}",
            f"Confidence: {float(detection.get('confidence', 0.0)):.3f}",
            f"Violation: {'yes' if bool(detection.get('is_violating', False)) else 'no'}",
            f"Violation type: {self._violation_type_text(detection.get('violation_type'))}",
            f"Violation ratio: {float(detection.get('violation_ratio', 0.0)):.2%}",
            f"Plate: {str(detection.get('plate_text', '') or '-')}",
            f"Plate confidence: {float(detection.get('plate_confidence', 0.0)):.3f}",
            f"Plate type: {str(detection.get('plate_type', '') or '-')}",
            f"Plate support count: {int(detection.get('plate_support_count', 0) or 0)}",
            f"Plate status: {self._plate_status_text(detection.get('plate_status'))}",
            f"Yaw: {yaw_deg:.1f} deg",
            f"Footprint area: {float(detection.get('footprint_area_px', 0.0)):.1f} px",
            f"2D box: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]",
            f"Footprint points: {len(detection.get('footprint', []))}",
            f"3D corner points: {len(detection.get('corners_2d', []))}",
            f"Clicked point: {point if point is not None else '-'}",
            f"Overlapping candidates: {candidate_count}",
        ]

        processed_path = result.get("processed_path")
        if processed_path:
            lines.append(f"Result image: {processed_path}")

        return "\n".join(lines)

    def _read_image_info(self, path: Path, result: dict):
        exists = path.exists() and path.is_file()
        size_bytes = None
        if exists:
            try:
                size_bytes = int(path.stat().st_size)
            except OSError:
                size_bytes = None

        width = int(result.get("image_width", 0) or 0)
        height = int(result.get("image_height", 0) or 0)
        if width <= 0 or height <= 0:
            reader = QtGui.QImageReader(str(path))
            size = reader.size()
            if size.isValid():
                width = max(width, int(size.width()))
                height = max(height, int(size.height()))

        return {
            "exists": exists,
            "size_bytes": size_bytes,
            "width": width,
            "height": height,
        }

    def _format_resolution(self, width, height):
        if int(width or 0) <= 0 or int(height or 0) <= 0:
            return "unknown"
        return f"{int(width)} x {int(height)}"

    def _format_file_size(self, size_bytes):
        if size_bytes is None or size_bytes < 0:
            return "unknown"
        size = float(size_bytes)
        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        return f"{size:.1f} {units[unit_index]}"

    def _format_duration(self, duration_s):
        value = float(duration_s or 0.0)
        if value <= 0.0:
            return "unknown"
        total_seconds = int(round(value))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

    def _display_count_line_source(self, source):
        text = str(source or "-").strip()
        if text == "auto_center_line":
            return "auto"
        return text or "-"

    def _display_path(self, path):
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _vehicle_type_text(self, vehicle_type):
        key = str(vehicle_type or "vehicle").strip().lower()
        return {
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
            "person": "person",
            "vehicle": "vehicle",
        }.get(key, str(vehicle_type or "vehicle"))

    def _violation_type_text(self, violation_type):
        key = str(violation_type or "").strip().lower()
        return {
            "emergency lane occupation": "Emergency Lane Occupation",
            "emergency_lane": "Emergency Lane Occupation",
        }.get(key, str(violation_type or "none"))

    def _plate_status_text(self, status):
        key = str(status or "").strip().lower()
        return {
            "detected": "detected",
            "not_found": "not found",
            "skipped": "skipped",
            "track_fused": "track fused",
        }.get(key, key or "-")

    def _format_plate_summary_line(self, item, *, include_track: bool):
        vehicle_type = self._vehicle_type_text(item.get("vehicle_type"))
        plate_text = str(item.get("plate_text", "") or "-")
        confidence = float(item.get("plate_confidence", 0.0) or 0.0)
        support = int(item.get("plate_support_count", 0) or 0)
        plate_type = str(item.get("plate_type", "") or "-")
        ratio = float(item.get("violation_ratio", item.get("max_violation_ratio", 0.0)) or 0.0)
        prefix = f"- {str(item.get('track_id', '-'))} | " if include_track else f"- Vehicle #{int(item.get('index', 0) or 0)} | "
        extra = ""
        if include_track:
            frames = list(item.get("source_frame_indices", []) or [])
            if frames:
                preview = ", ".join(str(int(v)) for v in frames[:4])
                if len(frames) > 4:
                    preview += ", ..."
                extra = f" | frames {preview}"
        return (
            f"{prefix}{vehicle_type} | {plate_text} | conf {confidence:.2f} | "
            f"support {support} | {plate_type} | ratio {ratio:.1%}{extra}"
        )

    def _region_rule_label_text(self, event):
        rule_type = str(event.get("rule_type", "") or "").strip().lower()
        return {
            "no_parking": "No Parking",
            "no_non_motor": "No Non-Motor",
            "no_wrong_way": "Wrong Way",
        }.get(rule_type, str(event.get("rule_label", "") or rule_type or "rule"))

    def _format_region_rule_event_line(self, item):
        rule_label = self._region_rule_label_text(item)
        region_name = str(item.get("region_name", "") or item.get("region_id", "-") or "-")
        track_id = str(item.get("track_id", "") or "-")
        vehicle_type = self._vehicle_type_text(item.get("vehicle_type"))
        frame_index = item.get("frame_index")
        timestamp_s = item.get("timestamp_s")
        frame_text = "-" if frame_index is None else str(int(frame_index))
        if timestamp_s is None:
            time_text = "-"
        else:
            time_text = f"{float(timestamp_s):.1f}s"
        return (
            f"- {rule_label} | region {region_name} | track {track_id} | "
            f"{vehicle_type} | frame {frame_text} | t {time_text}"
        )

    def _value(self, obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
