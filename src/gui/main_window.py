import hashlib
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

try:
    from .qt import QtCore, QtGui, QtWidgets
    from .viewer_panel import ViewerPanel
    from .workspace_panel import WorkspacePanel
    from .violations_table import ViolationsTable
    from .processing_worker import ProcessingWorker
    from .video_processing_worker import VideoProcessingWorker
    from ..services.renderer import render_result
    from ..services.scene_profile import PolygonRegion, RegionRuleBinding, SceneProfile, load_scene_profile, save_scene_profile
    from ..services.video_reader import VideoFrameSession, is_video_path, read_video_info
except Exception:
    try:
        from gui.qt import QtCore, QtGui, QtWidgets
        from gui.viewer_panel import ViewerPanel
        from gui.workspace_panel import WorkspacePanel
        from gui.violations_table import ViolationsTable
        from gui.processing_worker import ProcessingWorker
        from gui.video_processing_worker import VideoProcessingWorker
        from services.renderer import render_result
        from services.scene_profile import PolygonRegion, RegionRuleBinding, SceneProfile, load_scene_profile, save_scene_profile
        from services.video_reader import VideoFrameSession, is_video_path, read_video_info
    except Exception:
        from qt import QtCore, QtGui, QtWidgets
        from viewer_panel import ViewerPanel
        from workspace_panel import WorkspacePanel
        from violations_table import ViolationsTable
        from processing_worker import ProcessingWorker
        from video_processing_worker import VideoProcessingWorker
        from services.renderer import render_result
        from services.scene_profile import PolygonRegion, RegionRuleBinding, SceneProfile, load_scene_profile, save_scene_profile
        from services.video_reader import VideoFrameSession, is_video_path, read_video_info


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TrafficScan Deep3DBox UI")
        self.resize(1480, 920)

        self._pending_render = False
        self._is_running = False
        self._run_thread = None
        self._run_worker = None
        self._run_label = ""
        self._run_started_at = 0.0
        self._run_stop_requested = False
        self._pending_run_request = None
        self._active_run_paths = []
        self._results_by_path = {}
        self._hover_detection_index = None
        self._selected_detection_index = None
        self._manual_roi_cache = {}
        self._video_info_cache = {}
        self._video_progress_log_state = {}
        self._video_session = None
        self._video_playback = None
        self._video_play_timer = QtCore.QTimer(self)
        self._video_play_timer.timeout.connect(self._on_video_playback_tick)

        self._build_ui()
        self._connect_signals()
        self._log("info", "UI ready")

    def _build_ui(self):
        self.topbar = QtWidgets.QFrame()
        self.topbar.setFixedHeight(72)
        self.topbar.setStyleSheet("background:#111827;border-bottom:1px solid #243041;")

        title = QtWidgets.QLabel("TrafficScan")
        title.setStyleSheet("color:#f8fafc;font-size:18px;font-weight:600;")
        subtitle = QtWidgets.QLabel("Deep3DBox + Manual ROI + Sequence Roadmap")
        subtitle.setStyleSheet("color:#94a3b8;font-size:12px;")
        badge = QtWidgets.QLabel("3D Detector: Deep3DBox")
        badge.setStyleSheet(
            "color:#bfdbfe;background:#1d4ed8;border:1px solid #1d4ed8;border-radius:10px;padding:4px 10px;"
        )

        title_box = QtWidgets.QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(3)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.btn_run_selected = QtWidgets.QPushButton("Run Selected")
        self.btn_run_all = QtWidgets.QPushButton("Run All")
        self.btn_run_selected.setStyleSheet(
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;padding:8px 14px;border-radius:8px;}"
            "QPushButton:hover{background:#273449;}"
        )
        self.btn_run_all.setStyleSheet(
            "QPushButton{background:#2563eb;border:1px solid #2563eb;color:#ffffff;padding:8px 14px;border-radius:8px;font-weight:600;}"
            "QPushButton:hover{background:#1d4ed8;}"
        )

        left = QtWidgets.QHBoxLayout()
        left.setContentsMargins(16, 10, 16, 10)
        left.setSpacing(12)
        left.addLayout(title_box)
        left.addWidget(badge)
        left.addStretch(1)

        right = QtWidgets.QHBoxLayout()
        right.setContentsMargins(0, 0, 16, 0)
        right.setSpacing(12)
        right.addWidget(self.btn_run_selected)
        right.addWidget(self.btn_run_all)

        bar = QtWidgets.QHBoxLayout(self.topbar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addLayout(left, 1)
        bar.addLayout(right, 0)

        self.workspace = WorkspacePanel()
        self.viewer_panel = ViewerPanel()
        self.violations = ViolationsTable()

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(6)
        self.splitter.addWidget(self.workspace)
        self.splitter.addWidget(self.viewer_panel)
        self.splitter.addWidget(self.violations)
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, True)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([330, 940, 390])

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.topbar, 0)
        root.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

    def _connect_signals(self):
        self.btn_run_selected.clicked.connect(self._run_selected)
        self.btn_run_all.clicked.connect(self._run_all)

        self.workspace.file_selected.connect(self._on_workspace_selected)
        self.workspace.files_added.connect(self._on_workspace_added)
        self.workspace.unsupported_dropped.connect(self._on_unsupported_dropped)

        self.splitter.splitterMoved.connect(lambda *_: self._schedule_render_preview())
        self.viewer_panel.resized.connect(self._schedule_render_preview)
        self.viewer_panel.layer_toggled.connect(self._on_layer_toggled)
        self.viewer_panel.image_clicked.connect(self._on_viewer_image_clicked)
        self.viewer_panel.image_hovered.connect(self._on_viewer_image_hovered)
        self.viewer_panel.image_hover_left.connect(self._on_viewer_image_hover_left)
        self.viewer_panel.manual_roi_start_requested.connect(self._on_manual_roi_start_requested)
        self.viewer_panel.manual_roi_finish_requested.connect(self._on_manual_roi_finish_requested)
        self.viewer_panel.manual_roi_clear_requested.connect(self._on_manual_roi_clear_requested)
        self.viewer_panel.manual_roi_save_requested.connect(self._on_manual_roi_save_requested)
        self.viewer_panel.manual_roi_changed.connect(self._on_manual_roi_changed)
        self.viewer_panel.manual_roi_finished.connect(self._on_manual_roi_finished)
        self.viewer_panel.manual_roi_context_requested.connect(self._on_manual_roi_context_requested)
        self.viewer_panel.region_direction_finished.connect(self._on_region_direction_finished)
        self.viewer_panel.video_play_toggled.connect(self._on_video_play_toggled)
        self.viewer_panel.video_seek_requested.connect(self._on_video_seek_requested)
        self.viewer_panel.video_step_requested.connect(self._on_video_step_requested)

    def _on_workspace_added(self, added: int):
        if added:
            self._log("success", f"Added {added} media file(s)")

    def _on_unsupported_dropped(self):
        self._log("warning", "No supported images or videos found")

    def _on_workspace_selected(self, file_path: str):
        self._hover_detection_index = None
        self._selected_detection_index = None
        self._set_preview(file_path)
        self._show_details_for_path(file_path)
        self._log("info", f"Selected {Path(file_path).name}")

    def _set_preview(self, file_path: str):
        key = self._norm_path(file_path)
        result = self._results_by_path.get(key)
        if self._is_video_path(file_path):
            self._set_video_preview(file_path, result=result)
            self._apply_manual_roi_to_viewer(file_path)
            return
        self._close_video_session()
        if result and self._render_result_preview(result):
            self._apply_manual_roi_to_viewer(file_path)
            return
        self.viewer_panel.set_image_path(file_path)
        self._apply_manual_roi_to_viewer(file_path)

    def _set_video_preview(self, file_path: str, *, result: dict | None = None):
        item_path = self._norm_path(file_path)
        source_path, source_label, info = self._video_preview_source_for(file_path, result=result)
        source_key = self._norm_path(source_path)
        reuse_state = (
            self._video_playback is not None
            and self._video_playback.get("item_path") == item_path
            and self._video_playback.get("source_path") == source_key
        )
        current_frame = 0
        was_playing = False
        if reuse_state:
            current_frame = int(self._video_playback.get("current_frame_index", 0) or 0)
            was_playing = bool(self._video_playback.get("is_playing", False))
        else:
            self._close_video_session()

        self._ensure_video_session(source_path, info)
        total_frames = max(1, int(getattr(info, "frame_count", 0) or 0))
        current_frame = max(0, min(current_frame, total_frames - 1))
        self._video_playback = {
            "item_path": item_path,
            "source_path": source_key,
            "source_label": source_label,
            "info": info,
            "current_frame_index": current_frame,
            "is_playing": False,
        }
        self._set_video_playback_frame(current_frame, reset_view=not reuse_state)
        if was_playing:
            self._set_video_playing(True)

    def _video_preview_source_for(self, file_path: str, *, result: dict | None = None):
        source_path = self._norm_path(file_path)
        source_label = "Preview: original video"
        if result and result.get("media_type") == "video":
            processed_path = str(result.get("processed_video_path") or "").strip()
            status = str(result.get("status", "") or "").strip().lower()
            if processed_path and status == "processed" and Path(processed_path).exists():
                source_path = self._norm_path(processed_path)
                source_label = "Preview: processed result"
        info = self._load_video_info(source_path)
        return source_path, source_label, info

    def _ensure_video_session(self, source_path: str, info):
        normalized_source = self._norm_path(source_path)
        if self._video_session is not None:
            current_path = self._norm_path(getattr(self._video_session, "path", normalized_source))
            if current_path == normalized_source and self._video_session.is_open():
                return
            self._close_video_session()
        self._video_session = VideoFrameSession(normalized_source, info=info)

    def _close_video_session(self):
        self._video_play_timer.stop()
        if self._video_session is not None:
            try:
                self._video_session.close()
            except Exception:
                pass
        self._video_session = None
        self._video_playback = None
        self.viewer_panel.set_video_playback_state(visible=False)

    def _format_video_clock(self, seconds) -> str:
        value = float(seconds or 0.0)
        if not math.isfinite(value) or value < 0.0:
            return "--:--"
        total_seconds = int(round(value))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:d}:{secs:02d}"

    def _video_playback_interval_ms(self, info) -> int:
        fps = float(getattr(info, "fps", 0.0) or 0.0)
        if fps <= 1e-6:
            preview_fps = 6.0
        else:
            preview_fps = min(fps, 12.0)
        return max(40, int(round(1000.0 / max(1.0, preview_fps))))

    def _video_position_text(self, info, current_frame: int, total_frames: int) -> str:
        total_frames = max(1, int(total_frames))
        current_frame = max(0, min(int(current_frame), total_frames - 1))
        fps = float(getattr(info, "fps", 0.0) or 0.0)
        current_s = (float(current_frame) / fps) if fps > 1e-6 else 0.0
        duration_s = float(getattr(info, "duration_s", 0.0) or 0.0)
        return f"Frame {current_frame + 1}/{total_frames} | {self._format_video_clock(current_s)} / {self._format_video_clock(duration_s)}"

    def _update_video_playback_ui(self):
        if not self._video_playback:
            self.viewer_panel.set_video_playback_state(visible=False)
            return
        info = self._video_playback.get("info")
        total_frames = max(1, int(getattr(info, "frame_count", 0) or 0))
        current_frame = max(0, min(int(self._video_playback.get("current_frame_index", 0) or 0), total_frames - 1))
        fps = float(getattr(info, "fps", 0.0) or 0.0)
        preview_fps = min(fps, 12.0) if fps > 1e-6 else 6.0
        hint_text = f"Frame preview playback is capped at {preview_fps:.1f} fps"
        self.viewer_panel.set_video_playback_state(
            visible=True,
            is_playing=bool(self._video_playback.get("is_playing", False)),
            current_frame=current_frame,
            total_frames=total_frames,
            position_text=self._video_position_text(info, current_frame, total_frames),
            source_text=str(self._video_playback.get("source_label", "Preview: original video")),
            hint_text=hint_text,
        )

    def _set_video_playback_frame(self, frame_index: int, *, reset_view: bool = False):
        if not self._video_playback or self._video_session is None:
            return
        info = self._video_playback.get("info")
        total_frames = max(1, int(getattr(info, "frame_count", 0) or 0))
        target = max(0, min(int(frame_index), total_frames - 1))
        try:
            frame = self._video_session.read_frame(target)
        except Exception as exc:
            self._log("error", f"Video frame preview failed: {exc}")
            self._set_video_playing(False)
            return
        self._video_playback["current_frame_index"] = int(target)
        item_name = Path(self._video_playback.get("item_path", "video")).name
        source_label = str(self._video_playback.get("source_label", "Preview: original video"))
        short_source = "processed" if "processed" in source_label.lower() else "original"
        preview_label = f"{item_name} [{short_source} {target + 1}/{total_frames}]"
        pixmap = self._pixmap_from_bgr(frame)
        self.viewer_panel.set_pixmap(pixmap, label=preview_label, reset_view=reset_view)
        self._update_video_playback_ui()

    def _set_video_playing(self, playing: bool):
        if not self._video_playback:
            self._video_play_timer.stop()
            self.viewer_panel.set_video_playback_state(visible=False)
            return
        total_frames = max(1, int(getattr(self._video_playback.get("info"), "frame_count", 0) or 0))
        should_play = bool(playing) and total_frames > 1
        if should_play and int(self._video_playback.get("current_frame_index", 0) or 0) >= (total_frames - 1):
            self._set_video_playback_frame(0, reset_view=False)
        self._video_playback["is_playing"] = should_play
        if should_play:
            interval_ms = self._video_playback_interval_ms(self._video_playback.get("info"))
            self._video_play_timer.start(interval_ms)
        else:
            self._video_play_timer.stop()
        self._update_video_playback_ui()

    def _on_video_playback_tick(self):
        if not self._video_playback:
            self._video_play_timer.stop()
            return
        info = self._video_playback.get("info")
        total_frames = max(1, int(getattr(info, "frame_count", 0) or 0))
        current_frame = int(self._video_playback.get("current_frame_index", 0) or 0)
        if current_frame >= total_frames - 1:
            self._set_video_playing(False)
            return
        self._set_video_playback_frame(current_frame + 1, reset_view=False)

    def _on_video_play_toggled(self, playing: bool):
        self._set_video_playing(bool(playing))

    def _on_video_seek_requested(self, frame_index: int):
        if not self._video_playback:
            return
        self._set_video_playing(False)
        self._set_video_playback_frame(frame_index, reset_view=False)

    def _on_video_step_requested(self, delta: int):
        if not self._video_playback:
            return
        self._set_video_playing(False)
        current_frame = int(self._video_playback.get("current_frame_index", 0) or 0)
        self._set_video_playback_frame(current_frame + int(delta), reset_view=False)

    def _apply_manual_roi_to_viewer(self, file_path: str):
        entries = self._load_scene_regions(file_path)
        self.viewer_panel.set_manual_roi([entry["points"] for entry in entries])
        self.viewer_panel.set_region_direction_lines([entry.get("direction_line", []) for entry in entries])

    def _default_region_rule_params(self, rule_type: str) -> dict:
        key = str(rule_type or "").strip().lower()
        if key == "no_parking":
            return {
                "target_classes": ["car", "truck", "bus", "motorcycle"],
                "min_stop_seconds": 5.0,
                "max_speed_px_per_s": 24.0,
                "min_confirmed_hits": 2,
            }
        if key == "no_non_motor":
            return {
                "target_classes": ["bicycle", "motorcycle", "person"],
                "min_consecutive_frames": 2,
                "min_confirmed_hits": 1,
            }
        if key == "no_wrong_way":
            return {
                "target_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
                "min_consecutive_frames": 2,
                "min_direction_distance_px": 24.0,
                "wrong_way_dot_threshold": -0.10,
                "min_roi_overlap_ratio": 0.20,
                "min_confirmed_hits": 2,
            }
        return {}

    def _rule_display_name(self, rule_type: str) -> str:
        mapping = {
            "no_parking": "禁止停车",
            "no_non_motor": "禁止非机动车",
            "no_wrong_way": "禁止逆行",
        }
        return mapping.get(str(rule_type or "").strip().lower(), str(rule_type or "rule"))

    def _normalize_rule_binding_entry(self, binding) -> dict | None:
        if binding is None:
            return None
        if isinstance(binding, RegionRuleBinding):
            payload = {
                "rule_type": str(binding.rule_type or "").strip().lower(),
                "enabled": bool(binding.enabled),
                "params": dict(binding.params or {}),
            }
        else:
            payload = {
                "rule_type": str(binding.get("rule_type", "") or "").strip().lower(),
                "enabled": bool(binding.get("enabled", True)),
                "params": dict(binding.get("params", {}) or {}),
            }
        if not payload["rule_type"]:
            return None
        defaults = self._default_region_rule_params(payload["rule_type"])
        merged_params = dict(defaults)
        merged_params.update(payload["params"])
        if payload["rule_type"] == "no_wrong_way":
            merged_params["min_consecutive_frames"] = min(
                max(1, int(merged_params.get("min_consecutive_frames", 2) or 2)),
                2,
            )
            merged_params["min_direction_distance_px"] = min(
                max(8.0, float(merged_params.get("min_direction_distance_px", 24.0) or 24.0)),
                24.0,
            )
            merged_params["wrong_way_dot_threshold"] = max(
                float(merged_params.get("wrong_way_dot_threshold", -0.10) or -0.10),
                -0.10,
            )
            merged_params["min_roi_overlap_ratio"] = min(
                1.0,
                max(0.0, float(merged_params.get("min_roi_overlap_ratio", 0.20) or 0.20)),
            )
        if payload["rule_type"] == "no_parking":
            merged_params["min_stop_seconds"] = 5.0
            merged_params["max_speed_px_per_s"] = max(
                24.0,
                float(merged_params.get("max_speed_px_per_s", 24.0) or 24.0),
            )
        if payload["rule_type"] == "no_non_motor":
            target_classes = [
                str(item).strip().lower()
                for item in list(merged_params.get("target_classes", []) or [])
                if str(item).strip()
            ]
            for extra_class in ("bicycle", "motorcycle", "person"):
                if extra_class not in target_classes:
                    target_classes.append(extra_class)
            merged_params["target_classes"] = target_classes
        payload["params"] = merged_params
        return payload

    def _normalize_direction_line(self, points) -> list:
        if not points or len(points) != 2:
            return []
        try:
            start = [float(points[0][0]), float(points[0][1])]
            end = [float(points[1][0]), float(points[1][1])]
        except Exception:
            return []
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) <= 1.0:
            return []
        return [start, end]

    def _create_region_entry(self, points, *, index: int = 0, existing=None) -> dict:
        if isinstance(existing, dict):
            region_id = str(existing.get("region_id", "") or "").strip() or uuid.uuid4().hex[:12]
            name = str(existing.get("name", "") or "").strip() or f"Region {index + 1}"
            region_type = str(existing.get("region_type", "parking_roi") or "parking_roi")
            source = str(existing.get("source", "manual") or "manual")
            enabled = bool(existing.get("enabled", True))
            direction_line = self._normalize_direction_line(existing.get("direction_line", []))
            bindings = [
                normalized
                for normalized in (
                    self._normalize_rule_binding_entry(binding)
                    for binding in existing.get("rule_bindings", [])
                )
                if normalized is not None
            ]
        else:
            region_id = uuid.uuid4().hex[:12]
            name = f"Region {index + 1}"
            region_type = "parking_roi"
            source = "manual"
            enabled = True
            direction_line = []
            bindings = []
        normalized_points = [
            [float(point[0]), float(point[1])]
            for point in (points or [])
            if point is not None and len(point) == 2
        ]
        return {
            "region_id": region_id,
            "name": name,
            "region_type": region_type,
            "points": normalized_points,
            "source": source,
            "enabled": enabled,
            "direction_line": direction_line,
            "rule_bindings": bindings,
        }

    def _normalize_scene_regions(self, entries) -> list:
        normalized_entries = []
        for index, entry in enumerate(entries or []):
            if isinstance(entry, PolygonRegion):
                payload = {
                    "region_id": str(getattr(entry, "region_id", "") or ""),
                    "name": str(getattr(entry, "name", "") or ""),
                    "region_type": str(getattr(entry, "region_type", "parking_roi") or "parking_roi"),
                    "points": [list(point) for point in getattr(entry, "points", [])],
                    "source": str(getattr(entry, "source", "manual") or "manual"),
                    "enabled": bool(getattr(entry, "enabled", True)),
                    "direction_line": [list(point) for point in getattr(entry, "direction_line", [])],
                    "rule_bindings": list(getattr(entry, "rule_bindings", []) or []),
                }
            else:
                payload = dict(entry or {})
            normalized = self._create_region_entry(payload.get("points", []), index=index, existing=payload)
            if len(normalized["points"]) >= 3:
                normalized_entries.append(normalized)
        return normalized_entries

    def _scene_regions_to_geometry(self, entries) -> list:
        return [
            [list(point) for point in entry.get("points", [])]
            for entry in self._normalize_scene_regions(entries)
        ]

    def _scene_regions_to_direction_lines(self, entries) -> list:
        return [
            [list(point) for point in entry.get("direction_line", [])]
            for entry in self._normalize_scene_regions(entries)
        ]

    def _load_scene_regions(self, file_path: str):
        scene_key = self._scene_key_for_path(file_path)
        if scene_key in self._manual_roi_cache:
            return self._normalize_scene_regions(self._manual_roi_cache[scene_key])

        profile_path = self._scene_profile_path_for_path(file_path)
        entries = []
        if profile_path.exists():
            try:
                profile = load_scene_profile(profile_path)
                entries = [
                    region
                    for region in profile.parking_regions
                    if region.enabled and str(region.region_type or "") == "parking_roi"
                ]
            except Exception as exc:
                self._log("warning", f"Failed to load scene profile {profile_path.name}: {exc}")
                entries = []
        normalized_entries = self._normalize_scene_regions(entries)
        self._manual_roi_cache[scene_key] = normalized_entries
        return self._normalize_scene_regions(normalized_entries)

    def _merge_scene_regions_from_geometry(self, file_path: str, regions, direction_lines=None):
        existing_entries = self._load_scene_regions(file_path)
        direction_lines = list(direction_lines or [])
        merged = []
        for index, region_points in enumerate(regions or []):
            if len(region_points or []) < 3:
                continue
            existing = existing_entries[index] if index < len(existing_entries) else None
            payload = self._create_region_entry(region_points, index=index, existing=existing)
            if index < len(direction_lines):
                payload["direction_line"] = self._normalize_direction_line(direction_lines[index])
            elif existing is not None:
                payload["direction_line"] = self._normalize_direction_line(existing.get("direction_line", []))
            merged.append(payload)
        return merged

    def _set_cached_scene_regions(self, file_path: str, entries):
        scene_key = self._scene_key_for_path(file_path)
        self._manual_roi_cache[scene_key] = self._normalize_scene_regions(entries)

    def _save_scene_regions(self, file_path: str, entries):
        profile_path = self._scene_profile_path_for_path(file_path)
        scene_key = self._scene_key_for_path(file_path)
        folder_name = Path(scene_key).name or "scene"
        normalized_entries = self._normalize_scene_regions(entries)
        self._manual_roi_cache[scene_key] = normalized_entries

        profile = None
        if profile_path.exists():
            try:
                profile = load_scene_profile(profile_path)
            except Exception as exc:
                self._log("warning", f"Failed to update existing scene profile {profile_path.name}: {exc}")

        if profile is None:
            profile = SceneProfile(
                camera_id=folder_name,
                fps=0.0,
                source_path=scene_key,
                notes="Manual ROI saved from GUI",
            )

        remaining_regions = [
            region
            for region in profile.parking_regions
            if region.region_type != "parking_roi"
        ]

        for index, entry in enumerate(normalized_entries, start=1):
            region_name = str(entry.get("name", "") or "").strip() or f"Region {index}"
            rule_bindings = [
                RegionRuleBinding(
                    rule_type=str(binding.get("rule_type", "") or ""),
                    enabled=bool(binding.get("enabled", True)),
                    params=dict(binding.get("params", {}) or {}),
                )
                for binding in entry.get("rule_bindings", [])
                if str(binding.get("rule_type", "") or "").strip()
            ]
            remaining_regions.append(
                PolygonRegion(
                    name=region_name,
                    region_type=str(entry.get("region_type", "parking_roi") or "parking_roi"),
                    points=[list(point) for point in entry.get("points", [])],
                    region_id=str(entry.get("region_id", "") or "").strip() or uuid.uuid4().hex[:12],
                    source=str(entry.get("source", "manual") or "manual"),
                    enabled=bool(entry.get("enabled", True)),
                    direction_line=[list(point) for point in entry.get("direction_line", [])],
                    rule_bindings=rule_bindings,
                )
            )

        profile.camera_id = profile.camera_id or folder_name
        profile.source_path = profile.source_path or scene_key
        if not profile.notes:
            profile.notes = "Manual ROI saved from GUI"
        profile.parking_regions = remaining_regions

        if profile.parking_regions or profile.count_lines:
            save_scene_profile(profile_path, profile)
            return

        if profile_path.exists():
            profile_path.unlink()

    def _region_has_bound_rules(self, entry) -> bool:
        return bool(
            [
                binding
                for binding in dict(entry or {}).get("rule_bindings", [])
                if bool(binding.get("enabled", True)) and str(binding.get("rule_type", "") or "").strip()
            ]
        )

    def _load_video_info(self, file_path: str):
        key = self._norm_path(file_path)
        cached = self._video_info_cache.get(key)
        if cached is not None:
            return cached
        info = read_video_info(file_path)
        self._video_info_cache[key] = info
        return info

    def _is_video_path(self, file_path: str) -> bool:
        return bool(is_video_path(file_path))

    def _on_layer_toggled(self, key: str, enabled: bool):
        current = self.workspace.selected_path()
        if current:
            self._set_preview(current)
        state_text = "shown" if enabled else "hidden"
        self._log("info", f"Layer changed: {self._layer_name(key)} -> {state_text}")

    def _render_result_preview(self, result: dict) -> bool:
        original_path = result.get("original_path")
        if not original_path:
            return False

        frame = cv2.imread(str(original_path))
        if frame is None:
            return False

        lane_mask = self._build_lane_mask_from_result(result, frame.shape[:2])
        rendered = render_result(
            frame,
            lane_mask,
            result.get("detections", []),
            layers=self.viewer_panel.layers_state(),
            selected_idx=self._active_detection_index(),
        )
        pixmap = self._pixmap_from_bgr(rendered)
        if pixmap is None or pixmap.isNull():
            return False

        self.viewer_panel.set_pixmap(pixmap, label=Path(original_path).name)
        return True

    def _build_lane_mask_from_result(self, result: dict, image_hw):
        h, w = image_hw
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        for polygon in result.get("lane_polygons", []):
            pts = np.asarray(polygon, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                continue
            cv2.fillPoly(lane_mask, [pts], 255)
        return lane_mask

    def _pixmap_from_bgr(self, image):
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = int(rgb.strides[0])
        qimage = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(qimage)

    def _on_manual_roi_start_requested(self):
        current = self.workspace.selected_path()
        if not current:
            self._log("warning", "Select an image or video before drawing a manual ROI")
            return
        if self._video_playback and bool(self._video_playback.get("is_playing", False)):
            self._set_video_playing(False)
        stale_count = self._invalidate_results_for_scene(current)
        self._selected_detection_index = None
        self._hover_detection_index = None
        if stale_count > 0:
            self._set_preview(current)
            self._show_details_for_path(current)
            self._log("info", f"Cleared {stale_count} stale result(s) for this scene because the manual ROI is being edited")
        self.viewer_panel.start_manual_roi_drawing(clear_existing=False)
        self._log("info", "Manual ROI drawing started: left click to add points, right click or double click to finish. Existing regions stay in place.")

    def _on_manual_roi_finish_requested(self):
        if not self.viewer_panel.is_manual_roi_drawing():
            self._log("warning", "Manual ROI drawing is not active")
            return
        region = self.viewer_panel.finish_manual_roi_drawing()
        if not region or len(region) < 3:
            self._log("warning", "Manual ROI needs at least 3 points")
            return
        total_regions = len(self.viewer_panel.manual_roi_regions())
        self._log("success", f"Manual ROI region added with {len(region)} points, total regions={total_regions}")

    def _on_manual_roi_changed(self, regions):
        current = self.workspace.selected_path()
        if not current:
            return
        merged_entries = self._merge_scene_regions_from_geometry(
            current,
            regions,
            direction_lines=self.viewer_panel.region_direction_lines(),
        )
        self._set_cached_scene_regions(current, merged_entries)
        if self._is_running:
            return
        self._invalidate_results_for_scene(current)
        self._selected_detection_index = None
        self._hover_detection_index = None
        self._set_preview(current)
        self._show_details_for_path(current)

    def _on_manual_roi_finished(self, regions):
        current = self.workspace.selected_path()
        if not current:
            return
        merged_entries = self._merge_scene_regions_from_geometry(
            current,
            regions,
            direction_lines=self.viewer_panel.region_direction_lines(),
        )
        self._set_cached_scene_regions(current, merged_entries)

    def _on_manual_roi_clear_requested(self):
        current = self.workspace.selected_path()
        if not current:
            self._log("warning", "Select an image or video before clearing the manual ROI")
            return
        self.viewer_panel.clear_manual_roi()
        self._set_cached_scene_regions(current, [])
        self._log("info", "All manual ROI regions were cleared from the current scene")

    def _on_manual_roi_save_requested(self):
        current = self.workspace.selected_path()
        if not current:
            self._log("warning", "Select an image or video before saving the manual ROI")
            return
        entries = self._merge_scene_regions_from_geometry(
            current,
            self.viewer_panel.manual_roi_regions(),
            direction_lines=self.viewer_panel.region_direction_lines(),
        )
        self._save_scene_regions(current, entries)
        if entries:
            self._log("success", f"Saved {len(entries)} manual ROI region(s) for scene {Path(current).parent.name}")
        else:
            self._log("success", f"Removed saved manual ROI for scene {Path(current).parent.name}")

    def _region_rule_enabled(self, entry: dict, rule_type: str) -> bool:
        target = str(rule_type or "").strip().lower()
        for binding in dict(entry or {}).get("rule_bindings", []):
            if str(binding.get("rule_type", "") or "").strip().lower() == target:
                return bool(binding.get("enabled", True))
        return False

    def _set_region_rule_enabled(self, file_path: str, region_index: int, rule_type: str, enabled: bool):
        entries = self._load_scene_regions(file_path)
        if not (0 <= int(region_index) < len(entries)):
            return None
        target = str(rule_type or "").strip().lower()
        entry = dict(entries[region_index])
        bindings = [
            self._normalize_rule_binding_entry(binding)
            for binding in entry.get("rule_bindings", [])
        ]
        bindings = [binding for binding in bindings if binding is not None]
        existing = None
        for binding in bindings:
            if str(binding.get("rule_type", "") or "").strip().lower() == target:
                existing = binding
                break
        if enabled:
            if existing is None:
                bindings.append(
                    {
                        "rule_type": target,
                        "enabled": True,
                        "params": self._default_region_rule_params(target),
                    }
                )
            else:
                existing["enabled"] = True
                defaults = self._default_region_rule_params(target)
                merged = dict(defaults)
                merged.update(dict(existing.get("params", {}) or {}))
                existing["params"] = merged
        else:
            bindings = [
                binding
                for binding in bindings
                if str(binding.get("rule_type", "") or "").strip().lower() != target
            ]
            if target == "no_wrong_way":
                entry["direction_line"] = []
        entry["rule_bindings"] = bindings
        entries[region_index] = self._create_region_entry(entry.get("points", []), index=region_index, existing=entry)
        self._set_cached_scene_regions(file_path, entries)
        self._save_scene_regions(file_path, entries)
        self._apply_manual_roi_to_viewer(file_path)
        return entries[region_index]

    def _on_manual_roi_context_requested(self, region_index: int, global_pos, _scene_pos):
        current = self.workspace.selected_path()
        if not current:
            return
        entries = self._load_scene_regions(current)
        if not (0 <= int(region_index) < len(entries)):
            return
        entry = entries[int(region_index)]

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#0f172a;border:1px solid #334155;min-width:220px;}"
            "QMenu::item{padding:8px 18px;color:#dbe4ee;}"
            "QMenu::item:selected{background:#1e293b;}"
        )
        title_action = menu.addAction(str(entry.get("name", f"Region {int(region_index) + 1}")))
        title_action.setEnabled(False)
        menu.addSeparator()

        act_no_parking = menu.addAction("禁止停车")
        act_no_parking.setCheckable(True)
        act_no_parking.setChecked(self._region_rule_enabled(entry, "no_parking"))

        act_no_non_motor = menu.addAction("禁止非机动车")
        act_no_non_motor.setCheckable(True)
        act_no_non_motor.setChecked(self._region_rule_enabled(entry, "no_non_motor"))

        act_no_wrong_way = menu.addAction("禁止逆行")
        act_no_wrong_way.setCheckable(True)
        act_no_wrong_way.setChecked(self._region_rule_enabled(entry, "no_wrong_way"))

        menu.addSeparator()
        act_set_direction = menu.addAction("设置逆行允许方向")
        act_set_direction.setEnabled(self._region_rule_enabled(entry, "no_wrong_way"))
        has_direction_line = len(entry.get("direction_line", [])) == 2
        act_clear_direction = menu.addAction("清除逆行方向")
        act_clear_direction.setEnabled(self._region_rule_enabled(entry, "no_wrong_way") and has_direction_line)

        menu.addSeparator()
        act_delete = menu.addAction("删除选中区域")

        exec_menu = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
        if isinstance(global_pos, (list, tuple)) and len(global_pos) == 2:
            anchor = QtCore.QPoint(int(global_pos[0]), int(global_pos[1]))
        else:
            anchor = QtGui.QCursor.pos()
        chosen = exec_menu(anchor)
        if chosen is None:
            return

        if chosen == act_no_parking:
            enabled = bool(act_no_parking.isChecked())
            self._set_region_rule_enabled(current, region_index, "no_parking", enabled)
            state_text = "enabled" if enabled else "disabled"
            self._log("success", f"Region {region_index + 1}: 禁止停车 {state_text}")
            return

        if chosen == act_no_non_motor:
            enabled = bool(act_no_non_motor.isChecked())
            self._set_region_rule_enabled(current, region_index, "no_non_motor", enabled)
            state_text = "enabled" if enabled else "disabled"
            self._log("success", f"Region {region_index + 1}: 禁止非机动车 {state_text}")
            return

        if chosen == act_no_wrong_way:
            enabled = bool(act_no_wrong_way.isChecked())
            updated = self._set_region_rule_enabled(current, region_index, "no_wrong_way", enabled)
            state_text = "enabled" if enabled else "disabled"
            self._log("success", f"Region {region_index + 1}: 禁止逆行 {state_text}")
            if enabled and updated is not None and len(updated.get("direction_line", [])) != 2:
                self.viewer_panel.start_region_direction_drawing(int(region_index))
                self._log("info", "Draw the allowed travel direction with two clicks inside the selected region")
            return

        if chosen == act_set_direction:
            self.viewer_panel.start_region_direction_drawing(int(region_index))
            self._log("info", "Draw the allowed travel direction with two clicks inside the selected region")
            return

        if chosen == act_clear_direction:
            entries = self._load_scene_regions(current)
            if 0 <= int(region_index) < len(entries):
                entries[int(region_index)]["direction_line"] = []
                self._set_cached_scene_regions(current, entries)
                self._save_scene_regions(current, entries)
                self._apply_manual_roi_to_viewer(current)
                self._log("info", f"Region {region_index + 1}: cleared wrong-way direction")
            return

        if chosen == act_delete:
            deleted = self.viewer_panel.delete_selected_manual_roi()
            if deleted is not None:
                self._save_scene_regions(
                    current,
                    self._merge_scene_regions_from_geometry(
                        current,
                        self.viewer_panel.manual_roi_regions(),
                        direction_lines=self.viewer_panel.region_direction_lines(),
                    ),
                )
                self._log("info", f"Deleted region {region_index + 1}")
            return

    def _on_region_direction_finished(self, region_index: int, direction_line):
        current = self.workspace.selected_path()
        if not current:
            return
        entries = self._load_scene_regions(current)
        if not (0 <= int(region_index) < len(entries)):
            return
        entries[int(region_index)]["direction_line"] = self._normalize_direction_line(direction_line)
        self._set_cached_scene_regions(current, entries)
        self._save_scene_regions(current, entries)
        self._apply_manual_roi_to_viewer(current)
        if len(entries[int(region_index)]["direction_line"]) == 2:
            self._log("success", f"Region {region_index + 1}: saved allowed direction for no-wrong-way rule")

    def _scene_key_for_path(self, file_path: str) -> str:
        try:
            return str(Path(file_path).resolve().parent)
        except Exception:
            return str(Path(file_path).parent)

    def _scene_profile_path_for_path(self, file_path: str) -> Path:
        runtime = self._runtime_paths()
        scene_key = self._scene_key_for_path(file_path)
        digest = hashlib.sha1(scene_key.encode("utf-8", errors="ignore")).hexdigest()[:10]
        folder_name = Path(scene_key).name or "scene"
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in folder_name)
        runtime["scene_profiles_dir"].mkdir(parents=True, exist_ok=True)
        return runtime["scene_profiles_dir"] / f"{safe_name}_{digest}.json"

    def _manual_lane_mapping_for_paths(self, paths):
        mapping = {}
        for path in paths:
            entries = self._load_scene_regions(path)
            polygons = [
                [list(point) for point in entry.get("points", [])]
                for entry in entries
                if not self._region_has_bound_rules(entry)
            ]
            if polygons:
                mapping[self._norm_path(path)] = polygons
        return mapping

    def _scene_region_mapping_for_paths(self, paths):
        mapping = {}
        for path in paths:
            entries = self._load_scene_regions(path)
            if entries:
                mapping[self._norm_path(path)] = entries
        return mapping

    def _load_count_lines(self, file_path: str):
        profile_path = self._scene_profile_path_for_path(file_path)
        if not profile_path.exists():
            return []
        try:
            profile = load_scene_profile(profile_path)
        except Exception as exc:
            self._log("warning", f"Failed to load count lines from {profile_path.name}: {exc}")
            return []

        lines = []
        for rule in profile.count_lines:
            if not bool(getattr(rule, "enabled", True)):
                continue
            lines.append(
                {
                    "name": str(getattr(rule, "name", "count_line") or "count_line"),
                    "start": [float(rule.start[0]), float(rule.start[1])],
                    "end": [float(rule.end[0]), float(rule.end[1])],
                    "direction_mode": str(getattr(rule, "direction_mode", "any") or "any"),
                    "enabled": True,
                }
            )
        return lines

    def _count_line_mapping_for_paths(self, paths):
        mapping = {}
        for path in paths:
            lines = self._load_count_lines(path)
            if lines:
                mapping[self._norm_path(path)] = lines
        return mapping

    def _on_viewer_image_clicked(self, x: int, y: int):
        if self.viewer_panel.is_manual_roi_drawing():
            return

        current = self.workspace.selected_path()
        if not current:
            return
        if self._is_video_path(current):
            self._log("info", "Video preview click inspection is not available yet. Use the summary panel on the right.")
            return

        key = self._norm_path(current)
        result = self._results_by_path.get(key)
        if not result:
            self._selected_detection_index = None
            self._show_details_for_path(current)
            if self._is_video_path(current):
                self._log("info", "Video preview is ready. Frame-by-frame detections will arrive in the next video stage.")
            else:
                self._log("warning", "Selected image has no processed result yet")
            return

        point = (float(x), float(y))
        hit = self._pick_detection_hit(result.get("detections", []), point)
        if hit is not None:
            det = hit["detection"]
            self._selected_detection_index = int(hit["list_index"])
            self._set_preview(current)
            self.violations.focus_detection(
                image_path=current,
                result=result,
                detection=det,
                candidate_count=hit["candidate_count"],
                point=(int(x), int(y)),
            )
            self._log("info", f"Selected vehicle #{det.get('index', '?')} at ({x}, {y})")
            return

        lane_hit = self._pick_lane_hit(result.get("lane_polygons", []), point)
        if lane_hit is not None:
            self._selected_detection_index = None
            self._set_preview(current)
            self.violations.show_lane_details(
                image_path=current,
                result=result,
                lane_index=lane_hit["index"],
                lane_polygon=lane_hit["polygon"],
                point=(int(x), int(y)),
            )
            self._log("info", f"Selected lane region #{lane_hit['index']} at ({x}, {y})")
            return

        self._selected_detection_index = None
        self._set_preview(current)
        self._show_details_for_path(current)
        self._log("info", f"No object hit at ({x}, {y})")

    def _on_viewer_image_hovered(self, x: int, y: int):
        if self.viewer_panel.is_manual_roi_drawing():
            return
        current = self.workspace.selected_path()
        if not current or self._is_video_path(current):
            return

        key = self._norm_path(current)
        result = self._results_by_path.get(key)
        if not result:
            return

        hit = self._pick_detection_hit(result.get("detections", []), (float(x), float(y)))
        new_index = None if hit is None else int(hit["list_index"])
        if new_index == self._hover_detection_index:
            return
        self._hover_detection_index = new_index
        self._set_preview(current)

    def _on_viewer_image_hover_left(self):
        if self.viewer_panel.is_manual_roi_drawing():
            return
        if self._hover_detection_index is None:
            return
        self._hover_detection_index = None
        current = self.workspace.selected_path()
        if current:
            self._set_preview(current)

    def _render_preview(self):
        self.viewer_panel.refresh()

    def _schedule_render_preview(self):
        if self._pending_render:
            return
        self._pending_render = True

        def _do():
            self._pending_render = False
            self._render_preview()

        QtCore.QTimer.singleShot(0, _do)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_render_preview()

    def _queue_or_start_run(self, paths: list, *, label: str):
        request = {
            "paths": list(paths),
            "label": str(label or "Run"),
        }
        if self._is_running:
            self._pending_run_request = request
            self._request_stop_current_run()
            return
        self._start_run(request["paths"], label=request["label"])

    def _request_stop_current_run(self):
        if not self._is_running or self._run_worker is None:
            return
        if self._run_stop_requested:
            self._log("warning", "Current run is already stopping; the latest run request is queued")
            return
        self._run_stop_requested = True
        try:
            request_stop = getattr(self._run_worker, "request_stop", None)
            if callable(request_stop):
                request_stop()
        except Exception:
            pass
        self._log("warning", "Another run was requested, stopping the current run and switching when it exits")

    def _mark_interrupted_run_paths(self):
        for path in list(self._active_run_paths):
            if self.workspace.status_of_path(path) != self.workspace.STATUS_RUNNING:
                continue
            self.workspace.set_status_for_path(path, self.workspace.STATUS_CANCELLED)
            self.workspace.set_failure_reason_for_path(path, "Interrupted by a newer run request")

    def _invalidate_results_for_scene(self, file_path: str) -> int:
        scene_key = self._scene_key_for_path(file_path)
        removed = 0
        for result_path in list(self._results_by_path.keys()):
            if self._scene_key_for_path(result_path) != scene_key:
                continue
            self._results_by_path.pop(result_path, None)
            if self.workspace.status_of_path(result_path) == self.workspace.STATUS_DONE:
                self.workspace.set_status_for_path(result_path, self.workspace.STATUS_PENDING)
            removed += 1
        return int(removed)

    def _run_selected(self):
        if self.workspace.list.count() == 0:
            self._log("warning", "Workspace is empty")
            return
        targets = self.workspace.selected_paths()
        if not targets:
            selected = self.workspace.selected_path()
            if selected:
                targets = [selected]
        if not targets:
            self._log("warning", "Select at least one image to run")
            return
        self._queue_or_start_run(targets, label="Run selected")

    def _run_all(self):
        if self.workspace.list.count() == 0:
            self._log("warning", "Workspace is empty")
            return
        targets = self.workspace.all_paths()
        if not targets:
            self._log("warning", "Workspace is empty")
            return
        self._queue_or_start_run(targets, label="Run all")

    def _start_run(self, paths: list, *, label: str):
        if self._is_running:
            self._pending_run_request = {
                "paths": list(paths),
                "label": str(label or "Run"),
            }
            self._request_stop_current_run()
            return

        media_types = {"video" if self._is_video_path(path) else "image" for path in paths}
        if len(media_types) > 1:
            self._log("warning", "Please run image files and video files separately")
            return

        if media_types == {"video"}:
            self._start_video_run(paths, label=label)
            return

        runtime = self._runtime_paths()
        if not runtime["vehicle_model_path"].exists():
            self._log("error", f"Vehicle model not found: {runtime['vehicle_model_path']}")
            return

        self._is_running = True
        self._run_label = label
        self._run_started_at = time.monotonic()
        self._run_stop_requested = False
        self._active_run_paths = list(paths)
        self.violations.clear_results()
        self.workspace.set_status_for_paths(paths, self.workspace.STATUS_RUNNING)
        for path in paths:
            self.workspace.set_failure_reason_for_path(path, "")
        manual_mapping = self._manual_lane_mapping_for_paths(paths)
        self._log("info", f"{label}: {len(paths)} image(s), manual ROI applied to {len(manual_mapping)} image(s)")

        self._run_thread = QtCore.QThread(self)
        self._run_worker = ProcessingWorker(
            image_paths=paths,
            lane_model_path=runtime["lane_model_path"],
            vehicle_model_path=runtime["vehicle_model_path"],
            output_dir=runtime["output_dir"],
            db_path=runtime["db_path"],
            threshold=0.3,
            location="GUI-Camera-01",
            layers=self.viewer_panel.layers_state(),
            manual_lane_polygons_by_path=manual_mapping,
        )
        self._run_worker.moveToThread(self._run_thread)

        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._log)
        self._run_worker.task_started.connect(self._on_task_started)
        self._run_worker.task_finished.connect(self._on_task_finished)
        self._run_worker.task_failed.connect(self._on_task_failed)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.finished.connect(self._run_worker.deleteLater)
        self._run_thread.finished.connect(self._on_run_thread_finished)
        self._run_thread.finished.connect(self._run_thread.deleteLater)

        self._run_thread.start()

    def _start_video_run(self, paths: list, *, label: str):
        runtime = self._runtime_paths()
        if not runtime["vehicle_model_path"].exists():
            self._log("error", f"Vehicle model not found: {runtime['vehicle_model_path']}")
            return

        self._is_running = True
        self._run_label = label
        self._run_started_at = time.monotonic()
        self._run_stop_requested = False
        self._active_run_paths = list(paths)
        self.violations.clear_results()
        self.workspace.set_status_for_paths(paths, self.workspace.STATUS_RUNNING)
        for path in paths:
            self.workspace.set_failure_reason_for_path(path, "")
        manual_mapping = self._manual_lane_mapping_for_paths(paths)
        scene_region_mapping = self._scene_region_mapping_for_paths(paths)
        count_line_mapping = self._count_line_mapping_for_paths(paths)
        frame_stride = self._recommended_video_frame_stride(paths)
        approx_processed_fps = 0.0
        try:
            fps_samples = [float(self._load_video_info(path).fps or 0.0) for path in paths]
            fps_samples = [value for value in fps_samples if value > 1e-6]
            if fps_samples:
                approx_processed_fps = max(fps_samples) / max(1, int(frame_stride))
        except Exception:
            approx_processed_fps = 0.0
        self._log(
            "info",
            f"{label}: {len(paths)} video(s), scene regions on {len(scene_region_mapping)} scene(s), legacy lane override on {len(manual_mapping)} scene(s), saved count lines on {len(count_line_mapping)} scene(s), frame_stride={frame_stride}" + (f", target processed FPS~{approx_processed_fps:.1f}" if approx_processed_fps > 0 else ""),
        )

        self._run_thread = QtCore.QThread(self)
        self._run_worker = VideoProcessingWorker(
            video_paths=paths,
            lane_model_path=runtime["lane_model_path"],
            vehicle_model_path=runtime["vehicle_model_path"],
            output_dir=runtime["video_output_dir"],
            preview_dir=runtime["video_preview_dir"],
            threshold=0.3,
            layers=self.viewer_panel.layers_state(),
            manual_lane_polygons_by_path=manual_mapping,
            scene_regions_by_path=scene_region_mapping,
            count_lines_by_path=count_line_mapping,
            frame_stride=frame_stride,
        )
        self._run_worker.moveToThread(self._run_thread)

        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._log)
        self._run_worker.task_started.connect(self._on_task_started)
        self._run_worker.task_progress.connect(self._on_video_task_progress)
        self._run_worker.task_finished.connect(self._on_task_finished)
        self._run_worker.task_failed.connect(self._on_task_failed)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.finished.connect(self._run_worker.deleteLater)
        self._run_thread.finished.connect(self._on_run_thread_finished)
        self._run_thread.finished.connect(self._run_thread.deleteLater)

        self._run_thread.start()

    def _on_task_started(self, image_path: str, index: int, total: int):
        key = self._norm_path(image_path)
        self.workspace.set_status_for_path(image_path, self.workspace.STATUS_RUNNING)
        self.workspace.set_failure_reason_for_path(image_path, "")
        self._video_progress_log_state.pop(key, None)
        current = self.workspace.selected_path()
        if current and self._norm_path(current) == key:
            self._show_details_for_path(image_path)
        self._log("info", f"Running {index}/{total}: {Path(image_path).name}")

    def _recommended_video_frame_stride(self, paths: list) -> int:
        target_processed_fps = 6.0
        fps_values = []
        for path in paths:
            try:
                info = self._load_video_info(path)
            except Exception:
                continue
            fps = float(getattr(info, "fps", 0.0) or 0.0)
            if fps > 1e-6:
                fps_values.append(fps)
        if not fps_values:
            return 10
        reference_fps = max(fps_values)
        return max(1, min(30, int(math.ceil(reference_fps / target_processed_fps))))

    def _format_processing_eta(self, seconds) -> str:
        if seconds is None:
            return "unknown"
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            return "unknown"
        total_seconds = int(round(value))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def _on_video_task_progress(self, video_path: str, progress: dict, index: int, total: int):
        key = self._norm_path(video_path)
        payload = dict(progress or {})
        payload.setdefault("media_type", "video")
        payload.setdefault("original_path", str(Path(video_path).resolve(strict=False)))
        self._results_by_path[key] = payload

        current = self.workspace.selected_path()
        if current and self._norm_path(current) == key:
            self._show_details_for_path(video_path)

        processed_frames = int(payload.get("processed_frame_count", 0) or 0)
        expected_frames = int(payload.get("expected_processed_frame_count", 0) or 0)
        now = time.monotonic()
        state = self._video_progress_log_state.get(key, {})
        last_count = int(state.get("processed_frames", 0) or 0)
        last_logged_at = float(state.get("logged_at", 0.0) or 0.0)
        should_log = (
            processed_frames <= 1
            or expected_frames <= 0
            or processed_frames >= expected_frames
            or (processed_frames - last_count) >= 5
            or (now - last_logged_at) >= 8.0
        )
        if not should_log:
            return

        percent = float(payload.get("progress_percent", 0.0) or 0.0)
        current_frame_index = int(payload.get("current_frame_index", 0) or 0)
        throughput = float(payload.get("processed_frames_per_s", 0.0) or 0.0)
        eta_text = self._format_processing_eta(payload.get("estimated_remaining_s"))
        current_vehicles = int(payload.get("current_vehicle_count", 0) or 0)
        traffic_count = int(payload.get("traffic_count_total", 0) or 0)
        current_region_rule_count = int(payload.get("current_region_rule_violation_count", 0) or 0)
        region_rule_total = int(payload.get("region_rule_event_count", 0) or 0)
        scene_region_count = int(payload.get("scene_region_count", 0) or 0)
        region_rule_suffix = ""
        if scene_region_count > 0:
            region_rule_suffix = (
                f", region_rule_events={region_rule_total}, "
                f"current_region_rule_hits={current_region_rule_count}"
            )
        frame_text = f"{processed_frames}/{expected_frames}" if expected_frames > 0 else f"{processed_frames}/?"
        self._log(
            "info",
            f"Progress {index}/{total}: {Path(video_path).name}, processed={frame_text} ({percent:.1f}%), source_frame={current_frame_index}, current_vehicles={current_vehicles}, traffic_count={traffic_count}{region_rule_suffix}, throughput={throughput:.2f} fps, eta={eta_text}",
        )
        self._video_progress_log_state[key] = {
            "processed_frames": int(processed_frames),
            "logged_at": float(now),
        }

    def _on_task_finished(self, image_path: str, result: dict, index: int, total: int):
        key = self._norm_path(image_path)
        self._video_progress_log_state.pop(key, None)
        self._results_by_path[key] = result
        self.workspace.set_status_for_path(image_path, self.workspace.STATUS_DONE)
        self.workspace.set_failure_reason_for_path(image_path, "")

        current = self.workspace.selected_path()
        if current and self._norm_path(current) == key:
            self._show_details_for_path(image_path)
            self._set_preview(image_path)

        elapsed = time.monotonic() - self._run_started_at
        if result.get("media_type") == "video":
            frames = int(result.get("frame_count", 0))
            processed_frames = int(result.get("processed_frame_count", 0))
            violations = int(result.get("total_violation_instances", 0))
            vehicles = int(result.get("total_vehicle_instances", 0))
            tracks = int(result.get("confirmed_track_count", 0))
            traffic_count = int(result.get("traffic_count_total", 0))
            plate_ocr_attempts = int(result.get("total_plate_ocr_attempt_count", 0) or 0)
            plate_ocr_successes = int(result.get("total_plate_ocr_success_count", 0) or 0)
            count_source = str(result.get("count_line_source", "none"))
            scene_region_count = int(result.get("scene_region_count", 0) or 0)
            region_rule_events = int(result.get("region_rule_event_count", 0) or 0)
            no_parking_events = int(result.get("region_rule_no_parking_count", 0) or 0)
            no_non_motor_events = int(result.get("region_rule_no_non_motor_count", 0) or 0)
            no_wrong_way_events = int(result.get("region_rule_no_wrong_way_count", 0) or 0)
            if count_source == "auto_center_line":
                count_source = "auto"
            region_rule_suffix = ""
            if scene_region_count > 0:
                region_rule_suffix = (
                    f", region_rules={region_rule_events}"
                    f" [no_parking={no_parking_events}, no_non_motor={no_non_motor_events}, no_wrong_way={no_wrong_way_events}]"
                )
            self._log(
                "success",
                f"Done {index}/{total}: {Path(image_path).name}, processed_frames={processed_frames}/{frames}, vehicle_instances={vehicles}, tracks={tracks}, traffic_count={traffic_count}, plate_ocr={plate_ocr_attempts}/{plate_ocr_successes}, counter={count_source}, violations={violations}{region_rule_suffix}, elapsed={elapsed:.1f}s",
            )
            return

        vehicles = int(result.get("vehicle_count", 0))
        violations = int(result.get("violation_count", 0))
        lane_source = str(result.get("lane_source", "auto"))
        self._log(
            "success",
            f"Done {index}/{total}: {Path(image_path).name}, vehicles={vehicles}, violations={violations}, lane={lane_source}, elapsed={elapsed:.1f}s",
        )

    def _on_task_failed(self, image_path: str, error: str, index: int, total: int):
        self._video_progress_log_state.pop(self._norm_path(image_path), None)
        self.workspace.set_status_for_path(image_path, self.workspace.STATUS_FAILED)
        self.workspace.set_failure_reason_for_path(image_path, error)
        current = self.workspace.selected_path()
        if current and self._norm_path(current) == self._norm_path(image_path):
            self._show_details_for_path(image_path)
        elapsed = time.monotonic() - self._run_started_at
        self._log("error", f"Failed {index}/{total}: {Path(image_path).name}, error={error}, elapsed={elapsed:.1f}s")

    def _on_run_finished(self):
        if not self._is_running:
            return
        elapsed = time.monotonic() - self._run_started_at
        if self._run_stop_requested:
            self._mark_interrupted_run_paths()
        self._is_running = False
        if self._run_stop_requested:
            self._log("warning", f"{self._run_label} interrupted after {elapsed:.1f}s")
        else:
            self._log("info", f"{self._run_label} finished in {elapsed:.1f}s")
        self._run_worker = None
        self._run_stop_requested = False
        self._active_run_paths = []

    def _on_run_thread_finished(self):
        self._run_thread = None
        pending = self._pending_run_request
        if pending is None or self._is_running:
            return
        self._pending_run_request = None
        QtCore.QTimer.singleShot(
            0,
            lambda request=pending: self._start_run(list(request.get("paths", [])), label=str(request.get("label", "Run"))),
        )

    def _pick_detection_hit(self, detections, point):
        px, py = point
        near_threshold = 8.0
        candidates = []
        for det_index, det in enumerate(detections):
            footprint = np.array(det.get("footprint", []), dtype=np.float32)
            corners = np.array(det.get("corners_2d", []), dtype=np.float32)

            inside_dist = -1.0
            if footprint.ndim == 2 and footprint.shape[0] >= 3:
                inside_dist = float(cv2.pointPolygonTest(footprint, (px, py), True))

            edge_dist = self._distance_to_detection_edges(point, footprint, corners)
            hit_priority = 0
            closeness = 0.0
            if inside_dist >= 0:
                hit_priority = 2
                closeness = inside_dist
            elif edge_dist <= near_threshold:
                hit_priority = 1
                closeness = near_threshold - edge_dist
            else:
                continue

            depth_hint = float(det.get("depth_hint", 0.0))
            confidence = float(det.get("confidence", 0.0))
            score = (hit_priority, closeness, depth_hint, confidence)
            candidates.append((score, det, det_index))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return {
            "detection": candidates[0][1],
            "candidate_count": len(candidates),
            "list_index": int(candidates[0][2]),
        }

    def _pick_lane_hit(self, lane_polygons, point):
        px, py = point
        best = None
        for idx, polygon in enumerate(lane_polygons, start=1):
            poly = np.array(polygon, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            dist = float(cv2.pointPolygonTest(poly, (px, py), True))
            if dist < 0:
                continue
            if best is None or dist > best["distance"]:
                best = {"index": idx, "polygon": polygon, "distance": dist}
        return best

    def _distance_to_detection_edges(self, point, footprint, corners):
        segments = []
        if footprint.ndim == 2 and footprint.shape[0] >= 2:
            for idx in range(len(footprint)):
                a = footprint[idx]
                b = footprint[(idx + 1) % len(footprint)]
                segments.append((a, b))

        if corners.ndim == 2 and corners.shape[0] >= 8:
            top_face = [0, 3, 7, 4]
            for idx in range(len(top_face)):
                a = corners[top_face[idx]]
                b = corners[top_face[(idx + 1) % len(top_face)]]
                segments.append((a, b))
            for start, end in [(0, 1), (3, 2), (7, 6), (4, 5)]:
                segments.append((corners[start], corners[end]))

        if not segments:
            return float("inf")
        return min(self._point_to_segment_distance(point, a, b) for a, b in segments)

    def _point_to_segment_distance(self, point, a, b):
        px, py = point
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _active_detection_index(self):
        if self._hover_detection_index is not None:
            return self._hover_detection_index
        return self._selected_detection_index

    def _show_details_for_path(self, file_path: str):
        key = self._norm_path(file_path)
        result = self._results_by_path.get(key)
        status = self.workspace.status_of_path(file_path) or ""
        failure_reason = self.workspace.failure_reason_of_path(file_path)
        if self._is_video_path(file_path):
            video_info = None
            try:
                video_info = self._load_video_info(file_path)
            except Exception:
                video_info = None
            self.violations.show_video_details(
                video_path=file_path,
                status=status,
                failure_reason=failure_reason,
                video_info=video_info,
                result=result,
            )
            return
        self.violations.show_file_details(
            image_path=file_path,
            status=status,
            failure_reason=failure_reason,
            result=result,
        )

    def _norm_path(self, path: str) -> str:
        return str(Path(path).resolve(strict=False))

    def _runtime_paths(self):
        src_dir = Path(__file__).resolve().parent.parent
        project_root = src_dir.parent
        return {
            "lane_model_path": project_root / "src" / "models" / "best.pt",
            "vehicle_model_path": project_root / "src" / "models" / "yolo11l.pt",
            "output_dir": project_root / "data" / "images",
            "db_path": project_root / "data" / "db" / "traffic_scan.db",
            "scene_profiles_dir": project_root / "data" / "scene_profiles",
            "video_preview_dir": project_root / "data" / "video_previews",
            "video_output_dir": project_root / "data" / "videos",
        }

    def _layer_name(self, key: str) -> str:
        return {
            "lane_mask": "Lane Mask",
            "footprint": "Footprint",
            "boxes_3d": "3D Boxes",
            "labels": "Labels",
        }.get(key, key)

    def _log(self, level: str, message: str):
        print(f"[{level}] {message}")
