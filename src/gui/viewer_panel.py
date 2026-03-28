from pathlib import Path

try:
    from .qt import QtCore, QtGui, QtWidgets
    from .image_viewer import ImageViewer
except Exception:
    try:
        from gui.qt import QtCore, QtGui, QtWidgets
        from gui.image_viewer import ImageViewer
    except Exception:
        from qt import QtCore, QtGui, QtWidgets
        from image_viewer import ImageViewer


class ViewerPanel(QtWidgets.QFrame):
    resized = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    zoom_changed = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    image_clicked = QtCore.pyqtSignal(int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int, int)
    image_hovered = QtCore.pyqtSignal(int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int, int)
    image_hover_left = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    layer_toggled = QtCore.pyqtSignal(str, bool) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, bool)
    manual_roi_start_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_finish_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_clear_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_save_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_changed = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_finished = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_context_requested = (
        QtCore.pyqtSignal(int, object, object)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(int, object, object)
    )
    region_direction_finished = (
        QtCore.pyqtSignal(int, object)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(int, object)
    )
    video_play_toggled = QtCore.pyqtSignal(bool) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(bool)
    video_seek_requested = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    video_step_requested = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers_state = {
            "lane_mask": True,
            "footprint": True,
            "boxes_3d": True,
            "labels": False,
        }
        self._updating_video_slider = False

        self.setStyleSheet("background:#0b1220;")

        self.viewer = ImageViewer("Drop/select an image to preview")
        self.viewer.setStyleSheet("background:#05070c;border-left:1px solid #243041;border-right:1px solid #243041;")
        self.viewer.setMinimumSize(240, 180)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setStyleSheet(
            "QFrame{background:#111827;border:1px solid #243041;border-bottom:0px;}"
            "QLabel{color:#cbd5e1;}"
            "QToolButton{background:#172033;border:1px solid #314055;color:#dbe4ee;"
            "padding:7px 11px;border-radius:9px;}"
            "QToolButton:hover{background:#1e293b;}"
            "QToolButton:checked{background:#1d4ed8;border-color:#2563eb;color:#ffffff;}"
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;padding:7px 11px;border-radius:9px;}"
            "QPushButton:hover{background:#273449;}"
            "QPushButton:disabled{color:#64748b;border-color:#1f2937;background:#111827;}"
        )
        toolbar_root = QtWidgets.QVBoxLayout(self.toolbar)
        toolbar_root.setContentsMargins(14, 10, 14, 10)
        toolbar_root.setSpacing(8)

        toolbar_row = QtWidgets.QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)

        tools_title = QtWidgets.QLabel("Preview Tools")
        tools_title.setStyleSheet("color:#f8fafc;font-size:12px;font-weight:600;padding-right:6px;")
        toolbar_row.addWidget(tools_title, 0)

        self.btn_layer_lane = self._make_toggle_button("Lane")
        self.btn_layer_footprint = self._make_toggle_button("Footprint")
        self.btn_layer_boxes = self._make_toggle_button("3D Box")
        self.btn_layer_labels = self._make_toggle_button("Labels")
        toolbar_row.addWidget(self.btn_layer_lane, 0)
        toolbar_row.addWidget(self.btn_layer_footprint, 0)
        toolbar_row.addWidget(self.btn_layer_boxes, 0)
        toolbar_row.addWidget(self.btn_layer_labels, 0)

        toolbar_row.addSpacing(12)
        toolbar_row.addWidget(self._make_separator(), 0)
        toolbar_row.addSpacing(12)

        self.btn_roi_draw = self._make_action_button("Draw ROI")
        self.btn_roi_finish = self._make_action_button("Finish")
        self.btn_roi_clear = self._make_action_button("Clear")
        self.btn_roi_save = self._make_action_button("Save ROI")
        toolbar_row.addWidget(self.btn_roi_draw, 0)
        toolbar_row.addWidget(self.btn_roi_finish, 0)
        toolbar_row.addWidget(self.btn_roi_clear, 0)
        toolbar_row.addWidget(self.btn_roi_save, 0)
        toolbar_row.addStretch(1)
        toolbar_root.addLayout(toolbar_row)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self.lbl_roi_status = QtWidgets.QLabel("ROI: auto lane region")
        self.lbl_roi_status.setWordWrap(True)
        self.lbl_roi_status.setStyleSheet("color:#cbd5e1;font-size:11px;")
        self.lbl_roi_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.lbl_toolbar_hint = QtWidgets.QLabel("Left click to add points, right click or double click to finish")
        self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")
        status_row.addWidget(self.lbl_roi_status, 1)
        status_row.addWidget(self.lbl_toolbar_hint, 0, QtCore.Qt.AlignRight)
        toolbar_root.addLayout(status_row)

        self.video_controls = QtWidgets.QFrame()
        self.video_controls.setStyleSheet(
            "QFrame{background:#0f172a;border-left:1px solid #243041;border-right:1px solid #243041;border-bottom:1px solid #243041;}"
            "QLabel{color:#cbd5e1;font-size:11px;}"
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;padding:6px 10px;border-radius:8px;}"
            "QPushButton:hover{background:#273449;}"
            "QPushButton:disabled{color:#64748b;border-color:#1f2937;background:#111827;}"
            "QPushButton:checked{background:#1d4ed8;border-color:#2563eb;color:#ffffff;}"
            "QSlider::groove:horizontal{height:6px;background:#1f2937;border-radius:3px;}"
            "QSlider::handle:horizontal{width:14px;margin:-5px 0;background:#60a5fa;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#2563eb;border-radius:3px;}"
        )
        video_root = QtWidgets.QVBoxLayout(self.video_controls)
        video_root.setContentsMargins(12, 8, 12, 8)
        video_root.setSpacing(6)

        video_row = QtWidgets.QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(8)
        self.btn_video_prev = self._make_action_button("Prev")
        self.btn_video_play = self._make_action_button("Play")
        self.btn_video_play.setCheckable(True)
        self.btn_video_next = self._make_action_button("Next")
        self.video_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(0)
        self.video_slider.setEnabled(False)
        self.lbl_video_position = QtWidgets.QLabel("Frame -- / --")
        self.lbl_video_position.setStyleSheet("color:#e2e8f0;font-size:11px;font-weight:600;")
        video_row.addWidget(self.btn_video_prev, 0)
        video_row.addWidget(self.btn_video_play, 0)
        video_row.addWidget(self.btn_video_next, 0)
        video_row.addWidget(self.video_slider, 1)
        video_row.addWidget(self.lbl_video_position, 0)
        video_root.addLayout(video_row)

        video_meta_row = QtWidgets.QHBoxLayout()
        video_meta_row.setContentsMargins(0, 0, 0, 0)
        video_meta_row.setSpacing(8)
        self.lbl_video_source = QtWidgets.QLabel("Preview: original video")
        self.lbl_video_source.setStyleSheet("color:#94a3b8;font-size:11px;")
        self.lbl_video_hint = QtWidgets.QLabel("Use play, slider, or frame stepping to inspect the video")
        self.lbl_video_hint.setStyleSheet("color:#64748b;font-size:11px;")
        video_meta_row.addWidget(self.lbl_video_source, 1)
        video_meta_row.addWidget(self.lbl_video_hint, 0, QtCore.Qt.AlignRight)
        video_root.addLayout(video_meta_row)
        self.video_controls.setVisible(False)

        self.viewer_status = QtWidgets.QFrame()
        self.viewer_status.setStyleSheet(
            "QFrame{background:#0a0f18;border:1px solid #243041;border-top:0px;border-radius:0px;}"
            "QLabel{color:#cbd5e1;font-size:12px;}"
        )
        status_root = QtWidgets.QHBoxLayout(self.viewer_status)
        status_root.setContentsMargins(12, 8, 12, 8)
        status_root.setSpacing(12)
        self.lbl_view_file = QtWidgets.QLabel("No image")
        self.lbl_view_zoom = QtWidgets.QLabel("Zoom --")
        status_root.addWidget(self.lbl_view_file, 1)
        status_root.addWidget(self.lbl_view_zoom, 0)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.toolbar, 0)
        root.addWidget(self.video_controls, 0)
        root.addWidget(self.viewer, 1)
        root.addWidget(self.viewer_status, 0)

        self.viewer.resized.connect(self.resized)
        self.viewer.zoom_changed.connect(self._on_viewer_zoom_changed)
        self.viewer.zoom_changed.connect(self.zoom_changed)
        self.viewer.point_clicked.connect(self.image_clicked)
        self.viewer.point_hovered.connect(self.image_hovered)
        self.viewer.hover_left.connect(self.image_hover_left)
        self.viewer.manual_roi_changed.connect(self._on_manual_roi_changed)
        self.viewer.manual_roi_finished.connect(self._on_manual_roi_finished)
        self.viewer.manual_roi_drawing_changed.connect(self._on_manual_roi_drawing_changed)
        self.viewer.manual_roi_context_requested.connect(self.manual_roi_context_requested)
        self.viewer.region_direction_finished.connect(self.region_direction_finished)

        self.btn_layer_lane.toggled.connect(lambda value: self._set_layer_enabled("lane_mask", value))
        self.btn_layer_footprint.toggled.connect(lambda value: self._set_layer_enabled("footprint", value))
        self.btn_layer_boxes.toggled.connect(lambda value: self._set_layer_enabled("boxes_3d", value))
        self.btn_layer_labels.toggled.connect(lambda value: self._set_layer_enabled("labels", value))

        self.btn_roi_draw.clicked.connect(self.manual_roi_start_requested.emit)
        self.btn_roi_finish.clicked.connect(self.manual_roi_finish_requested.emit)
        self.btn_roi_clear.clicked.connect(self.manual_roi_clear_requested.emit)
        self.btn_roi_save.clicked.connect(self.manual_roi_save_requested.emit)

        self.btn_video_prev.clicked.connect(lambda: self.video_step_requested.emit(-1))
        self.btn_video_next.clicked.connect(lambda: self.video_step_requested.emit(1))
        self.btn_video_play.toggled.connect(self._on_video_play_toggled)
        self.video_slider.valueChanged.connect(self._on_video_slider_changed)

        self.btn_layer_lane.setChecked(self._layers_state["lane_mask"])
        self.btn_layer_footprint.setChecked(self._layers_state["footprint"])
        self.btn_layer_boxes.setChecked(self._layers_state["boxes_3d"])
        self.btn_layer_labels.setChecked(self._layers_state["labels"])
        self.btn_roi_finish.setEnabled(False)

    def _make_toggle_button(self, text: str):
        button = QtWidgets.QToolButton()
        button.setCheckable(True)
        button.setText(text)
        return button

    def _make_action_button(self, text: str):
        button = QtWidgets.QPushButton(text)
        return button

    def _make_separator(self):
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setStyleSheet("background:#243041;color:#243041;")
        separator.setFixedWidth(1)
        return separator

    def _on_viewer_zoom_changed(self, percent: int):
        if percent <= 0:
            self.lbl_view_zoom.setText("Zoom --")
            return
        self.lbl_view_zoom.setText(f"Zoom {percent}%")

    def _set_layer_enabled(self, key: str, enabled: bool):
        self._layers_state[key] = enabled
        self.layer_toggled.emit(key, enabled)

    def _on_manual_roi_changed(self, regions):
        self._update_roi_status(regions)
        self.manual_roi_changed.emit(regions)

    def _on_manual_roi_finished(self, points):
        self._update_roi_status(points)
        self.manual_roi_finished.emit(points)

    def _on_manual_roi_drawing_changed(self, active: bool):
        self.btn_roi_draw.setEnabled(not active)
        self.btn_roi_finish.setEnabled(active)
        if active:
            self.lbl_roi_status.setText("ROI: drawing in progress")
            self.lbl_toolbar_hint.setText("ROI mode: left click to add points, right click or double click to finish")
            self.lbl_toolbar_hint.setStyleSheet("color:#fbbf24;font-size:11px;font-weight:600;")
        else:
            self._update_roi_status(self.viewer.manual_roi_regions())
            self.lbl_toolbar_hint.setText("Click a region to select it. Right click binds rules, Delete removes selected, Clear removes all")
            self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")

    def _on_video_play_toggled(self, checked: bool):
        self.btn_video_play.setText("Pause" if checked else "Play")
        self.video_play_toggled.emit(bool(checked))

    def _on_video_slider_changed(self, value: int):
        if self._updating_video_slider:
            return
        self.video_seek_requested.emit(int(value))

    def _update_roi_status(self, regions):
        count = len(regions or [])
        selected_index = self.viewer.selected_manual_roi_index()
        if count > 0:
            if selected_index is not None:
                self.lbl_roi_status.setText(f"ROI: {count} manual region(s), selected #{selected_index + 1}")
            else:
                self.lbl_roi_status.setText(f"ROI: {count} manual region(s)")
        else:
            self.lbl_roi_status.setText("ROI: auto lane region")

    def layers_state(self):
        return dict(self._layers_state)

    def set_image_path(self, file_path: str):
        path = Path(file_path)
        self.lbl_view_file.setText(path.name)
        pixmap = QtGui.QPixmap(str(path))
        self.viewer.set_pixmap(None if pixmap.isNull() else pixmap, reset_view=True)
        if pixmap.isNull():
            self.lbl_view_file.setText("Preview unavailable")

    def set_pixmap(self, pixmap, *, label: str = "", reset_view: bool = True):
        if label:
            self.lbl_view_file.setText(label)
        self.viewer.set_pixmap(pixmap, reset_view=reset_view)
        if pixmap is None or pixmap.isNull():
            self.lbl_view_file.setText("Preview unavailable")

    def set_video_controls_visible(self, visible: bool):
        self.video_controls.setVisible(bool(visible))
        if visible:
            return
        self._updating_video_slider = True
        try:
            self.video_slider.setRange(0, 0)
            self.video_slider.setValue(0)
        finally:
            self._updating_video_slider = False
        self.btn_video_prev.setEnabled(False)
        self.btn_video_next.setEnabled(False)
        self.btn_video_play.setEnabled(False)
        self.btn_video_play.blockSignals(True)
        self.btn_video_play.setChecked(False)
        self.btn_video_play.blockSignals(False)
        self.btn_video_play.setText("Play")
        self.lbl_video_position.setText("Frame -- / --")
        self.lbl_video_source.setText("Preview: original video")
        self.lbl_video_hint.setText("Use play, slider, or frame stepping to inspect the video")

    def set_video_playback_state(
        self,
        *,
        visible: bool,
        is_playing: bool = False,
        current_frame: int = 0,
        total_frames: int = 0,
        position_text: str = "",
        source_text: str = "",
        hint_text: str = "",
    ):
        self.set_video_controls_visible(visible)
        if not visible:
            return

        total_frames = max(0, int(total_frames))
        current_frame = max(0, int(current_frame))
        max_index = max(0, total_frames - 1)
        current_frame = min(current_frame, max_index)

        self._updating_video_slider = True
        try:
            self.video_slider.setRange(0, max_index)
            self.video_slider.setEnabled(total_frames > 1)
            self.video_slider.setValue(current_frame)
        finally:
            self._updating_video_slider = False

        self.btn_video_prev.setEnabled(total_frames > 0 and current_frame > 0)
        self.btn_video_next.setEnabled(total_frames > 0 and current_frame < max_index)
        self.btn_video_play.setEnabled(total_frames > 1)
        self.btn_video_play.blockSignals(True)
        self.btn_video_play.setChecked(bool(is_playing))
        self.btn_video_play.blockSignals(False)
        self.btn_video_play.setText("Pause" if is_playing else "Play")
        self.lbl_video_position.setText(position_text or f"Frame {current_frame + 1}/{max(1, total_frames)}")
        self.lbl_video_source.setText(source_text or "Preview: original video")
        self.lbl_video_hint.setText(hint_text or "Use play, slider, or frame stepping to inspect the video")

    def refresh(self):
        self.viewer.refresh()

    def set_manual_roi(self, points):
        self.viewer.set_manual_roi(points)
        self._update_roi_status(points)

    def manual_roi_regions(self):
        return self.viewer.manual_roi_regions()

    def start_manual_roi_drawing(self, *, clear_existing: bool = False):
        self.viewer.start_manual_roi_drawing(clear_existing=clear_existing)

    def finish_manual_roi_drawing(self):
        new_region = self.viewer.finish_manual_roi_drawing()
        self._update_roi_status(self.viewer.manual_roi_regions())
        return new_region

    def clear_manual_roi(self):
        self.viewer.clear_manual_roi()
        self._update_roi_status([])

    def delete_selected_manual_roi(self):
        deleted = self.viewer.delete_selected_manual_roi()
        self._update_roi_status(self.viewer.manual_roi_regions())
        return deleted

    def set_region_direction_lines(self, direction_lines):
        self.viewer.set_region_direction_lines(direction_lines)

    def region_direction_lines(self):
        return self.viewer.region_direction_lines()

    def start_region_direction_drawing(self, region_index: int):
        self.viewer.start_region_direction_drawing(region_index)

    def is_manual_roi_drawing(self) -> bool:
        return self.viewer.is_manual_roi_drawing()
