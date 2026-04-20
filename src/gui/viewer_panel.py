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
    lane_recognition_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_start_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_finish_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_clear_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_save_requested = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_changed = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_finished = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_selection_changed = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
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
        self._processing_layers_state = {
            "lane_mask": False,
            "footprint": True,
            "boxes_3d": True,
            "labels": False,
        }
        self._preview_display_mode = "仅显示底面"
        self._scene_loaded = False
        self._updating_video_slider = False
        self._lane_recognition_busy = False

        self.setStyleSheet("background:#0b1220;")

        self.viewer = ImageViewer(self._placeholder_html())
        self.viewer.setStyleSheet("background:#05070c;border-left:1px solid #243041;border-right:1px solid #243041;")
        self.viewer.setMinimumSize(240, 180)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setStyleSheet(
            "QFrame{background:#111827;border:1px solid #243041;border-bottom:0px;}"
            "QLabel{color:#cbd5e1;}"
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;padding:8px 12px;border-radius:10px;}"
            "QPushButton:hover{background:#273449;}"
            "QPushButton:disabled{color:#64748b;border-color:#1f2937;background:#111827;}"
        )
        toolbar_root = QtWidgets.QVBoxLayout(self.toolbar)
        toolbar_root.setContentsMargins(14, 10, 14, 10)
        toolbar_root.setSpacing(8)

        toolbar_row = QtWidgets.QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)

        tools_title = QtWidgets.QLabel("区域操作")
        tools_title.setStyleSheet("color:#f8fafc;font-size:13px;font-weight:600;padding-right:6px;")
        toolbar_row.addWidget(tools_title, 0)

        self.btn_roi_draw = self._make_action_button("框选车道")
        self.btn_lane_recognize = self._make_action_button("识别应急车道")
        self.btn_roi_finish = self._make_action_button("完成框选")
        self.btn_roi_clear = self._make_action_button("清空区域")
        self.btn_roi_save = self._make_action_button("保存规则")
        self.btn_roi_save.setVisible(False)
        toolbar_row.addWidget(self.btn_roi_draw, 0)
        toolbar_row.addWidget(self.btn_lane_recognize, 0)
        toolbar_row.addWidget(self.btn_roi_finish, 0)
        toolbar_row.addWidget(self.btn_roi_clear, 0)
        toolbar_row.addWidget(self.btn_roi_save, 0)
        toolbar_row.addStretch(1)
        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItems(["无", "仅显示底面", "3D框"])
        self.display_mode_combo.setCurrentText(self._preview_display_mode)
        self.display_mode_combo.setStyleSheet(
            "QComboBox{background:#0f172a;border:1px solid #334155;color:#e5e7eb;padding:6px 12px;border-radius:10px;min-width:132px;}"
            "QComboBox::drop-down{border:0px;width:22px;}"
            "QComboBox QAbstractItemView{background:#0f172a;border:1px solid #334155;color:#e5e7eb;selection-background-color:#1d4ed8;}"
        )
        toolbar_row.addWidget(self.display_mode_combo, 0)
        toolbar_root.addLayout(toolbar_row)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self.lbl_roi_status = QtWidgets.QLabel("当前还没有区域")
        self.lbl_roi_status.setWordWrap(True)
        self.lbl_roi_status.setStyleSheet("color:#cbd5e1;font-size:11px;")
        self.lbl_roi_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.lbl_toolbar_hint = QtWidgets.QLabel("从左侧导入素材后，可在这里框选区域并查看分析结果")
        self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")
        self.lbl_toolbar_hint.setWordWrap(True)
        self.lbl_toolbar_hint.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        status_row.addWidget(self.lbl_roi_status, 1)
        status_row.addWidget(self.lbl_toolbar_hint, 1)
        self.lbl_roi_status.setVisible(False)
        self.lbl_toolbar_hint.setVisible(False)
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
        self.btn_video_prev = self._make_action_button("上一帧")
        self.btn_video_play = self._make_action_button("播放")
        self.btn_video_play.setCheckable(True)
        self.btn_video_next = self._make_action_button("下一帧")
        self.video_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(0)
        self.video_slider.setEnabled(False)
        self.lbl_video_position = QtWidgets.QLabel("帧 -- / --")
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
        self.lbl_video_source = QtWidgets.QLabel("视频预览")
        self.lbl_video_source.setStyleSheet("color:#94a3b8;font-size:11px;")
        self.lbl_video_hint = QtWidgets.QLabel("可播放、拖动时间条或逐帧查看")
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
        self.lbl_view_file = QtWidgets.QLabel("未选择素材")
        self.lbl_view_zoom = QtWidgets.QLabel("缩放 --")
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
        self.viewer.manual_roi_selection_changed.connect(self._on_manual_roi_selection_changed)
        self.viewer.manual_roi_context_requested.connect(self.manual_roi_context_requested)
        self.viewer.region_direction_finished.connect(self.region_direction_finished)
        self.viewer.region_direction_drawing_changed.connect(self._on_region_direction_drawing_changed)

        self.btn_roi_draw.clicked.connect(self.manual_roi_start_requested.emit)
        self.btn_lane_recognize.clicked.connect(self.lane_recognition_requested.emit)
        self.btn_roi_finish.clicked.connect(self.manual_roi_finish_requested.emit)
        self.btn_roi_clear.clicked.connect(self.manual_roi_clear_requested.emit)
        self.btn_roi_save.clicked.connect(self.manual_roi_save_requested.emit)
        self.display_mode_combo.currentTextChanged.connect(self.set_display_mode)

        self.btn_video_prev.clicked.connect(lambda: self.video_step_requested.emit(-1))
        self.btn_video_next.clicked.connect(lambda: self.video_step_requested.emit(1))
        self.btn_video_play.toggled.connect(self._on_video_play_toggled)
        self.video_slider.valueChanged.connect(self._on_video_slider_changed)

        self._refresh_toolbar_state()
        self.show_empty_state()

    def _make_action_button(self, text: str):
        return QtWidgets.QPushButton(text)

    def _placeholder_html(self) -> str:
        return (
            "<div style='text-align:center;color:#94a3b8;'>"
            "<div style='font-size:22px;color:#e2e8f0;font-weight:600;margin-bottom:10px;'>"
            "左侧导入图片或视频后，这里会显示预览与分析结果"
            "</div>"
            "<div style='font-size:13px;line-height:1.8;'>"
            "1. 从左侧导入素材<br/>"
            "2. 需要时框选车道并设置规则<br/>"
            "3. 点击“开始分析”查看结果"
            "</div>"
            "<div style='font-size:12px;color:#64748b;margin-top:16px;'>"
            "分析完成后，可在这里点击区域查看规则"
            "</div>"
            "</div>"
        )

    def _on_viewer_zoom_changed(self, percent: int):
        if percent <= 0:
            self.lbl_view_zoom.setText("缩放 --")
            return
        self.lbl_view_zoom.setText(f"缩放 {percent}%")

    def _set_layer_enabled(self, key: str, enabled: bool):
        self._processing_layers_state[key] = enabled
        self.layer_toggled.emit(key, enabled)

    def set_display_mode(self, mode: str):
        normalized = str(mode or "").strip()
        if normalized not in {"无", "仅显示底面", "3D框"}:
            normalized = "仅显示底面"
        if self.display_mode_combo.currentText() != normalized:
            self.display_mode_combo.blockSignals(True)
            self.display_mode_combo.setCurrentText(normalized)
            self.display_mode_combo.blockSignals(False)
        if normalized == self._preview_display_mode:
            return
        self._preview_display_mode = normalized
        self.layer_toggled.emit("preview_mode", True)

    def preview_layers_state(self):
        state = {
            "lane_mask": False,
            "footprint": False,
            "boxes_3d": False,
            "labels": False,
        }
        if self._preview_display_mode == "仅显示底面":
            state["footprint"] = True
        elif self._preview_display_mode == "3D框":
            state["footprint"] = True
            state["boxes_3d"] = True
        return state

    def display_mode_text(self) -> str:
        return str(self._preview_display_mode or "仅显示底面")

    def _on_manual_roi_changed(self, regions):
        self._update_roi_status(regions)
        self._refresh_toolbar_state()
        self.manual_roi_changed.emit(regions)

    def _on_manual_roi_finished(self, points):
        self._update_roi_status(points)
        self._refresh_toolbar_state()
        self.manual_roi_finished.emit(points)

    def _on_manual_roi_selection_changed(self, region_index):
        self._update_roi_status(self.viewer.manual_roi_regions())
        self.manual_roi_selection_changed.emit(region_index)

    def _on_manual_roi_drawing_changed(self, active: bool):
        self._refresh_toolbar_state()
        if active:
            self.lbl_roi_status.setText("正在框选车道")
            self.lbl_toolbar_hint.setText("左键依次点击，回到起点闭合；按 Esc 取消当前框选")
            self.lbl_toolbar_hint.setStyleSheet("color:#fbbf24;font-size:11px;font-weight:600;")
        else:
            self._update_roi_status(self.viewer.manual_roi_regions())
            self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")
            if self._scene_loaded:
                self.lbl_toolbar_hint.setText("点击中间画面中的某个区域查看或编辑规则")
            else:
                self.lbl_toolbar_hint.setText("从左侧导入素材后，可在这里框选车道并查看分析结果")

    def _on_region_direction_drawing_changed(self, active: bool):
        self._refresh_toolbar_state()
        if active:
            self.lbl_toolbar_hint.setText("请在选中区域内点击两次设置允许方向")
            self.lbl_toolbar_hint.setStyleSheet("color:#fbbf24;font-size:11px;font-weight:600;")
        else:
            self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")
            if self.viewer.is_manual_roi_drawing():
                self.lbl_toolbar_hint.setText("左键依次点击，回到起点闭合；按 Esc 取消当前框选")
            elif self._scene_loaded:
                self.lbl_toolbar_hint.setText("点击中间画面中的某个区域查看或编辑规则")
            else:
                self.lbl_toolbar_hint.setText("从左侧导入素材后，可在这里框选车道并查看分析结果")

    def _on_video_play_toggled(self, checked: bool):
        self.btn_video_play.setText("暂停" if checked else "播放")
        self.video_play_toggled.emit(bool(checked))

    def _on_video_slider_changed(self, value: int):
        if self._updating_video_slider:
            return
        self.video_seek_requested.emit(int(value))

    def _update_roi_status(self, regions):
        if not self._scene_loaded:
            self.lbl_roi_status.setText("当前还没有区域")
            return
        count = len(regions or [])
        selected_index = self.viewer.selected_manual_roi_index()
        if count <= 0:
            self.lbl_roi_status.setText("当前还没有区域")
            return
        if selected_index is not None:
            self.lbl_roi_status.setText(f"已选中区域 {selected_index + 1} / 共 {count} 个区域")
            return
        self.lbl_roi_status.setText(f"已创建 {count} 个区域，点击预览区可选中区域")

    def _refresh_toolbar_state(self):
        has_regions = len(self.viewer.manual_roi_regions()) > 0
        is_drawing = self.viewer.is_manual_roi_drawing()
        is_direction = self.viewer.is_region_direction_drawing()
        can_edit = self._scene_loaded and not is_direction and not self._lane_recognition_busy
        self.btn_roi_draw.setEnabled(can_edit and not is_drawing)
        self.btn_lane_recognize.setEnabled(can_edit and not is_drawing)
        self.btn_roi_finish.setEnabled(can_edit and is_drawing)
        self.btn_roi_clear.setEnabled(self._scene_loaded and has_regions and not is_direction and not self._lane_recognition_busy)
        self.btn_roi_save.setEnabled(self._scene_loaded and not is_drawing and not is_direction and not self._lane_recognition_busy)

    def layers_state(self):
        return dict(self._processing_layers_state)

    def set_lane_recognition_busy(self, busy: bool):
        self._lane_recognition_busy = bool(busy)
        self.btn_lane_recognize.setText("识别中..." if self._lane_recognition_busy else "识别应急车道")
        self._refresh_toolbar_state()

    def show_empty_state(self):
        self._scene_loaded = False
        self.viewer.clear(self._placeholder_html())
        self.set_video_controls_visible(False)
        self.lbl_view_file.setText("未选择素材")
        self.lbl_view_zoom.setText("缩放 --")
        self.lbl_roi_status.setText("当前还没有区域")
        self.lbl_toolbar_hint.setText("从左侧导入素材后，可在这里框选车道并查看分析结果")
        self.lbl_toolbar_hint.setStyleSheet("color:#64748b;font-size:11px;")
        self._refresh_toolbar_state()

    def set_image_path(self, file_path: str):
        path = Path(file_path)
        self._scene_loaded = True
        self.lbl_view_file.setText(path.name)
        pixmap = QtGui.QPixmap(str(path))
        self.viewer.set_pixmap(None if pixmap.isNull() else pixmap, reset_view=True)
        if pixmap.isNull():
            self.lbl_view_file.setText("预览不可用")
        self._update_roi_status(self.viewer.manual_roi_regions())
        self._refresh_toolbar_state()

    def set_pixmap(self, pixmap, *, label: str = "", reset_view: bool = True):
        self._scene_loaded = True
        if label:
            self.lbl_view_file.setText(label)
        self.viewer.set_pixmap(pixmap, reset_view=reset_view)
        if pixmap is None or pixmap.isNull():
            self.lbl_view_file.setText("预览不可用")
        self._update_roi_status(self.viewer.manual_roi_regions())
        self._refresh_toolbar_state()

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
        self.btn_video_play.setText("播放")
        self.lbl_video_position.setText("帧 -- / --")
        self.lbl_video_source.setText("视频预览")
        self.lbl_video_hint.setText("可播放、拖动时间条或逐帧查看")

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
        self.btn_video_play.setText("暂停" if is_playing else "播放")
        self.lbl_video_position.setText(position_text or f"帧 {current_frame + 1}/{max(1, total_frames)}")
        self.lbl_video_source.setText(source_text or "视频预览")
        self.lbl_video_hint.setText(hint_text or "可播放、拖动时间条或逐帧查看")

    def refresh(self):
        self.viewer.refresh()

    def set_manual_roi(self, points, selected_index=None):
        self.viewer.set_manual_roi(points, selected_index=selected_index)
        self._update_roi_status(self.viewer.manual_roi_regions())
        self._refresh_toolbar_state()

    def manual_roi_regions(self):
        return self.viewer.manual_roi_regions()

    def selected_manual_roi_index(self):
        return self.viewer.selected_manual_roi_index()

    def start_manual_roi_drawing(self, *, clear_existing: bool = False):
        self.viewer.start_manual_roi_drawing(clear_existing=clear_existing)
        self._refresh_toolbar_state()

    def finish_manual_roi_drawing(self):
        new_region = self.viewer.finish_manual_roi_drawing()
        self._update_roi_status(self.viewer.manual_roi_regions())
        self._refresh_toolbar_state()
        return new_region

    def clear_manual_roi(self):
        self.viewer.clear_manual_roi()
        self._update_roi_status([])
        self._refresh_toolbar_state()

    def delete_selected_manual_roi(self):
        deleted = self.viewer.delete_selected_manual_roi()
        self._update_roi_status(self.viewer.manual_roi_regions())
        self._refresh_toolbar_state()
        return deleted

    def set_region_direction_lines(self, direction_lines):
        self.viewer.set_region_direction_lines(direction_lines)

    def region_direction_lines(self):
        return self.viewer.region_direction_lines()

    def start_region_direction_drawing(self, region_index: int):
        self.viewer.start_region_direction_drawing(region_index)
        self._refresh_toolbar_state()

    def is_manual_roi_drawing(self) -> bool:
        return self.viewer.is_manual_roi_drawing()
