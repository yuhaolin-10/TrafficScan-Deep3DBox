from pathlib import Path

try:
    from .qt import QtCore, QtWidgets
except Exception:
    try:
        from gui.qt import QtCore, QtWidgets
    except Exception:
        from qt import QtCore, QtWidgets


class RegionRulesPanel(QtWidgets.QFrame):
    rule_toggled = (
        QtCore.pyqtSignal(int, str, bool)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(int, str, bool)
    )
    set_direction_requested = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    clear_direction_requested = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    frame_jump_requested = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_region_index = None
        self.setMinimumWidth(320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background:#0b1220;border-left:1px solid #243041;")

        header = QtWidgets.QFrame()
        header.setStyleSheet("background:#111827;border-bottom:1px solid #243041;")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 14)
        header_layout.setSpacing(4)

        title = QtWidgets.QLabel("信息面板")
        title.setStyleSheet("color:#f8fafc;font-size:16px;font-weight:600;")
        subtitle = QtWidgets.QLabel("查看当前媒体的违规车辆信息，并继续编辑选中区域规则。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#94a3b8;font-size:12px;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        subtitle.setVisible(False)

        self.content = QtWidgets.QWidget()
        self.content.setStyleSheet(
            "QLabel{color:#d1d5db;}"
            "QCheckBox{color:#e5e7eb;font-size:12px;padding:4px 0;}"
            "QCheckBox::indicator{width:16px;height:16px;}"
            "QCheckBox::indicator:unchecked{border:1px solid #475569;background:#0f172a;border-radius:4px;}"
            "QCheckBox::indicator:checked{border:1px solid #2563eb;background:#2563eb;border-radius:4px;}"
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;padding:8px 12px;border-radius:10px;}"
            "QPushButton:hover{background:#273449;}"
            "QPushButton:disabled{color:#64748b;border-color:#1f2937;background:#111827;}"
            "QListWidget{background:#0b1220;border:1px solid #243041;border-radius:12px;padding:6px;outline:0;}"
            "QListWidget::item{border-bottom:1px solid #162135;padding:8px 6px;}"
        )
        body = QtWidgets.QVBoxLayout(self.content)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(12)

        self.media_card = QtWidgets.QFrame()
        self.media_card.setStyleSheet("QFrame{background:#0f172a;border:1px solid #1f2a3a;border-radius:14px;}")
        self.media_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.media_card.setMaximumHeight(260)
        media_root = QtWidgets.QVBoxLayout(self.media_card)
        media_root.setContentsMargins(14, 14, 14, 14)
        media_root.setSpacing(10)
        self.lbl_media_title = QtWidgets.QLabel("当前媒体")
        self.lbl_media_title.setStyleSheet("color:#f8fafc;font-size:15px;font-weight:600;")
        self.lbl_media_summary = QtWidgets.QLabel("请选择左侧媒体。")
        self.lbl_media_summary.setWordWrap(True)
        self.lbl_media_summary.setStyleSheet("color:#94a3b8;font-size:12px;line-height:1.6;")
        self.lbl_media_hint = QtWidgets.QLabel("违规车辆")
        self.lbl_media_hint.setStyleSheet("color:#93c5fd;font-size:12px;font-weight:600;")
        self.media_list = QtWidgets.QListWidget()
        self.media_list.setWordWrap(True)
        self.media_list.setAlternatingRowColors(False)
        self.media_list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.media_list.setMaximumHeight(184)
        self.media_list.itemClicked.connect(self._on_media_item_clicked)
        self.lbl_media_hint.setVisible(False)
        media_root.addWidget(self.lbl_media_title)
        media_root.addWidget(self.lbl_media_summary)
        media_root.addWidget(self.lbl_media_hint)
        media_root.addWidget(self.media_list)

        self.empty_card = QtWidgets.QFrame()
        self.empty_card.setStyleSheet("QFrame{background:#0f172a;border:1px solid #1f2a3a;border-radius:14px;}")
        self.empty_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.empty_card.setMinimumHeight(260)
        empty_root = QtWidgets.QVBoxLayout(self.empty_card)
        empty_root.setContentsMargins(14, 14, 14, 14)
        empty_root.setSpacing(8)
        self.lbl_empty_title = QtWidgets.QLabel("当前还没有区域")
        self.lbl_empty_title.setStyleSheet("color:#f8fafc;font-size:14px;font-weight:600;")
        self.lbl_empty_text = QtWidgets.QLabel("先点击“框选车道”创建区域。")
        self.lbl_empty_text.setWordWrap(True)
        self.lbl_empty_text.setStyleSheet("color:#94a3b8;font-size:12px;line-height:1.6;")
        empty_root.addStretch(1)
        empty_root.addWidget(self.lbl_empty_title)
        empty_root.addWidget(self.lbl_empty_text)
        empty_root.addStretch(1)

        self.region_card = QtWidgets.QFrame()
        self.region_card.setStyleSheet("QFrame{background:#0f172a;border:1px solid #1f2a3a;border-radius:14px;}")
        self.region_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.region_card.setMinimumHeight(340)
        region_root = QtWidgets.QVBoxLayout(self.region_card)
        region_root.setContentsMargins(14, 14, 14, 14)
        region_root.setSpacing(14)

        self.lbl_region_name = QtWidgets.QLabel("区域 1")
        self.lbl_region_name.setStyleSheet("color:#f8fafc;font-size:15px;font-weight:600;")
        self.lbl_region_status = QtWidgets.QLabel("已启用")
        self.lbl_region_status.setStyleSheet("color:#93c5fd;font-size:11px;")
        self.lbl_region_hint = QtWidgets.QLabel("为当前选中区域启用规则后，分析时会直接生效。")
        self.lbl_region_hint.setWordWrap(True)
        self.lbl_region_hint.setStyleSheet("color:#94a3b8;font-size:12px;line-height:1.6;")

        region_root.addWidget(self.lbl_region_name)
        region_root.addWidget(self.lbl_region_status)
        region_root.addWidget(self.lbl_region_hint)
        self.lbl_region_hint.setVisible(False)

        self.chk_no_parking = QtWidgets.QCheckBox("禁止停车")
        self.chk_no_non_motor = QtWidgets.QCheckBox("禁止非机动车")
        self.chk_no_wrong_way = QtWidgets.QCheckBox("禁止逆行")
        region_root.addWidget(self.chk_no_parking)
        region_root.addWidget(self.chk_no_non_motor)
        region_root.addWidget(self.chk_no_wrong_way)

        self.direction_card = QtWidgets.QFrame()
        self.direction_card.setStyleSheet("QFrame{background:#0b1220;border:1px solid #243041;border-radius:12px;}")
        direction_root = QtWidgets.QVBoxLayout(self.direction_card)
        direction_root.setContentsMargins(12, 12, 12, 12)
        direction_root.setSpacing(8)
        self.lbl_direction_title = QtWidgets.QLabel("允许方向")
        self.lbl_direction_title.setStyleSheet("color:#f8fafc;font-size:13px;font-weight:600;")
        self.lbl_direction_status = QtWidgets.QLabel("启用“禁止逆行”后可设置允许方向。")
        self.lbl_direction_status.setWordWrap(True)
        self.lbl_direction_status.setStyleSheet("color:#94a3b8;font-size:12px;line-height:1.6;")
        direction_root.addWidget(self.lbl_direction_title)
        direction_root.addWidget(self.lbl_direction_status)

        direction_actions = QtWidgets.QHBoxLayout()
        direction_actions.setContentsMargins(0, 0, 0, 0)
        direction_actions.setSpacing(8)
        self.btn_set_direction = QtWidgets.QPushButton("设置允许方向")
        self.btn_clear_direction = QtWidgets.QPushButton("清除允许方向")
        direction_actions.addWidget(self.btn_set_direction, 1)
        direction_actions.addWidget(self.btn_clear_direction, 1)
        direction_root.addLayout(direction_actions)

        region_root.addStretch(1)
        region_root.addWidget(self.direction_card)

        body.addWidget(self.media_card, 0)
        body.addWidget(self.empty_card, 1)
        body.addWidget(self.region_card, 1)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{background:#0b1220;border:0px;}")
        self.scroll.setWidget(self.content)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header, 0)
        root.addWidget(self.scroll, 1)

        self.chk_no_parking.toggled.connect(lambda checked: self._emit_rule_toggle("no_parking", checked))
        self.chk_no_non_motor.toggled.connect(lambda checked: self._emit_rule_toggle("no_non_motor", checked))
        self.chk_no_wrong_way.toggled.connect(lambda checked: self._emit_rule_toggle("no_wrong_way", checked))
        self.btn_set_direction.clicked.connect(self._emit_set_direction)
        self.btn_clear_direction.clicked.connect(self._emit_clear_direction)

        self.show_no_media_selected()

    def _emit_rule_toggle(self, rule_type: str, enabled: bool):
        if self._current_region_index is None:
            return
        self.rule_toggled.emit(int(self._current_region_index), str(rule_type), bool(enabled))

    def _emit_set_direction(self):
        if self._current_region_index is None:
            return
        self.set_direction_requested.emit(int(self._current_region_index))

    def _emit_clear_direction(self):
        if self._current_region_index is None:
            return
        self.clear_direction_requested.emit(int(self._current_region_index))

    def _on_media_item_clicked(self, item):
        if item is None:
            return
        frame_index = item.data(QtCore.Qt.UserRole)
        if frame_index is None:
            return
        try:
            self.frame_jump_requested.emit(int(frame_index))
        except Exception:
            return

    def _set_media_visible(self, visible: bool):
        self.media_card.setVisible(bool(visible))

    def _set_region_mode(self, *, empty_title: str = "", empty_text: str = "", show_region: bool = False):
        self.empty_card.setVisible(not show_region)
        self.region_card.setVisible(show_region)
        self._refresh_panel_density(show_region=show_region)
        if not show_region:
            self.lbl_empty_title.setText(empty_title)
            self.lbl_empty_text.setText(empty_text)

    def _refresh_panel_density(self, *, show_region: bool):
        self.lbl_media_summary.setVisible(not show_region)
        self.media_list.setMaximumHeight(160 if show_region else 220)

    def _add_media_entry(self, text: str, *, frame_index=None):
        item = QtWidgets.QListWidgetItem(str(text))
        item.setData(QtCore.Qt.UserRole, frame_index)
        if frame_index is not None:
            item.setToolTip(f"点击跳转到第 {int(frame_index) + 1} 帧")
        self.media_list.addItem(item)

    def _set_media_entries(self, title: str, summary: str, entries: list[dict], *, clickable: bool = False):
        self._set_media_visible(True)
        entry_count = len(entries or [])
        self.lbl_media_title.setText(f"{title} ({entry_count})" if entry_count > 0 else title)
        self.lbl_media_summary.setText(summary)
        self.media_list.clear()
        self.lbl_media_hint.setText(f"违规车辆 ({len(entries)})" if entries else "违规车辆")
        if not entries:
            self._add_media_entry("当前未识别到可展示的违规车辆信息。")
            return
        for entry in entries:
            frame_index = entry.get("frame_index") if clickable else None
            self._add_media_entry(str(entry.get("text", "")), frame_index=frame_index)

    def _rule_label(self, rule_type: str) -> str:
        mapping = {
            "no_parking": "禁止停车",
            "no_non_motor": "禁止非机动车",
            "no_wrong_way": "禁止逆行",
        }
        return mapping.get(str(rule_type or "").strip().lower(), str(rule_type or "未知规则"))

    def _violation_label(self, text: str, *, rule_type: str = "") -> str:
        raw = str(text or "").strip()
        if "," in raw:
            parts = [self._violation_label(part, rule_type=rule_type) for part in raw.split(",")]
            parts = [part for part in parts if str(part).strip()]
            if parts:
                return "、".join(parts)
        normalized = raw.lower().replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())
        mapping = {
            "no parking": "禁止停车",
            "parking": "禁止停车",
            "no non motor": "禁止非机动车",
            "no non-motor": "禁止非机动车",
            "non motor": "禁止非机动车",
            "non-motor": "禁止非机动车",
            "wrong way": "禁止逆行",
            "no wrong way": "禁止逆行",
            "emergency lane occupation": "占用应急车道",
            "emergency lane": "占用应急车道",
            "emergency_lane": "占用应急车道",
        }
        translated = mapping.get(normalized)
        if translated:
            return translated
        if raw:
            return raw
        return self._rule_label(rule_type) if rule_type else "违规"

    def show_media_overview(self, file_path: str, result: dict | None):
        path = Path(file_path)
        if not result:
            self._set_media_entries(
                path.name,
                "当前素材还没有分析结果，先勾选后点击“开始分析”即可。",
                [],
            )
            return

        media_type = str(result.get("media_type", "image") or "image").lower()
        if media_type == "video":
            events_by_track = {}
            for event in list(result.get("region_rule_events", []) or []):
                track_id = str(event.get("track_id", "") or "").strip() or f"track-{len(events_by_track) + 1}"
                bucket = events_by_track.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "vehicle_type": str(event.get("vehicle_type", "vehicle") or "vehicle"),
                        "rule_labels": [],
                        "first_frame_index": event.get("frame_index"),
                    },
                )
                bucket["vehicle_type"] = str(event.get("vehicle_type", bucket["vehicle_type"]) or bucket["vehicle_type"])
                frame_index = event.get("frame_index")
                if frame_index is not None:
                    current_min = bucket.get("first_frame_index")
                    if current_min is None or int(frame_index) < int(current_min):
                        bucket["first_frame_index"] = int(frame_index)
                rule_label = self._violation_label(
                    event.get("rule_label", ""),
                    rule_type=event.get("rule_type", ""),
                )
                if rule_label not in bucket["rule_labels"]:
                    bucket["rule_labels"].append(rule_label)

            for plate_item in list(result.get("violating_track_plates", []) or []):
                track_id = str(plate_item.get("track_id", "") or "").strip()
                if not track_id:
                    continue
                bucket = events_by_track.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "vehicle_type": str(plate_item.get("vehicle_type", "vehicle") or "vehicle"),
                        "rule_labels": [],
                        "first_frame_index": plate_item.get("first_violation_frame"),
                    },
                )
                bucket["plate_text"] = str(plate_item.get("plate_text", "") or "").strip()
                bucket["plate_confidence"] = float(plate_item.get("plate_confidence", 0.0) or 0.0)
                bucket["plate_support_count"] = int(plate_item.get("plate_support_count", 0) or 0)
                if bucket.get("first_frame_index") is None and plate_item.get("first_violation_frame") is not None:
                    bucket["first_frame_index"] = int(plate_item.get("first_violation_frame"))

            entries = []
            for item in events_by_track.values():
                plate_text = str(item.get("plate_text", "") or "").strip() or "未识别到车牌"
                rule_text = "、".join(item.get("rule_labels", [])) or "违规"
                frame_index = item.get("first_frame_index")
                frame_text = f"第 {int(frame_index) + 1} 帧" if frame_index is not None else "帧未知"
                plate_suffix = ""
                if plate_text != "未识别到车牌":
                    plate_suffix = (
                        f" | 车牌 {plate_text}"
                        f" | 置信度 {float(item.get('plate_confidence', 0.0) or 0.0):.2f}"
                    )
                entries.append(
                    {
                        "frame_index": frame_index,
                        "text": (
                            f"{rule_text} | 识别类型 {str(item.get('vehicle_type', 'vehicle'))}"
                            f" | {frame_text}{plate_suffix if plate_suffix else ' | 未识别到车牌'}"
                        ),
                    }
                )

            entries.sort(key=lambda item: int(item.get("frame_index", 10**9) if item.get("frame_index") is not None else 10**9))
            processed_frames = int(result.get("processed_frame_count", 0) or 0)
            frame_count = int(result.get("frame_count", 0) or 0)
            summary = (
                f"视频已分析：{processed_frames}/{frame_count} 帧，"
                f"累计违规事件 {int(result.get('region_rule_event_count', 0) or 0)} 条。"
                f" 点击下方条目可跳转到对应帧。"
            )
            self._set_media_entries(f"{path.name} · 视频", summary, entries, clickable=True)
            return

        entries = []
        for detection in list(result.get("detections", []) or []):
            if not bool(detection.get("is_violating", False)):
                continue
            plate_text = str(detection.get("plate_text", "") or "").strip() or "未识别到车牌"
            entries.append(
                {
                    "text": (
                        f"{self._violation_label(detection.get('violation_type', ''))} | "
                        f"识别类型 {str(detection.get('vehicle_type', 'vehicle'))} | "
                        f"{plate_text}"
                    )
                }
            )
        summary = (
            f"图片共检测到 {int(result.get('vehicle_count', 0) or 0)} 辆车，"
            f"其中违规 {int(result.get('violation_count', 0) or 0)} 辆。"
        )
        self._set_media_entries(f"{path.name} · 图片", summary, entries, clickable=False)

    def show_no_media_selected(self):
        self._current_region_index = None
        self._set_media_visible(False)
        self.media_list.clear()
        self._set_region_mode(
            empty_title="先从左侧选择一个素材",
            empty_text="选择图片或视频后，这里会显示当前媒体的违规车辆信息和区域规则。",
            show_region=False,
        )

    def show_no_regions(self):
        self._current_region_index = None
        self._set_region_mode(
            empty_title="当前还没有区域",
            empty_text="先点击“框选车道”创建区域，再为该区域启用规则。",
            show_region=False,
        )

    def show_region_selection_hint(self, region_count: int):
        self._current_region_index = None
        self._set_region_mode(
            empty_title="请选择一个区域",
            empty_text=f"当前共有 {int(region_count)} 个区域。点击中间画面中的某个区域后，这里会显示该区域的规则。",
            show_region=False,
        )

    def show_region_rules(self, entry: dict, region_index: int):
        self._current_region_index = int(region_index)
        self._set_region_mode(show_region=True)

        region_name = str(entry.get("name", "") or "").strip() or f"区域 {int(region_index) + 1}"
        enabled = bool(entry.get("enabled", True))
        direction_line = list(entry.get("direction_line", []) or [])
        rule_types = {
            str(binding.get("rule_type", "") or "").strip().lower(): bool(binding.get("enabled", True))
            for binding in list(entry.get("rule_bindings", []) or [])
            if str(binding.get("rule_type", "") or "").strip()
        }

        self.lbl_region_name.setText(region_name)
        self.lbl_region_status.setText("已启用" if enabled else "未启用")

        self.chk_no_parking.blockSignals(True)
        self.chk_no_non_motor.blockSignals(True)
        self.chk_no_wrong_way.blockSignals(True)
        try:
            self.chk_no_parking.setChecked(bool(rule_types.get("no_parking", False)))
            self.chk_no_non_motor.setChecked(bool(rule_types.get("no_non_motor", False)))
            self.chk_no_wrong_way.setChecked(bool(rule_types.get("no_wrong_way", False)))
        finally:
            self.chk_no_parking.blockSignals(False)
            self.chk_no_non_motor.blockSignals(False)
            self.chk_no_wrong_way.blockSignals(False)

        wrong_way_enabled = bool(rule_types.get("no_wrong_way", False))
        has_direction = len(direction_line) == 2
        self.direction_card.setVisible(wrong_way_enabled or has_direction)
        if wrong_way_enabled:
            if has_direction:
                self.lbl_direction_status.setText("允许方向已设置，分析时会按该方向判断逆行。")
            else:
                self.lbl_direction_status.setText("允许方向尚未设置，请点击下方按钮完成设置。")
        else:
            self.lbl_direction_status.setText("启用“禁止逆行”后，可在这里设置允许方向。")
        self.btn_set_direction.setEnabled(wrong_way_enabled)
        self.btn_clear_direction.setEnabled(wrong_way_enabled and has_direction)
