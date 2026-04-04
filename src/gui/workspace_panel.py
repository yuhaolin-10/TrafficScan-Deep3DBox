from pathlib import Path

try:
    from .qt import QtCore, QtGui, QtWidgets
except Exception:
    try:
        from gui.qt import QtCore, QtGui, QtWidgets
    except Exception:
        from qt import QtCore, QtGui, QtWidgets

try:
    from ..services.video_reader import IMAGE_EXTS, VIDEO_EXTS, read_preview_frame, read_video_frame
except Exception:
    try:
        from services.video_reader import IMAGE_EXTS, VIDEO_EXTS, read_preview_frame, read_video_frame
    except Exception:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
        read_preview_frame = None
        read_video_frame = None


class DropArea(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    files_dropped = QtCore.pyqtSignal(list) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._active = False
        self._apply_style()

        icon = QtWidgets.QLabel("导入")
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setStyleSheet("color:#60a5fa;font-size:20px;font-weight:700;")

        title = QtWidgets.QLabel("导入图片或视频")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color:#e2e8f0;font-size:15px;font-weight:600;")

        text = QtWidgets.QLabel("拖拽到这里，或点击导入文件 / 文件夹")
        text.setAlignment(QtCore.Qt.AlignCenter)
        text.setWordWrap(True)
        text.setStyleSheet("color:#94a3b8;font-size:12px;")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(6)
        root.addStretch(1)
        root.addWidget(icon)
        root.addWidget(title)
        root.addWidget(text)
        root.addStretch(1)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                "QFrame{border:2px dashed #3b82f6;border-radius:14px;background:rgba(59,130,246,0.08);}"
                "QLabel{border:0px;background:transparent;}"
            )
        else:
            self.setStyleSheet(
                "QFrame{border:2px dashed #334155;border-radius:14px;background:#0f172a;}"
                "QLabel{border:0px;background:transparent;}"
            )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._active = True
            self._apply_style()
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self._active = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._active = False
        self._apply_style()
        if not event.mimeData().hasUrls():
            return
        paths = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if local_path:
                paths.append(local_path)
        self.files_dropped.emit(paths)
        event.acceptProposedAction()


class WorkspaceListWidget(QtWidgets.QListWidget):
    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _is_background_left_click(self, event) -> bool:
        if event is None or event.button() != QtCore.Qt.LeftButton:
            return False
        return self.itemAt(self._event_pos(event)) is None

    def mousePressEvent(self, event):
        if self._is_background_left_click(event):
            current = self.currentItem()
            if current is not None:
                current.setSelected(True)
                self.setCurrentItem(current)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_background_left_click(event):
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WorkspaceItemWidget(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceItemCard")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._selected = False
        self._status = ""
        self._checked = True

        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setCursor(QtCore.Qt.PointingHandCursor)
        self.checkbox.setFixedSize(18, 18)
        self.checkbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.checkbox.setStyleSheet(
            "QCheckBox{padding:0;margin:0;background:transparent;border:none;}"
            "QCheckBox::indicator{width:16px;height:16px;}"
            "QCheckBox::indicator:unchecked{border:1px solid #475569;background:#0f172a;border-radius:4px;}"
            "QCheckBox::indicator:checked{border:1px solid #2563eb;background:#2563eb;border-radius:4px;}"
        )

        self.thumb = QtWidgets.QLabel()
        self.thumb.setFixedSize(76, 56)
        self.thumb.setAlignment(QtCore.Qt.AlignCenter)
        self.thumb.setStyleSheet(
            "background:#111827;border:1px solid #1f2937;border-radius:10px;"
            "color:#94a3b8;font-size:11px;font-weight:600;"
        )

        self.name = QtWidgets.QLabel("-")
        self.name.setWordWrap(True)
        self.name.setStyleSheet("background:transparent;border:none;color:#e5e7eb;font-size:13px;font-weight:600;")

        self.meta = QtWidgets.QLabel("-")
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet("background:transparent;border:none;color:#94a3b8;font-size:11px;")

        self.summary = QtWidgets.QLabel("-")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("background:transparent;border:none;color:#cbd5e1;font-size:11px;")

        text_panel = QtWidgets.QWidget()
        text_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        text_panel.setStyleSheet("background:transparent;border:none;")

        text_layout = QtWidgets.QVBoxLayout(text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)
        text_layout.addWidget(self.name)
        text_layout.addWidget(self.meta)
        text_layout.addWidget(self.summary)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self.checkbox, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        root.addWidget(self.thumb, 0, QtCore.Qt.AlignTop)
        root.addWidget(text_panel, 1)
        self._apply_style()

    def update_content(self, *, file_path: str, media_type: str, status: str, summary: str, selected: bool, checked: bool):
        path = Path(file_path)
        is_video = str(media_type or "").strip().lower() == "video"
        self.name.setText(path.name)
        self.meta.setText(f"{self._media_type_text(media_type)} · {self._status_text(status)}")
        self.summary.setText(summary or self._default_summary(status))
        self._status = str(status or "")
        self._selected = bool(selected)
        self._checked = bool(checked)
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(self._checked)
        self.checkbox.blockSignals(False)
        self.meta.setVisible(not is_video)
        self.summary.setVisible(not is_video)
        if is_video:
            self.meta.clear()
            self.summary.clear()
        self._set_thumbnail(path, media_type)
        self._apply_style()

    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_style()

    def _apply_style(self):
        border = "#3b82f6" if self._selected else "#1f2a3a"
        background = "#162135" if self._selected else "#0f172a"
        self.setStyleSheet(
            f"#workspaceItemCard{{background:{background};border:1px solid {border};border-radius:14px;}}"
        )
        self.name.setStyleSheet(
            "background:transparent;border:none;color:#e5e7eb;font-size:13px;font-weight:600;"
            if self._checked
            else "background:transparent;border:none;color:#94a3b8;font-size:13px;font-weight:600;"
        )
        self.meta.setStyleSheet("background:transparent;border:none;color:#94a3b8;font-size:11px;")
        if self._selected:
            summary_style = "background:transparent;border:none;color:#eff6ff;font-size:11px;"
        elif self._status == WorkspacePanel.STATUS_FAILED:
            summary_style = "background:transparent;border:none;color:#fca5a5;font-size:11px;"
        elif self._status == WorkspacePanel.STATUS_DONE:
            summary_style = "background:transparent;border:none;color:#bfdbfe;font-size:11px;"
        else:
            summary_style = "background:transparent;border:none;color:#cbd5e1;font-size:11px;"
        self.summary.setStyleSheet(summary_style)

    def _frame_to_pixmap(self, frame):
        if frame is None:
            return QtGui.QPixmap()
        try:
            if len(frame.shape) == 2:
                h, w = frame.shape[:2]
                image = QtGui.QImage(
                    frame.data,
                    w,
                    h,
                    int(frame.strides[0]),
                    QtGui.QImage.Format_Grayscale8,
                ).copy()
            else:
                h, w = frame.shape[:2]
                channels = int(frame.shape[2])
                if channels == 4:
                    rgba = frame[:, :, [2, 1, 0, 3]].copy()
                    image = QtGui.QImage(
                        rgba.data,
                        w,
                        h,
                        int(rgba.strides[0]),
                        QtGui.QImage.Format_RGBA8888,
                    ).copy()
                else:
                    rgb = frame[:, :, ::-1].copy()
                    image = QtGui.QImage(
                        rgb.data,
                        w,
                        h,
                        int(rgb.strides[0]),
                        QtGui.QImage.Format_RGB888,
                    ).copy()
            return QtGui.QPixmap.fromImage(image)
        except Exception:
            return QtGui.QPixmap()

    def _load_video_thumbnail(self, path: Path):
        loaders = []
        if callable(read_preview_frame):
            loaders.append(lambda: read_preview_frame(str(path))[1])
        if callable(read_video_frame):
            loaders.append(lambda: read_video_frame(str(path), frame_index=0))

        for loader in loaders:
            try:
                frame = loader()
            except Exception:
                continue
            pixmap = self._frame_to_pixmap(frame)
            if not pixmap.isNull():
                return pixmap
        return QtGui.QPixmap()

    def _placeholder_pixmap(self, media_type: str):
        pixmap = QtGui.QPixmap(self.thumb.size())
        pixmap.fill(QtGui.QColor("#111827"))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(pixmap.rect(), QtGui.QColor("#111827"))
        text_rect = pixmap.rect().adjusted(8, 8, -8, -8)
        painter.setPen(QtGui.QColor("#60a5fa"))
        painter.setFont(QtGui.QFont("Microsoft YaHei UI", 9, QtGui.QFont.Bold))
        painter.drawText(
            text_rect,
            int(QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap),
            self._thumbnail_text(media_type),
        )
        painter.end()
        return pixmap

    def _set_thumbnail(self, path: Path, media_type: str):
        pixmap = QtGui.QPixmap()
        if media_type == "image":
            loaded = QtGui.QPixmap(str(path))
            if not loaded.isNull():
                pixmap = loaded
        elif media_type == "video":
            pixmap = self._load_video_thumbnail(path)

        if pixmap.isNull():
            pixmap = self._placeholder_pixmap(media_type)
        else:
            pixmap = pixmap.scaled(
                self.thumb.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )
        self.thumb.setPixmap(pixmap)

    def _thumbnail_text(self, media_type: str) -> str:
        if media_type == "video":
            return "视频"
        if media_type == "image":
            return "图片"
        return "文件"

    def _media_type_text(self, media_type: str) -> str:
        if media_type == "video":
            return "视频"
        if media_type == "image":
            return "图片"
        return "文件"

    def _status_text(self, status: str) -> str:
        return {
            WorkspacePanel.STATUS_PENDING: "待分析",
            WorkspacePanel.STATUS_RUNNING: "分析中",
            WorkspacePanel.STATUS_DONE: "已完成",
            WorkspacePanel.STATUS_FAILED: "失败",
            WorkspacePanel.STATUS_CANCELLED: "已停止",
        }.get(str(status or ""), "未知")

    def _default_summary(self, status: str) -> str:
        return {
            WorkspacePanel.STATUS_PENDING: "未分析",
            WorkspacePanel.STATUS_RUNNING: "正在分析当前素材",
            WorkspacePanel.STATUS_DONE: "分析已完成",
            WorkspacePanel.STATUS_FAILED: "分析失败，请重试",
            WorkspacePanel.STATUS_CANCELLED: "分析已停止",
        }.get(str(status or ""), "未分析")


class WorkspacePanel(QtWidgets.QFrame):
    file_selected = QtCore.pyqtSignal(str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str)
    files_added = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    unsupported_dropped = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    content_changed = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    selection_cleared = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    IMAGE_EXTS = set(IMAGE_EXTS)
    VIDEO_EXTS = set(VIDEO_EXTS)
    SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS
    SUMMARY_ROLE = QtCore.Qt.UserRole + 4
    CHECKED_ROLE = QtCore.Qt.UserRole + 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#111827;border-right:1px solid #243041;")
        self.setMinimumWidth(250)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        title = QtWidgets.QLabel("工作区")
        title.setStyleSheet("color:#e5e7eb;font-size:16px;font-weight:600;")

        subtitle = QtWidgets.QLabel("从这里导入并切换当前素材")
        subtitle.setStyleSheet("color:#94a3b8;font-size:12px;")

        self.drop_area = DropArea()
        self.drop_area.setFixedHeight(138)

        self.list = WorkspaceListWidget()
        self.list.setStyleSheet(
            "QListWidget{background:#0b1220;border:1px solid #243041;border-radius:14px;padding:8px;outline:0;}"
            "QListWidget::item{padding:0px;margin:0 0 8px 0;border:0px;background:transparent;}"
            "QListWidget::item:selected{background:transparent;}"
        )
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list.setSpacing(6)

        self.btn_select_all = QtWidgets.QPushButton("全选")
        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_remove = QtWidgets.QPushButton("移除")
        button_style = (
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;"
            "padding:9px 12px;border-radius:10px;font-size:12px;}"
            "QPushButton:hover{background:#273449;}"
            "QPushButton:disabled{color:#64748b;border-color:#1f2937;background:#111827;}"
        )
        self.btn_select_all.setStyleSheet(button_style)
        self.btn_clear.setStyleSheet(button_style)
        self.btn_remove.setStyleSheet(button_style)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        root.addWidget(title, 0)
        root.addWidget(subtitle, 0)
        root.addWidget(self.drop_area, 0)
        root.addWidget(self.list, 1)

        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addWidget(self.btn_select_all, 1)
        actions.addWidget(self.btn_clear, 1)
        actions.addWidget(self.btn_remove, 1)
        root.addLayout(actions, 0)

        self.drop_area.clicked.connect(self._open_import_menu)
        self.drop_area.files_dropped.connect(self._on_drop_paths)
        self.btn_select_all.clicked.connect(self.toggle_select_all)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.list.currentItemChanged.connect(self._emit_current)
        self.list.itemSelectionChanged.connect(self._refresh_selection_state)
        self._refresh_action_state()

    def _open_import_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#0f172a;border:1px solid #334155;min-width:220px;}"
            "QMenu::item{padding:8px 18px;color:#dbe4ee;}"
            "QMenu::item:selected{background:#1e293b;}"
        )
        act_files = menu.addAction("导入文件...")
        act_folder = menu.addAction("导入文件夹...")

        exec_menu = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
        chosen = exec_menu(QtGui.QCursor.pos())
        if chosen is None:
            return

        if chosen == act_files:
            filter_text = (
                "Media files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;"
                "Images (*.png *.jpg *.jpeg *.bmp);;"
                "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;"
                "All files (*)"
            )
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "导入素材", "", filter_text)
            if not paths:
                return
            added = self.add_paths(paths)
            if added == 0:
                self.unsupported_dropped.emit()
            return

        if chosen == act_folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "导入文件夹", "")
            if not folder:
                return
            added = self.add_paths([folder])
            if added == 0:
                self.unsupported_dropped.emit()

    def _on_drop_paths(self, paths: list):
        added = self.add_paths(paths)
        if added == 0:
            self.unsupported_dropped.emit()

    def selected_path(self):
        items = self.list.selectedItems()
        if not items:
            return None
        return items[0].data(QtCore.Qt.UserRole)

    def selected_paths(self) -> list:
        paths = []
        for item in self.list.selectedItems():
            path = item.data(QtCore.Qt.UserRole)
            if path:
                paths.append(path)
        return paths

    def checked_paths(self) -> list:
        paths = []
        for index in range(self.list.count()):
            item = self.list.item(index)
            if not bool(item.data(self.CHECKED_ROLE)):
                continue
            path = item.data(QtCore.Qt.UserRole)
            if path:
                paths.append(path)
        return paths

    def all_checked(self) -> bool:
        return self.list.count() > 0 and len(self.checked_paths()) == self.list.count()

    def all_paths(self) -> list:
        paths = []
        for index in range(self.list.count()):
            item = self.list.item(index)
            path = item.data(QtCore.Qt.UserRole)
            if path:
                paths.append(path)
        return paths

    def media_type_of_path(self, path: str):
        item = self._find_item_by_path(path)
        if item is None:
            return None
        return str(item.data(QtCore.Qt.UserRole + 3) or "")

    def status_of_path(self, path: str):
        item = self._find_item_by_path(path)
        if item is None:
            return None
        return item.data(QtCore.Qt.UserRole + 1)

    def failure_reason_of_path(self, path: str):
        item = self._find_item_by_path(path)
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole + 2) or "")

    def result_summary_of_path(self, path: str) -> str:
        item = self._find_item_by_path(path)
        if item is None:
            return ""
        return str(item.data(self.SUMMARY_ROLE) or "")

    def set_result_summary_for_path(self, path: str, summary: str):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(self.SUMMARY_ROLE, str(summary or ""))
        self._refresh_item_widget(item)
        self._update_item_tooltip(item)

    def clear_result_summary_for_path(self, path: str):
        self.set_result_summary_for_path(path, "")

    def set_checked_for_path(self, path: str, checked: bool):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(self.CHECKED_ROLE, bool(checked))
        self._refresh_item_widget(item)
        self._update_item_tooltip(item)
        self._refresh_action_state()

    def toggle_select_all(self):
        should_check = not self.all_checked()
        for index in range(self.list.count()):
            item = self.list.item(index)
            item.setData(self.CHECKED_ROLE, bool(should_check))
            self._refresh_item_widget(item)
            self._update_item_tooltip(item)
        self._refresh_action_state()
        self.content_changed.emit(self.list.count())

    def set_status_for_paths(self, paths: list, status: str):
        for path in paths:
            self.set_status_for_path(path, status)

    def set_status_for_path(self, path: str, status: str):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(QtCore.Qt.UserRole + 1, status)
        if status == self.STATUS_PENDING and not item.data(self.SUMMARY_ROLE):
            item.setData(self.SUMMARY_ROLE, "")
        if status not in {self.STATUS_FAILED, self.STATUS_CANCELLED}:
            item.setData(QtCore.Qt.UserRole + 2, "")
        self._refresh_item_widget(item)
        self._update_item_tooltip(item)

    def set_failure_reason_for_path(self, path: str, reason: str):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(QtCore.Qt.UserRole + 2, str(reason or ""))
        self._refresh_item_widget(item)
        self._update_item_tooltip(item)

    def clear_all(self):
        self.list.clear()
        self._refresh_action_state()
        self.content_changed.emit(0)
        self.selection_cleared.emit()

    def remove_selected(self):
        rows = sorted({self.list.row(item) for item in self.list.selectedItems()}, reverse=True)
        if not rows:
            return
        next_row = min(rows[-1], max(0, self.list.count() - len(rows) - 1))
        for row in rows:
            self.list.takeItem(row)
        if self.list.count() > 0:
            self.list.setCurrentRow(next_row)
        else:
            self.selection_cleared.emit()
        self._refresh_selection_state()
        self.content_changed.emit(self.list.count())

    def add_paths(self, paths: list) -> int:
        existing = set()
        for index in range(self.list.count()):
            item = self.list.item(index)
            current_path = item.data(QtCore.Qt.UserRole)
            if current_path:
                existing.add(str(Path(current_path).resolve(strict=False)))

        expanded = []
        for raw in paths:
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_dir():
                for child in path.iterdir():
                    if child.is_file() and child.suffix.lower() in self.SUPPORTED_EXTS:
                        expanded.append(str(child.resolve(strict=False)))
            elif path.suffix.lower() in self.SUPPORTED_EXTS:
                expanded.append(str(path.resolve(strict=False)))

        added = 0
        for path in expanded:
            normalized = str(Path(path).resolve(strict=False))
            if normalized in existing:
                continue
            media_path = Path(path).resolve(strict=False)
            media_type = self._media_type_from_suffix(media_path.suffix)
            status = self.STATUS_PENDING
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, str(media_path))
            item.setData(QtCore.Qt.UserRole + 1, status)
            item.setData(QtCore.Qt.UserRole + 2, "")
            item.setData(QtCore.Qt.UserRole + 3, media_type)
            item.setData(self.SUMMARY_ROLE, "")
            item.setData(self.CHECKED_ROLE, True)
            item.setSizeHint(QtCore.QSize(0, 88))
            self.list.addItem(item)
            widget = WorkspaceItemWidget(self.list)
            self.list.setItemWidget(item, widget)
            widget.checkbox.toggled.connect(lambda checked, current_item=item: self._on_item_checked(current_item, checked))
            self._refresh_item_widget(item)
            self._update_item_tooltip(item)
            existing.add(normalized)
            added += 1

        if self.list.count() and not self.list.selectedItems():
            self.list.setCurrentRow(0)

        self._refresh_selection_state()
        self.content_changed.emit(self.list.count())
        if added:
            self.files_added.emit(added)
        return added

    def _emit_current(self, current, _previous):
        if current is None:
            if self.list.count() == 0:
                self.selection_cleared.emit()
            return
        path = current.data(QtCore.Qt.UserRole)
        if path:
            self.file_selected.emit(path)

    def _refresh_selection_state(self):
        for index in range(self.list.count()):
            item = self.list.item(index)
            self._refresh_item_widget(item)
        self._refresh_action_state()

    def _refresh_action_state(self):
        has_items = self.list.count() > 0
        has_selection = len(self.list.selectedItems()) > 0
        checked_count = len(self.checked_paths())
        self.btn_select_all.setEnabled(has_items)
        self.btn_select_all.setText("取消全选" if has_items and checked_count == self.list.count() else "全选")
        self.btn_clear.setEnabled(has_items)
        self.btn_remove.setEnabled(has_selection)

    def _refresh_item_widget(self, item):
        widget = self.list.itemWidget(item)
        if widget is None:
            return
        path = str(item.data(QtCore.Qt.UserRole) or "")
        media_type = str(item.data(QtCore.Qt.UserRole + 3) or "file")
        status = str(item.data(QtCore.Qt.UserRole + 1) or "")
        summary = str(item.data(self.SUMMARY_ROLE) or "")
        checked = bool(item.data(self.CHECKED_ROLE))
        widget.update_content(
            file_path=path,
            media_type=media_type,
            status=status,
            summary=summary,
            selected=item.isSelected(),
            checked=checked,
        )

    def _on_item_checked(self, item, checked: bool):
        if item is None:
            return
        item.setData(self.CHECKED_ROLE, bool(checked))
        self._refresh_item_widget(item)
        self._update_item_tooltip(item)
        self._refresh_action_state()
        self.content_changed.emit(self.list.count())

    def _find_item_by_path(self, path: str):
        normalized = str(Path(path).resolve(strict=False))
        for index in range(self.list.count()):
            item = self.list.item(index)
            current_path = item.data(QtCore.Qt.UserRole)
            if current_path and str(Path(current_path).resolve(strict=False)) == normalized:
                return item
        return None

    def _update_item_tooltip(self, item):
        path = str(item.data(QtCore.Qt.UserRole) or "")
        status = str(item.data(QtCore.Qt.UserRole + 1) or "")
        reason = str(item.data(QtCore.Qt.UserRole + 2) or "").strip()
        media_type = str(item.data(QtCore.Qt.UserRole + 3) or "file")
        summary = str(item.data(self.SUMMARY_ROLE) or "").strip()
        checked = bool(item.data(self.CHECKED_ROLE))

        lines = [
            f"文件：{Path(path).name if path else '-'}",
            f"类型：{self._media_type_label(media_type)}",
            f"状态：{self._status_label(status)}",
        ]
        lines.append(f"参与分析：{'是' if checked else '否'}")
        if summary:
            lines.append(f"结果：{summary}")
        if path:
            lines.append(f"路径：{path}")
        if reason:
            lines.extend(["", "失败原因：", reason])
        item.setToolTip("\n".join(lines))

    def _media_type_from_suffix(self, suffix: str) -> str:
        suffix = str(suffix or "").lower()
        if suffix in self.VIDEO_EXTS:
            return "video"
        if suffix in self.IMAGE_EXTS:
            return "image"
        return "file"

    def _media_type_label(self, media_type: str) -> str:
        return {"video": "视频", "image": "图片", "file": "文件"}.get(str(media_type or ""), "文件")

    def _status_label(self, status: str) -> str:
        return {
            self.STATUS_PENDING: "待分析",
            self.STATUS_RUNNING: "分析中",
            self.STATUS_DONE: "已完成",
            self.STATUS_FAILED: "失败",
            self.STATUS_CANCELLED: "已停止",
        }.get(str(status or ""), "未知")
