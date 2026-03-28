from pathlib import Path

try:
    from .qt import QtCore, QtGui, QtWidgets
except Exception:
    try:
        from gui.qt import QtCore, QtGui, QtWidgets
    except Exception:
        from qt import QtCore, QtGui, QtWidgets

try:
    from ..services.video_reader import IMAGE_EXTS, VIDEO_EXTS
except Exception:
    try:
        from services.video_reader import IMAGE_EXTS, VIDEO_EXTS
    except Exception:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


class DropArea(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    files_dropped = QtCore.pyqtSignal(list) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._active = False
        self._apply_style()

        icon = QtWidgets.QLabel("MEDIA")
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setStyleSheet("color:#60a5fa;font-size:20px;font-weight:700;")

        title = QtWidgets.QLabel("Drag images or videos here")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color:#e2e8f0;font-size:14px;font-weight:600;")

        text = QtWidgets.QLabel("Or click here to import files or a folder")
        text.setAlignment(QtCore.Qt.AlignCenter)
        text.setStyleSheet("color:#94a3b8;font-size:12px;")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
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
                "QFrame{border:2px dashed #3b82f6;border-radius:12px;background:rgba(59,130,246,0.08);}"
                "QLabel{border:0px;background:transparent;}"
            )
        else:
            self.setStyleSheet(
                "QFrame{border:2px dashed #334155;border-radius:12px;background:#0f172a;}"
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


class WorkspacePanel(QtWidgets.QFrame):
    file_selected = QtCore.pyqtSignal(str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str)
    files_added = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    unsupported_dropped = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    IMAGE_EXTS = set(IMAGE_EXTS)
    VIDEO_EXTS = set(VIDEO_EXTS)
    SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#111827;border-right:1px solid #243041;")
        self.setMinimumWidth(220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        title = QtWidgets.QLabel("Workspace")
        title.setStyleSheet("color:#e5e7eb;font-size:16px;font-weight:600;")

        subtitle = QtWidgets.QLabel("Manage images and videos")
        subtitle.setStyleSheet("color:#94a3b8;font-size:12px;")

        self.drop_area = DropArea()
        self.drop_area.setFixedHeight(130)

        self.list = QtWidgets.QListWidget()
        self.list.setStyleSheet(
            "QListWidget{background:#0b1220;border:1px solid #243041;border-radius:12px;padding:6px;}"
            "QListWidget::item{padding:8px 10px;border-radius:8px;color:#dbe4ee;}"
            "QListWidget::item:selected{background:#1d4ed8;color:#ffffff;}"
        )
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        button_style = (
            "QPushButton{background:#1f2937;border:1px solid #334155;color:#e5e7eb;"
            "padding:8px 12px;border-radius:8px;}"
            "QPushButton:hover{background:#273449;}"
        )
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
        actions.addWidget(self.btn_clear, 1)
        actions.addWidget(self.btn_remove, 1)
        root.addLayout(actions, 0)

        self.drop_area.clicked.connect(self._open_import_menu)
        self.drop_area.files_dropped.connect(self._on_drop_paths)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.list.currentItemChanged.connect(self._emit_current)

    def _open_import_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#0f172a;border:1px solid #334155;min-width:220px;}"
            "QMenu::item{padding:8px 18px;color:#dbe4ee;}"
            "QMenu::item:selected{background:#1e293b;}"
        )
        act_files = menu.addAction("Import media files...")
        act_folder = menu.addAction("Import folder...")

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
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Import media", "", filter_text)
            if not paths:
                return
            added = self.add_paths(paths)
            if added == 0:
                self.unsupported_dropped.emit()
            return

        if chosen == act_folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Import folder", "")
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

    def set_status_for_paths(self, paths: list, status: str):
        for path in paths:
            self.set_status_for_path(path, status)

    def set_status_for_path(self, path: str, status: str):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(QtCore.Qt.UserRole + 1, status)
        name = Path(path).name
        media_type = str(item.data(QtCore.Qt.UserRole + 3) or "file")
        item.setText(self._format_item_text(name, status, media_type))
        if status not in {self.STATUS_FAILED, self.STATUS_CANCELLED}:
            item.setData(QtCore.Qt.UserRole + 2, "")
        self._update_item_tooltip(item)

    def set_failure_reason_for_path(self, path: str, reason: str):
        item = self._find_item_by_path(path)
        if item is None:
            return
        item.setData(QtCore.Qt.UserRole + 2, str(reason or ""))
        self._update_item_tooltip(item)

    def clear_all(self):
        self.list.clear()

    def remove_selected(self):
        rows = sorted({self.list.row(item) for item in self.list.selectedItems()}, reverse=True)
        for row in rows:
            self.list.takeItem(row)

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
            item = QtWidgets.QListWidgetItem(self._format_item_text(media_path.name, status, media_type))
            item.setData(QtCore.Qt.UserRole, str(media_path))
            item.setData(QtCore.Qt.UserRole + 1, status)
            item.setData(QtCore.Qt.UserRole + 2, "")
            item.setData(QtCore.Qt.UserRole + 3, media_type)
            self._update_item_tooltip(item)
            self.list.addItem(item)
            existing.add(normalized)
            added += 1

        if self.list.count() and not self.list.selectedItems():
            self.list.setCurrentRow(0)

        if added:
            self.files_added.emit(added)
        return added

    def _emit_current(self, current, _previous):
        if current is None:
            return
        path = current.data(QtCore.Qt.UserRole)
        if path:
            self.file_selected.emit(path)

    def _find_item_by_path(self, path: str):
        normalized = str(Path(path).resolve(strict=False))
        for index in range(self.list.count()):
            item = self.list.item(index)
            current_path = item.data(QtCore.Qt.UserRole)
            if current_path and str(Path(current_path).resolve(strict=False)) == normalized:
                return item
        return None

    def _format_item_text(self, name: str, status: str, media_type: str) -> str:
        return f"[{status}] [{media_type}] {name}"

    def _update_item_tooltip(self, item):
        path = str(item.data(QtCore.Qt.UserRole) or "")
        status = str(item.data(QtCore.Qt.UserRole + 1) or "")
        reason = str(item.data(QtCore.Qt.UserRole + 2) or "").strip()
        media_type = str(item.data(QtCore.Qt.UserRole + 3) or "file")

        lines = [
            f"File: {Path(path).name if path else '-'}",
            f"Type: {media_type}",
            f"Status: {status or '-'}",
        ]
        if path:
            lines.append(f"Path: {path}")
        if reason:
            lines.extend(["", "Failure reason:", reason])
        item.setToolTip("\n".join(lines))

    def _media_type_from_suffix(self, suffix: str) -> str:
        suffix = str(suffix or "").lower()
        if suffix in self.VIDEO_EXTS:
            return "video"
        if suffix in self.IMAGE_EXTS:
            return "image"
        return "file"
