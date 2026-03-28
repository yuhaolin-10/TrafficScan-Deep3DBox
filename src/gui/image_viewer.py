from typing import List, Optional

try:
    from .qt import QtCore, QtGui, QtWidgets
except Exception:
    try:
        from gui.qt import QtCore, QtGui, QtWidgets
    except Exception:
        from qt import QtCore, QtGui, QtWidgets


class _ImageViewerView(QtWidgets.QGraphicsView):
    def __init__(self, owner, scene):
        super().__init__(scene)
        self._owner = owner
        self._press_pos = None

    def _event_view_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        if hasattr(event, "pos"):
            return event.pos()
        return None

    def wheelEvent(self, event):
        if self._owner._has_pixmap():
            self._owner._on_wheel(event)
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if self._owner._handle_key_press(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._owner.is_region_direction_drawing() and event.button() == QtCore.Qt.LeftButton:
            self._owner.finish_region_direction_drawing()
            event.accept()
            return
        if self._owner.is_manual_roi_drawing() and event.button() == QtCore.Qt.LeftButton:
            self._owner.finish_manual_roi_drawing()
            event.accept()
            return
        if self._owner._has_pixmap() and event.button() == QtCore.Qt.LeftButton:
            self._owner.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if self._owner._has_pixmap():
            self.setFocus(QtCore.Qt.MouseFocusReason)
        if self._owner.is_region_direction_drawing():
            if event.button() == QtCore.Qt.RightButton:
                self._owner.cancel_region_direction_drawing()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                self._press_pos = self._event_view_pos(event)
                event.accept()
                return
        if self._owner.is_manual_roi_drawing():
            if event.button() == QtCore.Qt.RightButton:
                self._owner.finish_manual_roi_drawing()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                self._press_pos = self._event_view_pos(event)
                event.accept()
                return
        if self._owner._has_pixmap() and event.button() == QtCore.Qt.LeftButton:
            self._press_pos = self._event_view_pos(event)
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        if self._owner._has_pixmap() and event.button() == QtCore.Qt.RightButton:
            self._press_pos = self._event_view_pos(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._owner.is_region_direction_drawing() and event.button() == QtCore.Qt.LeftButton:
            release_pos = self._event_view_pos(event)
            if self._press_pos is not None and release_pos is not None:
                if (release_pos - self._press_pos).manhattanLength() <= 4:
                    self._owner._emit_direction_click(release_pos)
            self._press_pos = None
            event.accept()
            return
        if self._owner.is_manual_roi_drawing() and event.button() == QtCore.Qt.LeftButton:
            release_pos = self._event_view_pos(event)
            if self._press_pos is not None and release_pos is not None:
                if (release_pos - self._press_pos).manhattanLength() <= 4:
                    self._owner._emit_click(release_pos)
            self._press_pos = None
            event.accept()
            return
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        if self._owner._has_pixmap() and event.button() == QtCore.Qt.RightButton:
            release_pos = self._event_view_pos(event)
            if self._press_pos is not None and release_pos is not None:
                if (release_pos - self._press_pos).manhattanLength() <= 4:
                    self._owner._emit_context_request(release_pos, event)
            self._press_pos = None
            event.accept()
            return
        if self._owner._has_pixmap() and event.button() == QtCore.Qt.LeftButton:
            release_pos = self._event_view_pos(event)
            if self._press_pos is not None and release_pos is not None:
                if (release_pos - self._press_pos).manhattanLength() <= 4:
                    self._owner._emit_click(release_pos)
            self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._owner._has_pixmap():
            pos = self._event_view_pos(event)
            if pos is not None:
                if self._owner.is_region_direction_drawing():
                    self._owner._update_direction_hover(pos)
                elif self._owner.is_manual_roi_drawing():
                    self._owner._update_manual_hover(pos)
                else:
                    self._owner._emit_hover(pos)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._owner.is_region_direction_drawing():
            self._owner._clear_direction_hover()
        elif self._owner.is_manual_roi_drawing():
            self._owner._clear_manual_hover()
        else:
            self._owner._emit_hover_leave()
        super().leaveEvent(event)


class ImageViewer(QtWidgets.QFrame):
    resized = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    zoom_changed = QtCore.pyqtSignal(int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int)
    point_clicked = QtCore.pyqtSignal(int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int, int)
    point_hovered = QtCore.pyqtSignal(int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int, int)
    hover_left = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()
    manual_roi_changed = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_finished = QtCore.pyqtSignal(object) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(object)
    manual_roi_drawing_changed = QtCore.pyqtSignal(bool) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(bool)
    manual_roi_context_requested = (
        QtCore.pyqtSignal(int, object, object)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(int, object, object)
    )
    region_direction_drawing_changed = QtCore.pyqtSignal(bool) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(bool)
    region_direction_finished = (
        QtCore.pyqtSignal(int, object)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(int, object)
    )

    def __init__(self, placeholder_text: str = "Drop/select an image to preview", parent=None):
        super().__init__(parent)
        self._mode = "fit"
        self._min_scale = 0.05
        self._max_scale = 20.0

        self._manual_roi_regions: List[List[List[float]]] = []
        self._selected_manual_roi_index: Optional[int] = None
        self._draft_roi_points: List[List[float]] = []
        self._draft_hover_point: Optional[List[float]] = None
        self._manual_roi_drawing = False
        self._region_direction_lines: List[List[List[float]]] = []
        self._direction_drawing = False
        self._direction_region_index: Optional[int] = None
        self._direction_draft_points: List[List[float]] = []
        self._direction_hover_point: Optional[List[float]] = None

        self._placeholder = QtWidgets.QLabel(placeholder_text)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setStyleSheet("color:#6b7280;font-family:Consolas;")

        self._scene = QtWidgets.QGraphicsScene(self)
        self._pix_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self._manual_polygon_item = QtWidgets.QGraphicsPathItem()
        self._manual_polygon_item.setZValue(10)
        manual_pen = QtGui.QPen(QtCore.Qt.NoPen)
        self._manual_polygon_item.setPen(manual_pen)
        self._manual_polygon_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 159, 67, 60)))
        self._scene.addItem(self._manual_polygon_item)

        self._selected_polygon_item = QtWidgets.QGraphicsPathItem()
        self._selected_polygon_item.setZValue(10.4)
        selected_pen = QtGui.QPen(QtCore.Qt.NoPen)
        self._selected_polygon_item.setPen(selected_pen)
        self._selected_polygon_item.setBrush(QtGui.QBrush(QtGui.QColor(250, 204, 21, 88)))
        self._scene.addItem(self._selected_polygon_item)

        self._draft_polygon_item = QtWidgets.QGraphicsPolygonItem()
        self._draft_polygon_item.setZValue(10.8)
        draft_fill_pen = QtGui.QPen(QtGui.QColor(255, 159, 67), 2.0)
        draft_fill_pen.setCosmetic(True)
        self._draft_polygon_item.setPen(draft_fill_pen)
        self._draft_polygon_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 159, 67, 110)))
        self._scene.addItem(self._draft_polygon_item)

        self._draft_path_item = QtWidgets.QGraphicsPathItem()
        self._draft_path_item.setZValue(11)
        draft_pen = QtGui.QPen(QtGui.QColor(56, 189, 248), 3.0)
        draft_pen.setCosmetic(True)
        draft_pen.setStyle(QtCore.Qt.DashLine)
        self._draft_path_item.setPen(draft_pen)
        self._scene.addItem(self._draft_path_item)

        self._direction_draft_item = QtWidgets.QGraphicsPathItem()
        self._direction_draft_item.setZValue(11.4)
        direction_draft_pen = QtGui.QPen(QtGui.QColor(34, 211, 238), 3.0)
        direction_draft_pen.setCosmetic(True)
        direction_draft_pen.setStyle(QtCore.Qt.DashLine)
        self._direction_draft_item.setPen(direction_draft_pen)
        self._scene.addItem(self._direction_draft_item)

        self._vertex_items: List[QtWidgets.QGraphicsEllipseItem] = []
        self._draft_vertex_items: List[QtWidgets.QGraphicsEllipseItem] = []
        self._direction_line_items: List[QtWidgets.QGraphicsPathItem] = []

        self._view = _ImageViewerView(self, self._scene)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._view.setBackgroundBrush(QtGui.QColor("#0f0f0f"))
        self._view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._view.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self._view.setInteractive(False)
        self._view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._view.setMouseTracking(True)
        self.setMouseTracking(True)

        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self._view)
        self._stack.setCurrentIndex(0)

        self._refresh_manual_roi_items()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
        if self._mode == "fit":
            self.fit_to_window()

    def _has_pixmap(self) -> bool:
        pix = self._pix_item.pixmap()
        return not pix.isNull()

    def set_placeholder_text(self, text: str):
        self._placeholder.setText(text)

    def clear(self, placeholder_text: Optional[str] = None):
        if placeholder_text is not None:
            self._placeholder.setText(placeholder_text)
        self._pix_item.setPixmap(QtGui.QPixmap())
        self._scene.setSceneRect(QtCore.QRectF())
        self._stack.setCurrentIndex(0)
        self._mode = "fit"
        self.zoom_changed.emit(0)
        self._refresh_manual_roi_items()

    def set_pixmap(self, pixmap: Optional[QtGui.QPixmap], *, reset_view: bool = True):
        if pixmap is None or pixmap.isNull():
            self.clear("Preview unavailable")
            return
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        self._stack.setCurrentIndex(1)
        self._refresh_manual_roi_items()
        if reset_view:
            self.fit_to_window()
        else:
            self.refresh()

    def refresh(self):
        self._refresh_manual_roi_items()
        if not self._has_pixmap():
            self.zoom_changed.emit(0)
            return
        if self._mode == "fit":
            self.fit_to_window()
            return
        self._emit_zoom()

    def fit_to_window(self):
        if not self._has_pixmap():
            self.zoom_changed.emit(0)
            return
        self._mode = "fit"
        self._view.resetTransform()
        rect = self._pix_item.boundingRect()
        if rect.isNull():
            self.zoom_changed.emit(0)
            return
        self._view.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self._view.centerOn(self._pix_item)
        self._emit_zoom()

    def zoom_in(self):
        self._zoom_by(1.25)

    def zoom_out(self):
        self._zoom_by(0.8)

    def _on_wheel(self, event):
        angle = event.angleDelta().y()
        if angle == 0:
            return
        self._mode = "manual"
        factor = 1.25 ** (angle / 120.0)
        if hasattr(event, "position"):
            view_pos = event.position().toPoint()
        elif hasattr(event, "pos"):
            view_pos = event.pos()
        else:
            view_pos = None
        self._zoom_by(factor, view_pos=view_pos)
        event.accept()

    def _zoom_by(self, factor: float, *, view_pos=None):
        if not self._has_pixmap():
            self.zoom_changed.emit(0)
            return
        self._mode = "manual"
        current = float(self._view.transform().m11()) or 1.0
        target = current * factor
        if target < self._min_scale:
            factor = self._min_scale / current
        elif target > self._max_scale:
            factor = self._max_scale / current

        before = None
        if view_pos is not None:
            before = self._view.mapToScene(view_pos)

        self._view.scale(factor, factor)

        if before is not None:
            after = self._view.mapToScene(view_pos)
            delta = before - after
            self._view.translate(delta.x(), delta.y())

        self._emit_zoom()

    def _emit_zoom(self):
        if not self._has_pixmap():
            self.zoom_changed.emit(0)
            return
        scale = float(self._view.transform().m11()) or 0.0
        percent = int(round(scale * 100))
        self.zoom_changed.emit(max(0, percent))

    def manual_roi_regions(self) -> List[List[List[float]]]:
        return [
            [[float(point[0]), float(point[1])] for point in region]
            for region in self._manual_roi_regions
        ]

    def manual_roi_points(self):
        return self.manual_roi_regions()

    def selected_manual_roi_index(self) -> Optional[int]:
        if self._selected_manual_roi_index is None:
            return None
        if 0 <= int(self._selected_manual_roi_index) < len(self._manual_roi_regions):
            return int(self._selected_manual_roi_index)
        return None

    def set_manual_roi(self, points, *, emit_signal: bool = False):
        normalized_regions = self._normalize_regions(points)
        self._manual_roi_regions = normalized_regions
        self._selected_manual_roi_index = None
        self._region_direction_lines = self._normalize_direction_lines(self._region_direction_lines, normalized_regions)
        self._draft_roi_points = []
        self._draft_hover_point = None
        self._manual_roi_drawing = False
        self._direction_drawing = False
        self._direction_region_index = None
        self._direction_draft_points = []
        self._direction_hover_point = None
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        if emit_signal:
            self.manual_roi_changed.emit(self.manual_roi_regions())
        self.manual_roi_drawing_changed.emit(False)
        self.region_direction_drawing_changed.emit(False)

    def clear_manual_roi(self, *, emit_signal: bool = True):
        self._manual_roi_regions = []
        self._selected_manual_roi_index = None
        self._region_direction_lines = []
        self._draft_roi_points = []
        self._draft_hover_point = None
        was_drawing = self._manual_roi_drawing
        self._manual_roi_drawing = False
        was_direction_drawing = self._direction_drawing
        self._direction_drawing = False
        self._direction_region_index = None
        self._direction_draft_points = []
        self._direction_hover_point = None
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        if emit_signal:
            self.manual_roi_changed.emit([])
        if was_drawing:
            self.manual_roi_drawing_changed.emit(False)
        if was_direction_drawing:
            self.region_direction_drawing_changed.emit(False)

    def start_manual_roi_drawing(self, *, clear_existing: bool = False):
        if self._direction_drawing:
            self.cancel_region_direction_drawing()
        if clear_existing:
            self._manual_roi_regions = []
            self._selected_manual_roi_index = None
            self._region_direction_lines = []
        self._draft_roi_points = []
        self._draft_hover_point = None
        self._manual_roi_drawing = True
        self._set_manual_roi_cursor(True)
        self._refresh_manual_roi_items()
        self.manual_roi_drawing_changed.emit(True)

    def finish_manual_roi_drawing(self):
        if len(self._draft_roi_points) < 3:
            return None
        region_points = self._normalize_points(self._draft_roi_points)
        self._manual_roi_regions.append(region_points)
        self._region_direction_lines.append([])
        self._selected_manual_roi_index = len(self._manual_roi_regions) - 1
        self._draft_roi_points = []
        self._draft_hover_point = None
        self._manual_roi_drawing = False
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        regions = self.manual_roi_regions()
        self.manual_roi_changed.emit(regions)
        self.manual_roi_finished.emit(regions)
        self.manual_roi_drawing_changed.emit(False)
        return [[float(point[0]), float(point[1])] for point in region_points]

    def cancel_manual_roi_drawing(self):
        if not self._manual_roi_drawing:
            return
        self._draft_roi_points = []
        self._draft_hover_point = None
        self._manual_roi_drawing = False
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        self.manual_roi_drawing_changed.emit(False)

    def delete_selected_manual_roi(self, *, emit_signal: bool = True):
        selected_index = self.selected_manual_roi_index()
        if selected_index is None:
            return None
        deleted = self._manual_roi_regions.pop(selected_index)
        if 0 <= selected_index < len(self._region_direction_lines):
            self._region_direction_lines.pop(selected_index)
        if not self._manual_roi_regions:
            self._selected_manual_roi_index = None
        else:
            self._selected_manual_roi_index = min(selected_index, len(self._manual_roi_regions) - 1)
        self._refresh_manual_roi_items()
        if emit_signal:
            self.manual_roi_changed.emit(self.manual_roi_regions())
        return [[float(point[0]), float(point[1])] for point in deleted]

    def is_manual_roi_drawing(self) -> bool:
        return bool(self._manual_roi_drawing)

    def region_direction_lines(self) -> List[List[List[float]]]:
        return [
            [[float(point[0]), float(point[1])] for point in line]
            if len(line) == 2
            else []
            for line in self._region_direction_lines
        ]

    def set_region_direction_lines(self, direction_lines):
        self._region_direction_lines = self._normalize_direction_lines(direction_lines, self._manual_roi_regions)
        self._refresh_manual_roi_items()

    def is_region_direction_drawing(self) -> bool:
        return bool(self._direction_drawing)

    def start_region_direction_drawing(self, region_index: int):
        if not (0 <= int(region_index) < len(self._manual_roi_regions)):
            return
        if self._manual_roi_drawing:
            self.cancel_manual_roi_drawing()
        self._selected_manual_roi_index = int(region_index)
        self._direction_region_index = int(region_index)
        self._direction_draft_points = []
        self._direction_hover_point = None
        self._direction_drawing = True
        self._set_manual_roi_cursor(True)
        self._refresh_manual_roi_items()
        self.region_direction_drawing_changed.emit(True)

    def finish_region_direction_drawing(self):
        if self._direction_region_index is None or len(self._direction_draft_points) < 2:
            self.cancel_region_direction_drawing()
            return None
        line = self._normalize_direction_line(self._direction_draft_points[:2])
        if line and 0 <= self._direction_region_index < len(self._manual_roi_regions):
            while len(self._region_direction_lines) < len(self._manual_roi_regions):
                self._region_direction_lines.append([])
            self._region_direction_lines[self._direction_region_index] = line
        finished_region_index = int(self._direction_region_index)
        self._direction_region_index = None
        self._direction_draft_points = []
        self._direction_hover_point = None
        self._direction_drawing = False
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        self.region_direction_finished.emit(finished_region_index, line or [])
        self.region_direction_drawing_changed.emit(False)
        return line

    def cancel_region_direction_drawing(self):
        if not self._direction_drawing:
            return
        self._direction_region_index = None
        self._direction_draft_points = []
        self._direction_hover_point = None
        self._direction_drawing = False
        self._set_manual_roi_cursor(False)
        self._refresh_manual_roi_items()
        self.region_direction_drawing_changed.emit(False)

    def _is_point_pair(self, value) -> bool:
        try:
            if len(value) != 2:
                return False
            float(value[0])
            float(value[1])
            return True
        except Exception:
            return False

    def _normalize_regions(self, regions) -> List[List[List[float]]]:
        normalized_regions: List[List[List[float]]] = []
        if not regions:
            return normalized_regions
        first = regions[0] if len(regions) > 0 else None
        if first is not None and self._is_point_pair(first):
            polygon = self._normalize_points(regions)
            if len(polygon) >= 3:
                normalized_regions.append(polygon)
            return normalized_regions

        for region in regions:
            polygon = self._normalize_points(region)
            if len(polygon) >= 3:
                normalized_regions.append(polygon)
        return normalized_regions

    def _normalize_direction_line(self, points) -> List[List[float]]:
        if not points or len(points) < 2:
            return []
        normalized = self._normalize_points(points[:2])
        if len(normalized) != 2:
            return []
        if self._distance(normalized[0], normalized[1]) <= 1.0:
            return []
        return normalized

    def _normalize_direction_lines(self, lines, regions) -> List[List[List[float]]]:
        normalized_lines: List[List[List[float]]] = []
        total_regions = len(regions or [])
        for index in range(total_regions):
            line = []
            if lines and index < len(lines):
                line = self._normalize_direction_line(lines[index])
            normalized_lines.append(line)
        return normalized_lines

    def _normalize_points(self, points) -> List[List[float]]:
        normalized: List[List[float]] = []
        if not points:
            return normalized
        rect = self._pix_item.boundingRect()
        for point in points:
            if point is None or len(point) != 2:
                continue
            x = float(point[0])
            y = float(point[1])
            if not rect.isNull():
                x = min(max(x, rect.left()), rect.right())
                y = min(max(y, rect.top()), rect.bottom())
            normalized.append([x, y])
        return normalized

    def _scene_point_from_view(self, view_pos):
        if not self._has_pixmap() or view_pos is None:
            return None
        scene_pos = self._view.mapToScene(view_pos)
        rect = self._pix_item.boundingRect()
        if rect.isNull():
            return None
        x = min(max(float(scene_pos.x()), rect.left()), rect.right())
        y = min(max(float(scene_pos.y()), rect.top()), rect.bottom())
        return [x, y]

    def _distance(self, point_a, point_b) -> float:
        dx = float(point_a[0]) - float(point_b[0])
        dy = float(point_a[1]) - float(point_b[1])
        return (dx * dx + dy * dy) ** 0.5

    def _handle_key_press(self, event) -> bool:
        key = int(event.key())
        if key == int(QtCore.Qt.Key_Escape) and self._direction_drawing:
            self.cancel_region_direction_drawing()
            return True
        if key == int(QtCore.Qt.Key_Escape) and self._manual_roi_drawing:
            self.cancel_manual_roi_drawing()
            return True
        if key in {int(QtCore.Qt.Key_Delete), int(QtCore.Qt.Key_Backspace)}:
            if self._manual_roi_drawing and self._draft_roi_points:
                self._draft_roi_points.pop()
                self._refresh_manual_roi_items()
                return True
            if self._direction_drawing and self._direction_draft_points:
                self._direction_draft_points.pop()
                self._refresh_manual_roi_items()
                return True
            if self.delete_selected_manual_roi() is not None:
                return True
        return False

    def _emit_click(self, view_pos):
        scene_point = self._scene_point_from_view(view_pos)
        if self.is_manual_roi_drawing():
            if scene_point is not None:
                self._add_manual_point(scene_point)
            return
        if not self._has_pixmap() or view_pos is None or scene_point is None:
            return

        hit_index = self._region_hit_index(scene_point)
        if hit_index is not None:
            if hit_index != self._selected_manual_roi_index:
                self._selected_manual_roi_index = hit_index
                self._refresh_manual_roi_items()
            return

        if self._selected_manual_roi_index is not None:
            self._selected_manual_roi_index = None
            self._refresh_manual_roi_items()

        scene_pos = self._view.mapToScene(view_pos)
        rect = self._pix_item.boundingRect()
        if not rect.contains(scene_pos):
            return
        x = int(round(scene_pos.x()))
        y = int(round(scene_pos.y()))
        self.point_clicked.emit(x, y)

    def _emit_context_request(self, view_pos, event):
        scene_point = self._scene_point_from_view(view_pos)
        if scene_point is None:
            return
        hit_index = self._region_hit_index(scene_point)
        if hit_index is None:
            return
        if hit_index != self._selected_manual_roi_index:
            self._selected_manual_roi_index = hit_index
            self._refresh_manual_roi_items()
        global_point = None
        if hasattr(event, "globalPosition"):
            global_point = event.globalPosition().toPoint()
        elif hasattr(event, "globalPos"):
            global_point = event.globalPos()
        if global_point is None:
            global_point = self._view.viewport().mapToGlobal(view_pos)
        self.manual_roi_context_requested.emit(
            int(hit_index),
            [int(global_point.x()), int(global_point.y())],
            [float(scene_point[0]), float(scene_point[1])],
        )

    def _emit_hover(self, view_pos):
        if not self._has_pixmap() or view_pos is None:
            return
        scene_pos = self._view.mapToScene(view_pos)
        rect = self._pix_item.boundingRect()
        if not rect.contains(scene_pos):
            self.hover_left.emit()
            return
        x = int(round(scene_pos.x()))
        y = int(round(scene_pos.y()))
        self.point_hovered.emit(x, y)

    def _emit_hover_leave(self):
        self.hover_left.emit()

    def _add_manual_point(self, point):
        if len(self._draft_roi_points) >= 3 and self._distance(point, self._draft_roi_points[0]) <= 10.0:
            self.finish_manual_roi_drawing()
            return
        self._draft_roi_points.append([float(point[0]), float(point[1])])
        self._refresh_manual_roi_items()

    def _emit_direction_click(self, view_pos):
        scene_point = self._scene_point_from_view(view_pos)
        if scene_point is None:
            return
        self._direction_draft_points.append([float(scene_point[0]), float(scene_point[1])])
        if len(self._direction_draft_points) >= 2:
            self.finish_region_direction_drawing()
            return
        self._refresh_manual_roi_items()

    def _update_manual_hover(self, view_pos):
        point = self._scene_point_from_view(view_pos)
        self._draft_hover_point = point
        self._refresh_manual_roi_items()

    def _clear_manual_hover(self):
        if self._draft_hover_point is None:
            return
        self._draft_hover_point = None
        self._refresh_manual_roi_items()

    def _update_direction_hover(self, view_pos):
        point = self._scene_point_from_view(view_pos)
        self._direction_hover_point = point
        self._refresh_manual_roi_items()

    def _clear_direction_hover(self):
        if self._direction_hover_point is None:
            return
        self._direction_hover_point = None
        self._refresh_manual_roi_items()

    def _clear_graphics_items(self, items: List[QtWidgets.QGraphicsItem]):
        for item in items:
            self._scene.removeItem(item)
        items.clear()

    def _make_vertex_item(self, point, color: QtGui.QColor, radius: float, z_value: float):
        item = QtWidgets.QGraphicsEllipseItem(-radius, -radius, radius * 2.0, radius * 2.0)
        item.setPos(float(point[0]), float(point[1]))
        item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        outline_pen = QtGui.QPen(QtGui.QColor("#ffffff"), 1.25)
        outline_pen.setCosmetic(True)
        item.setPen(outline_pen)
        item.setBrush(QtGui.QBrush(color))
        item.setZValue(z_value)
        self._scene.addItem(item)
        return item

    def _make_direction_item(self, line_points, *, highlighted: bool = False):
        if not line_points or len(line_points) != 2:
            return None
        start = QtCore.QPointF(float(line_points[0][0]), float(line_points[0][1]))
        end = QtCore.QPointF(float(line_points[1][0]), float(line_points[1][1]))
        path = QtGui.QPainterPath(start)
        path.lineTo(end)
        dx = float(end.x() - start.x())
        dy = float(end.y() - start.y())
        length = (dx * dx + dy * dy) ** 0.5
        if length > 1e-6:
            ux = dx / length
            uy = dy / length
            arrow_len = min(18.0, max(10.0, length * 0.18))
            wing = arrow_len * 0.55
            left = QtCore.QPointF(
                float(end.x() - ux * arrow_len + uy * wing),
                float(end.y() - uy * arrow_len - ux * wing),
            )
            right = QtCore.QPointF(
                float(end.x() - ux * arrow_len - uy * wing),
                float(end.y() - uy * arrow_len + ux * wing),
            )
            path.moveTo(end)
            path.lineTo(left)
            path.moveTo(end)
            path.lineTo(right)

        item = QtWidgets.QGraphicsPathItem(path)
        pen = QtGui.QPen(QtGui.QColor(34, 211, 238) if not highlighted else QtGui.QColor(250, 204, 21), 3.0 if highlighted else 2.5)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QtCore.Qt.NoBrush)
        item.setZValue(12.2 if highlighted else 11.8)
        self._scene.addItem(item)
        return item

    def _set_manual_roi_cursor(self, active: bool):
        viewport = self._view.viewport()
        if active:
            viewport.setCursor(QtCore.Qt.CrossCursor)
            return
        viewport.unsetCursor()

    def _polygon_path(self, region) -> QtGui.QPainterPath:
        polygon = QtGui.QPolygonF([QtCore.QPointF(point[0], point[1]) for point in region])
        path = QtGui.QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()
        return path

    def _combined_polygon_path(self, regions) -> QtGui.QPainterPath:
        combined = QtGui.QPainterPath()
        for region in regions or []:
            if len(region) < 3:
                continue
            combined.addPath(self._polygon_path(region))
        return combined

    def _region_hit_index(self, point) -> Optional[int]:
        if not self._manual_roi_regions:
            return None
        point_f = QtCore.QPointF(float(point[0]), float(point[1]))
        scale = abs(float(self._view.transform().m11()) or 1.0)
        pick_width = max(6.0, 12.0 / max(0.1, scale))
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(float(pick_width))
        for index in range(len(self._manual_roi_regions) - 1, -1, -1):
            region = self._manual_roi_regions[index]
            if len(region) < 3:
                continue
            path = self._polygon_path(region)
            if path.contains(point_f):
                return index
            if stroker.createStroke(path).contains(point_f):
                return index
        return None

    def _refresh_manual_roi_items(self):
        rect = self._pix_item.boundingRect()
        if rect.isNull():
            self._manual_polygon_item.setVisible(False)
            self._manual_polygon_item.setPath(QtGui.QPainterPath())
            self._selected_polygon_item.setVisible(False)
            self._selected_polygon_item.setPath(QtGui.QPainterPath())
            self._draft_polygon_item.setVisible(False)
            self._draft_polygon_item.setPolygon(QtGui.QPolygonF())
            self._draft_path_item.setVisible(False)
            self._draft_path_item.setPath(QtGui.QPainterPath())
            self._direction_draft_item.setVisible(False)
            self._direction_draft_item.setPath(QtGui.QPainterPath())
            self._clear_graphics_items(self._vertex_items)
            self._clear_graphics_items(self._draft_vertex_items)
            self._clear_graphics_items(self._direction_line_items)
            return

        self._clear_graphics_items(self._vertex_items)
        self._clear_graphics_items(self._draft_vertex_items)
        self._clear_graphics_items(self._direction_line_items)

        normal_regions = []
        selected_regions = []
        selected_index = self.selected_manual_roi_index()
        for index, region in enumerate(self._manual_roi_regions):
            if len(region) < 3:
                continue
            if selected_index is not None and index == selected_index:
                selected_regions.append(region)
            else:
                normal_regions.append(region)

        normal_path = self._combined_polygon_path(normal_regions)
        self._manual_polygon_item.setPath(normal_path)
        self._manual_polygon_item.setVisible(not normal_path.isEmpty())

        selected_path = self._combined_polygon_path(selected_regions)
        self._selected_polygon_item.setPath(selected_path)
        self._selected_polygon_item.setVisible(not selected_path.isEmpty())

        for index, line_points in enumerate(self._region_direction_lines):
            if not line_points or len(line_points) != 2:
                continue
            self._direction_line_items.append(
                self._make_direction_item(
                    line_points,
                    highlighted=(selected_index is not None and index == selected_index),
                )
            )

        draft_points = list(self._draft_roi_points)
        if self._manual_roi_drawing and self._draft_hover_point is not None:
            draft_points = draft_points + [self._draft_hover_point]

        if self._manual_roi_drawing and len(draft_points) >= 3:
            draft_polygon = QtGui.QPolygonF([QtCore.QPointF(point[0], point[1]) for point in draft_points])
            self._draft_polygon_item.setPolygon(draft_polygon)
            self._draft_polygon_item.setVisible(True)
        else:
            self._draft_polygon_item.setVisible(False)
            self._draft_polygon_item.setPolygon(QtGui.QPolygonF())

        if self._manual_roi_drawing and len(draft_points) >= 1:
            path = QtGui.QPainterPath(QtCore.QPointF(draft_points[0][0], draft_points[0][1]))
            for point in draft_points[1:]:
                path.lineTo(point[0], point[1])
            if len(self._draft_roi_points) >= 3 and self._draft_hover_point is None:
                path.closeSubpath()
            self._draft_path_item.setPath(path)
            self._draft_path_item.setVisible(True)
            for point in self._draft_roi_points:
                self._draft_vertex_items.append(self._make_vertex_item(point, QtGui.QColor(56, 189, 248), 3.5, 13.0))
            if self._draft_hover_point is not None:
                self._draft_vertex_items.append(
                    self._make_vertex_item(self._draft_hover_point, QtGui.QColor(148, 163, 184), 3.0, 13.0)
                )
        else:
            self._draft_path_item.setVisible(False)
            self._draft_path_item.setPath(QtGui.QPainterPath())

        direction_points = list(self._direction_draft_points)
        if self._direction_drawing and self._direction_hover_point is not None:
            direction_points = direction_points + [self._direction_hover_point]
        if self._direction_drawing and len(direction_points) >= 1:
            path = QtGui.QPainterPath(QtCore.QPointF(direction_points[0][0], direction_points[0][1]))
            if len(direction_points) >= 2:
                path.lineTo(direction_points[1][0], direction_points[1][1])
            self._direction_draft_item.setPath(path)
            self._direction_draft_item.setVisible(True)
            for point in self._direction_draft_points:
                self._draft_vertex_items.append(
                    self._make_vertex_item(point, QtGui.QColor(34, 211, 238), 3.5, 13.2)
                )
            if self._direction_hover_point is not None:
                self._draft_vertex_items.append(
                    self._make_vertex_item(self._direction_hover_point, QtGui.QColor(125, 211, 252), 3.0, 13.2)
                )
        else:
            self._direction_draft_item.setVisible(False)
            self._direction_draft_item.setPath(QtGui.QPainterPath())
