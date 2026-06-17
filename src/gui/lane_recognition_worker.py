import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .qt import QtCore
    from ..core.lane_segmenter import LaneSegmenter
    from ..services.video_reader import read_video_frame
except Exception:
    try:
        from gui.qt import QtCore
    except Exception:
        from qt import QtCore

    from core.lane_segmenter import LaneSegmenter
    from services.video_reader import read_video_frame


class LaneRecognitionWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str, str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str)
    progress = QtCore.pyqtSignal(int, str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(int, str)
    task_finished = (
        QtCore.pyqtSignal(str, object)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(str, object)
    )
    task_failed = QtCore.pyqtSignal(str, str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str)
    finished = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    def __init__(self, media_path, lane_model_path, *, is_video=False, parent=None):
        super().__init__(parent)
        self.media_path = str(Path(media_path).resolve(strict=False))
        self.lane_model_path = str(lane_model_path)
        self.is_video = bool(is_video)

    def _normalize_polygons(self, polygons):
        normalized = []
        if polygons is None:
            return normalized
        for polygon in list(polygons):
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                continue
            points = []
            for point in arr:
                points.append([float(point[0]), float(point[1])])
            normalized.append(points)
        return normalized

    def _load_frame(self):
        if self.is_video:
            return read_video_frame(self.media_path, frame_index=0)
        frame = cv2.imread(self.media_path)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {self.media_path}")
        return frame

    def run(self):
        try:
            model_path = Path(self.lane_model_path)
            if not model_path.exists():
                raise RuntimeError(f"Lane segmentation model not found: {model_path}")

            self.progress.emit(10, "正在加载应急车道分割模型")
            self.log.emit("info", f"Loading lane segmentation model for {Path(self.media_path).name}")
            lane_detector = LaneSegmenter(str(model_path))

            self.progress.emit(45, "正在读取待识别画面")
            frame = self._load_frame()

            self.progress.emit(75, "正在识别应急车道")
            lane_mask, lane_polygons = lane_detector.detect(frame)
            normalized_polygons = self._normalize_polygons(lane_polygons)

            payload = {
                "media_path": self.media_path,
                "lane_polygons": normalized_polygons,
                "lane_mask_available": lane_mask is not None,
                "source_frame_index": 0 if self.is_video else None,
            }
            self.progress.emit(100, "应急车道识别完成")
            self.task_finished.emit(self.media_path, payload)
        except Exception as exc:
            self.task_failed.emit(self.media_path, str(exc))
        finally:
            self.finished.emit()
