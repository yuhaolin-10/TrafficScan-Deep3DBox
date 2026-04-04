import sys
import threading
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .qt import QtCore
    from ..core.lane_segmenter import LaneSegmenter
    from ..core.plate_recognizer import create_plate_recognizer
    from ..core.vehicle_detector_deep3dbox import VehicleDetector3D
    from ..core.violation_checker import ViolationChecker
    from ..services.video_pipeline import process_video
except Exception:
    try:
        from gui.qt import QtCore
    except Exception:
        from qt import QtCore
    from core.lane_segmenter import LaneSegmenter
    from core.plate_recognizer import create_plate_recognizer
    from core.vehicle_detector_deep3dbox import VehicleDetector3D
    from core.violation_checker import ViolationChecker
    from services.video_pipeline import process_video


class VideoProcessingWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str, str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str)
    task_started = QtCore.pyqtSignal(str, int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, int, int)
    task_progress = (
        QtCore.pyqtSignal(str, object, int, int)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(str, object, int, int)
    )
    task_finished = (
        QtCore.pyqtSignal(str, object, int, int)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(str, object, int, int)
    )
    task_failed = QtCore.pyqtSignal(str, str, int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str, int, int)
    finished = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    def __init__(
        self,
        video_paths,
        lane_model_path,
        vehicle_model_path,
        output_dir,
        preview_dir,
        threshold=0.3,
        layers=None,
        manual_lane_polygons_by_path=None,
        scene_regions_by_path=None,
        count_lines_by_path=None,
        frame_stride=1,
        max_frames=None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_paths = [str(Path(path).resolve(strict=False)) for path in list(video_paths)]
        self.lane_model_path = str(lane_model_path)
        self.vehicle_model_path = str(vehicle_model_path)
        self.output_dir = str(output_dir)
        self.preview_dir = str(preview_dir)
        self.threshold = float(threshold)
        self.layers = dict(layers or {})
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = None if max_frames is None else max(1, int(max_frames))
        self._stop_event = threading.Event()
        self.manual_lane_polygons_by_path = {
            str(Path(path).resolve(strict=False)): list(polygons or [])
            for path, polygons in dict(manual_lane_polygons_by_path or {}).items()
        }
        self.scene_regions_by_path = {
            str(Path(path).resolve(strict=False)): list(regions or [])
            for path, regions in dict(scene_regions_by_path or {}).items()
        }
        self.count_lines_by_path = {
            str(Path(path).resolve(strict=False)): list(lines or [])
            for path, lines in dict(count_lines_by_path or {}).items()
        }

    def request_stop(self):
        self._stop_event.set()

    def is_stop_requested(self) -> bool:
        return bool(self._stop_event.is_set())

    def _create_components(self):
        lane_detector = None
        if Path(self.lane_model_path).exists():
            lane_detector = LaneSegmenter(self.lane_model_path)
        vehicle_detector = VehicleDetector3D(self.vehicle_model_path)
        violation_checker = ViolationChecker(threshold=self.threshold)
        plate_recognizer = create_plate_recognizer()
        return lane_detector, vehicle_detector, violation_checker, plate_recognizer

    def _process_video(self, video_path, lane_detector, vehicle_detector, violation_checker, plate_recognizer, *, progress_callback=None):
        normalized_path = str(Path(video_path).resolve(strict=False))
        lane_override = self.manual_lane_polygons_by_path.get(normalized_path)
        scene_regions = self.scene_regions_by_path.get(normalized_path)
        count_lines = self.count_lines_by_path.get(normalized_path)
        return process_video(
            video_path=video_path,
            lane_detector=lane_detector,
            vehicle_detector=vehicle_detector,
            violation_checker=violation_checker,
            plate_recognizer=plate_recognizer,
            output_dir=self.output_dir,
            preview_dir=self.preview_dir,
            layers=self.layers,
            lane_override_polygons=lane_override,
            scene_regions=scene_regions,
            count_lines=count_lines,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
            stop_requested=self.is_stop_requested,
            progress_callback=progress_callback,
        )

    def run(self):
        total = len(self.video_paths)
        if total == 0:
            self.finished.emit()
            return

        try:
            self.log.emit("info", "Loading lane segmentation and Deep3DBox models for video processing...")
            lane_detector, vehicle_detector, violation_checker, plate_recognizer = self._create_components()
            if lane_detector is not None:
                self.log.emit("success", "Emergency-lane segmentation is enabled for video processing")
            else:
                self.log.emit("warning", f"Lane segmentation model not found: {self.lane_model_path}")
            if getattr(plate_recognizer, "enabled", False):
                self.log.emit("success", "Plate recognition is enabled for video processing")
            else:
                reason = str(getattr(plate_recognizer, "reason", "disabled"))
                self.log.emit("warning", f"Plate recognition is disabled for video processing: {reason}")
            self.log.emit("success", "Video model initialization complete")
        except Exception as exc:
            message = f"Video initialization failed: {exc}"
            self.log.emit("error", message)
            for index, video_path in enumerate(self.video_paths, start=1):
                self.task_failed.emit(video_path, message, index, total)
            self.finished.emit()
            return

        try:
            for index, video_path in enumerate(self.video_paths, start=1):
                if self.is_stop_requested():
                    self.log.emit("warning", "Video processing was interrupted by a newer run request")
                    break
                self.task_started.emit(video_path, index, total)
                try:
                    result = self._process_video(
                        video_path,
                        lane_detector,
                        vehicle_detector,
                        violation_checker,
                        plate_recognizer,
                        progress_callback=lambda payload, path=video_path, idx=index, total_count=total: self.task_progress.emit(path, payload, idx, total_count),
                    )
                    self.task_finished.emit(video_path, result, index, total)
                    if self.is_stop_requested():
                        self.log.emit("warning", "Video processing stop requested, remaining files were skipped")
                        break
                except InterruptedError:
                    self.log.emit("warning", f"Video processing interrupted: {Path(video_path).name}")
                    break
                except Exception as exc:
                    self.task_failed.emit(video_path, str(exc), index, total)
        finally:
            self.finished.emit()
