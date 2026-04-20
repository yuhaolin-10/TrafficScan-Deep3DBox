import sys
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
    from ..services.database_manager import DatabaseManager
    from ..services.pipeline import process_image
except Exception:
    try:
        from gui.qt import QtCore
    except Exception:
        from qt import QtCore

    from core.lane_segmenter import LaneSegmenter
    from core.plate_recognizer import create_plate_recognizer
    from core.vehicle_detector_deep3dbox import VehicleDetector3D
    from core.violation_checker import ViolationChecker
    from services.database_manager import DatabaseManager
    from services.pipeline import process_image


class ProcessingWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str, str) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str)
    task_started = QtCore.pyqtSignal(str, int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, int, int)
    task_finished = (
        QtCore.pyqtSignal(str, object, int, int)
        if hasattr(QtCore, "pyqtSignal")
        else QtCore.Signal(str, object, int, int)
    )
    task_failed = QtCore.pyqtSignal(str, str, int, int) if hasattr(QtCore, "pyqtSignal") else QtCore.Signal(str, str, int, int)
    finished = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    def __init__(
        self,
        image_paths,
        lane_model_path,
        vehicle_model_path,
        output_dir,
        db_path,
        threshold=0.3,
        location="Camera01",
        layers=None,
        manual_lane_polygons_by_path=None,
        scene_regions_by_path=None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.lane_model_path = str(lane_model_path)
        self.vehicle_model_path = str(vehicle_model_path)
        self.output_dir = str(output_dir)
        self.db_path = str(db_path)
        self.threshold = float(threshold)
        self.location = str(location)
        self.layers = dict(layers or {})
        self._stop_event = threading.Event()
        self.manual_lane_polygons_by_path = {
            str(Path(path).resolve(strict=False)): list(polygons or [])
            for path, polygons in dict(manual_lane_polygons_by_path or {}).items()
        }
        self.scene_regions_by_path = {
            str(Path(path).resolve(strict=False)): list(regions or [])
            for path, regions in dict(scene_regions_by_path or {}).items()
        }

    def request_stop(self):
        self._stop_event.set()

    def is_stop_requested(self) -> bool:
        return bool(self._stop_event.is_set())

    def run(self):
        total = len(self.image_paths)
        if total == 0:
            self.finished.emit()
            return

        db_manager = None
        try:
            self.log.emit("info", "Loading lane segmentation and Deep3DBox models...")
            lane_detector = None
            if Path(self.lane_model_path).exists():
                lane_detector = LaneSegmenter(self.lane_model_path)
                self.log.emit("success", "Emergency-lane segmentation is enabled")
            else:
                self.log.emit("warning", f"Lane segmentation model not found: {self.lane_model_path}")
            vehicle_detector = VehicleDetector3D(self.vehicle_model_path)
            violation_checker = ViolationChecker(threshold=self.threshold)
            plate_recognizer = create_plate_recognizer()
            db_manager = DatabaseManager(self.db_path)
            if getattr(plate_recognizer, "enabled", False):
                self.log.emit("success", "Plate recognition is enabled")
            else:
                reason = str(getattr(plate_recognizer, "reason", "disabled"))
                self.log.emit("warning", f"Plate recognition is disabled: {reason}")
            self.log.emit("success", "Model initialization complete")
        except Exception as exc:
            message = f"Initialization failed: {exc}"
            self.log.emit("error", message)
            for index, image_path in enumerate(self.image_paths, start=1):
                self.task_failed.emit(image_path, message, index, total)
            self.finished.emit()
            return

        try:
            for index, image_path in enumerate(self.image_paths, start=1):
                if self.is_stop_requested():
                    self.log.emit("warning", "Image processing was interrupted by a newer run request")
                    break
                self.task_started.emit(image_path, index, total)
                record_id = db_manager.start_record(image_path)
                try:
                    normalized_path = str(Path(image_path).resolve(strict=False))
                    lane_override = self.manual_lane_polygons_by_path.get(normalized_path)
                    scene_regions = self.scene_regions_by_path.get(normalized_path)
                    result = process_image(
                        image_path=image_path,
                        lane_detector=lane_detector,
                        vehicle_detector=vehicle_detector,
                        violation_checker=violation_checker,
                        plate_recognizer=plate_recognizer,
                        output_dir=self.output_dir,
                        layers=self.layers,
                        lane_override_polygons=lane_override,
                        scene_regions=scene_regions,
                    )
                    if record_id <= 0:
                        raise RuntimeError("Failed to create running database record")
                    db_manager.complete_record_success(record_id, result)
                    result["record_id"] = int(record_id)
                    self.task_finished.emit(image_path, result, index, total)
                    if self.is_stop_requested():
                        self.log.emit("warning", "Image processing stop requested, remaining files were skipped")
                        break
                except Exception as exc:
                    if record_id and record_id > 0:
                        db_manager.mark_record_failed(record_id, exc)
                    self.task_failed.emit(image_path, str(exc), index, total)
        finally:
            if db_manager is not None:
                db_manager.close()
            self.finished.emit()
