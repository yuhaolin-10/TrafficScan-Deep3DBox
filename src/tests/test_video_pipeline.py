import os
import sys
from types import SimpleNamespace

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.plate_recognizer import PlateCandidate
from services.pipeline import process_frame
from services.video_pipeline import (
    _new_track_plate_ocr_state,
    _should_run_track_plate_ocr,
    _summarize_violation_plate_candidates,
)


class _DummyRecognizer:
    enabled = True
    min_confidence = 0.55


class _DummyVehicleDetector:
    def detect(self, _frame, lane_polygons=None):
        return []


class _DummyViolationChecker:
    def check(self, _footprint, _lane_mask):
        return False, 0.0


def test_violation_plate_summary_uses_only_nearby_frames():
    summary = _summarize_violation_plate_candidates(
        [
            PlateCandidate(text="浜珹12345", confidence=0.92, frame_index=12, plate_type="blue", plate_type_id=0),
            PlateCandidate(text="娲38869D", confidence=0.88, frame_index=78, plate_type="green_new_energy", plate_type_id=3),
        ],
        frame_start=70,
        frame_end=82,
        window_frames=12,
        min_confidence=0.0,
    )

    assert summary is not None
    assert summary["plate_text"] == "娲38869D"
    assert summary["plate_source_frame_indices"] == [78]


def test_violation_plate_summary_returns_none_when_no_candidate_is_nearby():
    summary = _summarize_violation_plate_candidates(
        [
            PlateCandidate(text="浜珹12345", confidence=0.92, frame_index=12, plate_type="blue", plate_type_id=0),
            PlateCandidate(text="娲38869D", confidence=0.88, frame_index=78, plate_type="green_new_energy", plate_type_id=3),
        ],
        frame_start=180,
        frame_end=192,
        window_frames=12,
        min_confidence=0.0,
    )

    assert summary is None


def test_track_plate_ocr_gates_by_hits_and_interval():
    recognizer = _DummyRecognizer()
    state = _new_track_plate_ocr_state()
    detection = {
        "track_id": "T0001",
        "vehicle_type": "car",
        "track_hits": 2,
        "track_anchor": [20.0, 20.0],
        "footprint": [[10, 10], [30, 10], [30, 30], [10, 30]],
        "bbox": [10, 10, 30, 30],
        "is_violating": False,
    }

    assert not _should_run_track_plate_ocr(
        detection,
        frame_index=10,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )

    detection["track_hits"] = 3
    assert _should_run_track_plate_ocr(
        detection,
        frame_index=10,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )

    state["last_attempt_frame"] = 10
    assert not _should_run_track_plate_ocr(
        detection,
        frame_index=15,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )
    assert _should_run_track_plate_ocr(
        detection,
        frame_index=16,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )

    state["plate_text"] = "京A12345"
    state["plate_confidence"] = 0.82
    state["plate_support_count"] = 2
    state["plate_stable"] = True
    state["last_attempt_frame"] = 16
    assert not _should_run_track_plate_ocr(
        detection,
        frame_index=30,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )
    assert _should_run_track_plate_ocr(
        detection,
        frame_index=34,
        ocr_state=state,
        region_specs=[],
        recognizer=recognizer,
    )


def test_track_plate_ocr_keeps_track_relevant_after_roi_contact():
    recognizer = _DummyRecognizer()
    state = _new_track_plate_ocr_state()
    region = SimpleNamespace(
        polygon=np.asarray([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
        points=[[0, 0], [100, 0], [100, 100], [0, 100]],
    )
    inside_detection = {
        "track_id": "T0002",
        "vehicle_type": "truck",
        "track_hits": 3,
        "track_anchor": [20.0, 20.0],
        "footprint": [[12, 12], [28, 12], [28, 28], [12, 28]],
        "bbox": [12, 12, 28, 28],
        "is_violating": False,
    }

    assert _should_run_track_plate_ocr(
        inside_detection,
        frame_index=12,
        ocr_state=state,
        region_specs=[region],
        recognizer=recognizer,
    )

    state["last_attempt_frame"] = 12
    outside_detection = dict(inside_detection)
    outside_detection["track_anchor"] = [180.0, 180.0]
    outside_detection["footprint"] = [[170, 170], [190, 170], [190, 190], [170, 190]]
    outside_detection["bbox"] = [170, 170, 190, 190]
    assert _should_run_track_plate_ocr(
        outside_detection,
        frame_index=18,
        ocr_state=state,
        region_specs=[region],
        recognizer=recognizer,
    )


def test_process_frame_can_skip_intermediate_render():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    result, rendered = process_frame(
        frame,
        lane_detector=None,
        vehicle_detector=_DummyVehicleDetector(),
        violation_checker=_DummyViolationChecker(),
        render_output=False,
    )

    assert rendered is None
    assert result["vehicle_count"] == 0
