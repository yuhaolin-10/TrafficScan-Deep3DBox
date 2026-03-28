import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.object_tracker import ObjectTracker


def test_tracker_reuses_track_id_across_neighboring_frames():
    tracker = ObjectTracker(max_match_distance=60.0, min_confirm_hits=2)

    frame_0 = tracker.update(
        [
            {
                "vehicle_type": "car",
                "bbox": [10, 10, 50, 50],
                "footprint": [[12, 48], [48, 48], [44, 34], [16, 34]],
            }
        ],
        frame_index=0,
    )
    frame_1 = tracker.update(
        [
            {
                "vehicle_type": "car",
                "bbox": [18, 14, 58, 54],
                "footprint": [[20, 52], [56, 52], [52, 38], [24, 38]],
            }
        ],
        frame_index=1,
    )

    assert len(frame_0) == 1
    assert len(frame_1) == 1
    assert frame_0[0]["track_id"] == frame_1[0]["track_id"]
    assert frame_1[0]["track_confirmed"] is True
    assert frame_1[0]["track_hits"] == 2
    assert frame_1[0]["track_previous_anchor"] == frame_0[0]["track_anchor"]


def test_tracker_creates_new_track_for_far_detection():
    tracker = ObjectTracker(max_match_distance=25.0, min_confirm_hits=2)

    frame_0 = tracker.update([{"vehicle_type": "car", "bbox": [10, 10, 30, 30]}], frame_index=0)
    frame_1 = tracker.update([{"vehicle_type": "car", "bbox": [160, 160, 190, 190]}], frame_index=1)

    assert frame_0[0]["track_id"] != frame_1[0]["track_id"]
    assert tracker.track_count() == 2


def test_tracker_does_not_reuse_track_across_motor_vehicle_and_person():
    tracker = ObjectTracker(max_match_distance=120.0, min_confirm_hits=2)

    frame_0 = tracker.update(
        [{"vehicle_type": "bus", "bbox": [100, 100, 220, 260]}],
        frame_index=0,
    )
    frame_1 = tracker.update(
        [{"vehicle_type": "person", "bbox": [130, 130, 180, 250]}],
        frame_index=1,
    )

    assert frame_0[0]["track_id"] != frame_1[0]["track_id"]
    assert tracker.track_count() == 2


def test_tracker_uses_stable_vehicle_type_within_same_family():
    tracker = ObjectTracker(max_match_distance=120.0, min_confirm_hits=2)

    frame_0 = tracker.update(
        [{"vehicle_type": "bus", "bbox": [100, 100, 220, 260]}],
        frame_index=0,
    )
    frame_1 = tracker.update(
        [{"vehicle_type": "truck", "bbox": [106, 104, 226, 264]}],
        frame_index=1,
    )
    frame_2 = tracker.update(
        [{"vehicle_type": "truck", "bbox": [112, 108, 232, 268]}],
        frame_index=2,
    )

    assert frame_0[0]["track_id"] == frame_1[0]["track_id"] == frame_2[0]["track_id"]
    assert frame_1[0]["vehicle_type_raw"] == "truck"
    assert frame_1[0]["vehicle_type"] == "bus"
    assert frame_1[0]["track_vehicle_type"] == "bus"
    assert frame_2[0]["vehicle_type"] == "truck"
    assert frame_2[0]["track_vehicle_type_support"] == 2
