import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from services.region_rule_engine import (
    RULE_NO_NON_MOTOR,
    RULE_NO_PARKING,
    RULE_NO_WRONG_WAY,
    RegionRuleEngine,
)


REGION_POLYGON = [[0, 0], [100, 0], [100, 100], [0, 100]]


def _scene_region(rule_type, *, params=None, direction_line=None):
    return [
        {
            "region_id": "region_01",
            "name": "Region 1",
            "points": REGION_POLYGON,
            "direction_line": list(direction_line or []),
            "rule_bindings": [
                {
                    "rule_type": rule_type,
                    "enabled": True,
                    "params": dict(params or {}),
                }
            ],
        }
    ]


def _detection(
    track_id,
    vehicle_type,
    anchor,
    *,
    previous_anchor=None,
    track_hits=2,
    footprint=None,
):
    anchor_x = float(anchor[0])
    anchor_y = float(anchor[1])
    footprint_points = list(footprint or [])
    if not footprint_points:
        footprint_points = [
            [anchor_x - 10.0, anchor_y + 6.0],
            [anchor_x + 10.0, anchor_y + 6.0],
            [anchor_x + 10.0, anchor_y - 6.0],
            [anchor_x - 10.0, anchor_y - 6.0],
        ]
    payload = {
        "track_id": track_id,
        "vehicle_type": vehicle_type,
        "track_anchor": [anchor_x, anchor_y],
        "track_hits": int(track_hits),
        "footprint": [[float(point[0]), float(point[1])] for point in footprint_points],
    }
    if previous_anchor is not None:
        payload["track_previous_anchor"] = [float(previous_anchor[0]), float(previous_anchor[1])]
    return payload


def test_region_rule_engine_triggers_no_non_motor_after_consecutive_frames():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_NON_MOTOR,
            params={"target_classes": ["bicycle", "person"], "min_consecutive_frames": 2, "min_confirmed_hits": 1},
        )
    )

    detections, events, summary = engine.apply(
        [_detection("T0001", "bicycle", (30, 40), track_hits=1)],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_NON_MOTOR] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0001", "bicycle", (32, 40), previous_anchor=(30, 40), track_hits=2)],
        frame_index=1,
        timestamp_s=1.0,
    )
    assert len(events) == 1
    assert events[0]["rule_type"] == RULE_NO_NON_MOTOR
    assert summary[RULE_NO_NON_MOTOR] == 1
    assert detections[0]["is_violating"] is True


def test_region_rule_engine_treats_person_as_non_motor_for_rule():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_NON_MOTOR,
            params={"target_classes": ["person"], "min_consecutive_frames": 2, "min_confirmed_hits": 1},
        )
    )

    detections, events, summary = engine.apply(
        [_detection("T0101", "person", (30, 40), track_hits=1)],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_NON_MOTOR] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0101", "person", (32, 40), previous_anchor=(30, 40), track_hits=2)],
        frame_index=1,
        timestamp_s=1.0,
    )
    assert len(events) == 1
    assert events[0]["rule_type"] == RULE_NO_NON_MOTOR
    assert summary[RULE_NO_NON_MOTOR] == 1
    assert detections[0]["is_violating"] is True


def test_region_rule_engine_triggers_no_parking_from_stationary_duration():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_PARKING,
            params={"min_stop_seconds": 5.0, "max_speed_px_per_s": 4.0, "min_confirmed_hits": 2},
        )
    )

    detections, events, summary = engine.apply(
        [_detection("T0002", "car", (50, 60), track_hits=2)],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_PARKING] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0002", "car", (51, 60), previous_anchor=(50, 60), track_hits=3)],
        frame_index=1,
        timestamp_s=2.0,
    )
    assert events == []
    assert summary[RULE_NO_PARKING] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0002", "car", (52, 60), previous_anchor=(51, 60), track_hits=4)],
        frame_index=2,
        timestamp_s=7.5,
    )
    assert len(events) == 1
    assert events[0]["rule_type"] == RULE_NO_PARKING
    assert summary[RULE_NO_PARKING] == 1
    assert detections[0]["is_violating"] is True


def test_region_rule_engine_triggers_no_wrong_way_against_allowed_direction():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_WRONG_WAY,
            params={
                "min_consecutive_frames": 2,
                "min_direction_distance_px": 10.0,
                "wrong_way_dot_threshold": -0.2,
                "min_roi_overlap_ratio": 0.2,
                "min_confirmed_hits": 2,
            },
            direction_line=[[20, 50], [80, 50]],
        )
    )

    detections, events, summary = engine.apply(
        [_detection("T0003", "car", (80, 50), track_hits=2)],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0003", "car", (60, 50), previous_anchor=(80, 50), track_hits=3)],
        frame_index=1,
        timestamp_s=1.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [_detection("T0003", "car", (35, 50), previous_anchor=(60, 50), track_hits=4)],
        frame_index=2,
        timestamp_s=2.0,
    )
    assert len(events) == 1
    assert events[0]["rule_type"] == RULE_NO_WRONG_WAY
    assert summary[RULE_NO_WRONG_WAY] == 1
    assert detections[0]["is_violating"] is True


def test_region_rule_engine_wrong_way_requires_enough_roi_overlap():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_WRONG_WAY,
            params={
                "min_consecutive_frames": 2,
                "min_direction_distance_px": 10.0,
                "wrong_way_dot_threshold": -0.2,
                "min_roi_overlap_ratio": 0.8,
                "min_confirmed_hits": 2,
            },
            direction_line=[[20, 50], [80, 50]],
        )
    )

    detections, events, summary = engine.apply(
        [
            _detection(
                "T0004",
                "car",
                (95, 50),
                track_hits=2,
                footprint=[[80, 56], [120, 56], [120, 44], [80, 44]],
            )
        ],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [
            _detection(
                "T0004",
                "car",
                (75, 50),
                previous_anchor=(95, 50),
                track_hits=3,
                footprint=[[60, 56], [110, 56], [110, 44], [60, 44]],
            )
        ],
        frame_index=1,
        timestamp_s=1.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False


def test_region_rule_engine_wrong_way_uses_overlap_even_if_anchor_is_outside():
    engine = RegionRuleEngine(
        _scene_region(
            RULE_NO_WRONG_WAY,
            params={
                "min_consecutive_frames": 2,
                "min_direction_distance_px": 10.0,
                "wrong_way_dot_threshold": -0.2,
                "min_roi_overlap_ratio": 0.3,
                "min_confirmed_hits": 2,
            },
            direction_line=[[20, 50], [80, 50]],
        )
    )

    detections, events, summary = engine.apply(
        [
            _detection(
                "T0005",
                "car",
                (108, 50),
                track_hits=2,
                footprint=[[80, 56], [110, 56], [110, 44], [80, 44]],
            )
        ],
        frame_index=0,
        timestamp_s=0.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [
            _detection(
                "T0005",
                "car",
                (84, 50),
                previous_anchor=(108, 50),
                track_hits=3,
                footprint=[[60, 56], [100, 56], [100, 44], [60, 44]],
            )
        ],
        frame_index=1,
        timestamp_s=1.0,
    )
    assert events == []
    assert summary[RULE_NO_WRONG_WAY] == 0
    assert detections[0].get("is_violating", False) is False

    detections, events, summary = engine.apply(
        [
            _detection(
                "T0005",
                "car",
                (60, 50),
                previous_anchor=(84, 50),
                track_hits=4,
                footprint=[[36, 56], [76, 56], [76, 44], [36, 44]],
            )
        ],
        frame_index=2,
        timestamp_s=2.0,
    )
    assert len(events) == 1
    assert events[0]["rule_type"] == RULE_NO_WRONG_WAY
    assert summary[RULE_NO_WRONG_WAY] == 1
    assert detections[0]["is_violating"] is True
