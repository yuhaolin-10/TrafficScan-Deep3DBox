import os
import sys

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.lane_geometry import estimate_lane_direction


def test_estimate_lane_direction_returns_anchor_arrows():
    lane_polygons = [
        [[420, 460], [560, 460], [420, 80], [360, 80]],
        [[520, 460], [640, 460], [520, 90], [470, 90]],
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[500, 420], [560, 360]],
    )

    assert result["available"] is True
    assert result["global_arrow"] is None
    assert result["global_confidence"] > 0.2
    assert len(result["lane_arrows"]) == 2
    assert len(result["anchor_arrows"]) >= 1
    assert len(result["anchor_priors"]) >= 1

    direction = np.asarray(result["global_direction"], dtype=np.float32)
    assert direction.shape == (2,)
    assert direction[1] < 0.0
    assert result["anchor_arrows"][0]["anchor_index"] == 0
    assert len(result["anchor_arrows"][0]["direction"]) == 2
    assert result["anchor_arrows"][0]["fit_mode"] in {"line", "curve"}
    assert result["lane_arrows"][0]["direction"][1] < 0.0
    assert result["anchor_priors"][0]["component_count"] >= 2


def test_estimate_lane_direction_prefers_smoothed_boundary_tangent():
    lane_polygons = [
        [
            [420, 460],
            [560, 460],
            [540, 320],
            [550, 285],
            [495, 80],
            [360, 80],
            [410, 285],
            [390, 320],
        ]
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[398, 420]],
    )

    assert result["available"] is True
    assert len(result["anchor_arrows"]) == 1

    arrow = result["anchor_arrows"][0]
    direction = np.asarray(arrow["direction"], dtype=np.float32)
    assert direction[1] < -0.7
    assert direction[0] < -0.05
    assert arrow["confidence"] > 0.35


def test_estimate_lane_direction_bridges_split_lane_segments():
    lane_polygons = [
        [[350, 470], [420, 470], [390, 330], [330, 330]],
        [[335, 250], [390, 250], [365, 90], [320, 90]],
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[345, 305]],
    )

    assert result["available"] is True
    assert len(result["anchor_arrows"]) == 1

    arrow = result["anchor_arrows"][0]
    direction = np.asarray(arrow["direction"], dtype=np.float32)
    assert direction[1] < -0.7
    assert arrow["anchor_distance"] < 35.0
    assert arrow["selection_score"] > 0.35


def test_estimate_lane_direction_can_prefer_smoother_single_side():
    lane_polygons = [
        [
            [310, 470],
            [430, 470],
            [425, 430],
            [424, 390],
            [423, 350],
            [422, 310],
            [421, 270],
            [420, 230],
            [419, 190],
            [418, 150],
            [417, 110],
            [416, 80],
            [350, 80],
            [330, 120],
            [360, 150],
            [328, 190],
            [362, 230],
            [326, 270],
            [361, 310],
            [324, 350],
            [360, 390],
            [322, 430],
            [350, 470],
        ]
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[405, 360]],
    )

    assert result["available"] is True
    assert len(result["anchor_arrows"]) == 1

    arrow = result["anchor_arrows"][0]
    assert arrow["preferred_side"] in {"left", "right"}
    assert arrow["preferred_side"] == arrow["curve_side"]
    assert arrow["side_preference_bonus"] >= 0.0
    assert arrow["side_quality_score"] > 0.35


def test_estimate_lane_direction_builds_neighbor_aware_anchor_prior():
    lane_polygons = [
        [[120, 470], [210, 470], [180, 90], [95, 90]],
        [[260, 470], [350, 470], [320, 90], [235, 90]],
        [[400, 470], [490, 470], [460, 90], [375, 90]],
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[305, 390]],
    )

    assert result["available"] is True
    assert len(result["lane_arrows"]) == 3
    assert len(result["anchor_priors"]) == 1

    lane_arrows = {int(item["polygon_index"]): item for item in result["lane_arrows"]}
    center_lane = next(item for item in lane_arrows.values() if item["lane_rank"] == 1)
    assert sorted(center_lane["neighbor_polygon_indices"]) == sorted(
        [
            lane_arrows[idx]["polygon_index"]
            for idx in lane_arrows
            if lane_arrows[idx]["lane_rank"] in {0, 2}
        ]
    )

    prior = result["anchor_priors"][0]
    direction = np.asarray(prior["direction"], dtype=np.float32)
    assert direction.shape == (2,)
    assert direction[1] < -0.7
    assert prior["polygon_index"] == center_lane["polygon_index"]
    assert len(prior["neighbor_polygon_indices"]) == 2
    assert prior["component_count"] >= 3
    assert prior["confidence"] > 0.3
    for component in prior["components"]:
        comp_dir = np.asarray(component["direction"], dtype=np.float32)
        assert comp_dir.shape == (2,)
        assert np.all(np.isfinite(comp_dir))


def test_estimate_lane_direction_reports_lane_based_vanishing_point():
    lane_polygons = [
        [[140, 470], [240, 470], [300, 100], [260, 100]],
        [[280, 470], [380, 470], [380, 100], [340, 100]],
    ]

    result = estimate_lane_direction(
        lane_polygons,
        image_shape=(480, 640, 3),
        anchors=[[250, 410]],
    )

    assert result["available"] is True
    assert result["vanishing_point"] is not None
    assert result["vanishing_point_confidence"] > 0.25
    vp = np.asarray(result["vanishing_point"], dtype=np.float32)
    assert vp.shape == (2,)
    assert 250.0 <= float(vp[0]) <= 390.0
    assert -260.0 <= float(vp[1]) <= 170.0
