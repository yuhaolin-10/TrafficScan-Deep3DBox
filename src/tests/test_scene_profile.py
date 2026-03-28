import json
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from services.scene_profile import (
    CountLine,
    PolygonRegion,
    RegionRuleBinding,
    SceneProfile,
    load_scene_profile,
    save_scene_profile,
)


def test_scene_profile_round_trip(tmp_path):
    profile = SceneProfile(
        camera_id="cam_01",
        fps=25.0,
        source_path="data/highway_cam_01",
        parking_regions=[
            PolygonRegion(
                name="manual_emergency_lane",
                region_type="parking_roi",
                points=[[10, 10], [50, 10], [48, 42], [12, 43]],
                region_id="region_a",
                direction_line=[[10, 26], [50, 26]],
                rule_bindings=[
                    RegionRuleBinding(
                        rule_type="no_parking",
                        enabled=True,
                        params={"min_stop_seconds": 12.0},
                    )
                ],
            ),
            PolygonRegion(
                name="manual_parking_slot",
                region_type="parking_roi",
                points=[[60, 15], [100, 18], [98, 52], [62, 50]],
                region_id="region_b",
                rule_bindings=[
                    RegionRuleBinding(
                        rule_type="no_wrong_way",
                        enabled=True,
                        params={"min_consecutive_frames": 4},
                    )
                ],
            ),
        ],
        count_lines=[
            CountLine(name="main_count", start=[0, 20], end=[100, 20], direction_mode="forward")
        ],
    )
    target = tmp_path / "scene_profile.json"
    save_scene_profile(target, profile)
    loaded = load_scene_profile(target)

    assert loaded.camera_id == "cam_01"
    assert loaded.fps == 25.0
    assert loaded.parking_regions[0].region_type == "parking_roi"
    assert len(loaded.parking_regions) == 2
    assert loaded.parking_regions[1].name == "manual_parking_slot"
    assert loaded.parking_regions[0].region_id == "region_a"
    assert loaded.parking_regions[0].direction_line == [[10.0, 26.0], [50.0, 26.0]]
    assert loaded.parking_regions[0].rule_bindings[0].rule_type == "no_parking"
    assert loaded.parking_regions[0].rule_bindings[0].params["min_stop_seconds"] == 12.0
    assert loaded.parking_regions[1].rule_bindings[0].rule_type == "no_wrong_way"
    assert loaded.count_lines[0].direction_mode == "forward"
