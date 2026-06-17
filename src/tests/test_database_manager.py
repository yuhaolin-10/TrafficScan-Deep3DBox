import os
import sqlite3
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from services.database_manager import DatabaseManager


def test_init_db_removes_obsolete_scene_config_tables(tmp_path):
    db_path = tmp_path / "traffic_scan.db"

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE scene_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL
            );
            CREATE TABLE scene_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_profile_id INTEGER NOT NULL,
                region_name TEXT NOT NULL
            );
            CREATE TABLE count_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_profile_id INTEGER NOT NULL,
                rule_name TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    manager = DatabaseManager(db_path)
    manager.close()

    conn = sqlite3.connect(db_path)
    try:
        table_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()

    assert "scene_profiles" not in table_names
    assert "scene_regions" not in table_names
    assert "count_rules" not in table_names


def test_start_record_persists_video_media_type(tmp_path):
    db_path = tmp_path / "traffic_scan.db"
    manager = DatabaseManager(db_path)
    try:
        record_id = manager.start_record(tmp_path / "sample.mp4", media_type=DatabaseManager.MEDIA_VIDEO)
    finally:
        manager.close()

    assert record_id > 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT media_type, status FROM records WHERE id = ?", (record_id,)).fetchone()
    finally:
        conn.close()

    assert row == (DatabaseManager.MEDIA_VIDEO, DatabaseManager.STATUS_RUNNING)


def test_complete_record_success_persists_video_summary_and_events(tmp_path):
    db_path = tmp_path / "traffic_scan.db"
    source_path = tmp_path / "sample.mp4"
    processed_path = tmp_path / "sample_processed.mp4"
    preview_frame_path = tmp_path / "sample_preview.jpg"
    preview_metadata_path = tmp_path / "sample_preview_frames.jsonl"

    manager = DatabaseManager(db_path)
    try:
        record_id = manager.start_record(source_path, media_type=DatabaseManager.MEDIA_VIDEO)
        result = {
            "media_type": "video",
            "status": "processed",
            "original_path": str(source_path),
            "processed_video_path": str(processed_path),
            "preview_frame_path": str(preview_frame_path),
            "preview_metadata_path": str(preview_metadata_path),
            "timestamp": "2026-04-19 10:00:00",
            "frame_count": 120,
            "processed_frame_count": 30,
            "expected_processed_frame_count": 30,
            "fps": 25.0,
            "output_fps": 5.0,
            "duration_s": 4.8,
            "image_width": 1280,
            "image_height": 720,
            "preview_frame_index": 60,
            "codec": "mp4v",
            "frame_stride": 5,
            "violation_plate_window_frames": 15,
            "violation_plate_window_s": 3.0,
            "lane_source": "auto",
            "lane_polygons": [[[0, 0], [20, 0], [20, 20], [0, 20]]],
            "lane_polygon_frame_index": 8,
            "count_line_source": "scene_profile",
            "count_line_count": 1,
            "count_line_names": ["northbound"],
            "scene_region_count": 2,
            "region_rule_event_count": 1,
            "region_rule_no_parking_count": 1,
            "region_rule_no_non_motor_count": 0,
            "region_rule_no_wrong_way_count": 0,
            "frames_with_region_rule_violation": 1,
            "region_rule_events": [
                {
                    "track_id": "T0001",
                    "vehicle_type": "car",
                    "region_id": "region_01",
                    "region_name": "No Parking Zone",
                    "rule_type": "no_parking",
                    "rule_label": "No Parking",
                    "frame_index": 12,
                    "timestamp_s": 2.4,
                }
            ],
            "total_vehicle_instances": 40,
            "max_vehicle_count": 3,
            "frames_with_violation": 2,
            "total_violation_instances": 2,
            "total_plate_ocr_attempt_count": 4,
            "total_plate_ocr_success_count": 2,
            "violating_track_count": 1,
            "violating_track_plate_count": 1,
            "unread_violating_track_count": 0,
            "violating_track_plates": [
                {
                    "track_id": "T0001",
                    "vehicle_type": "car",
                    "plate_text": "TEST123",
                    "plate_confidence": 0.93,
                    "plate_support_count": 2,
                    "plate_type": "blue",
                    "source_frame_indices": [11, 12],
                    "violation_labels": ["No Parking"],
                    "max_violation_ratio": 1.0,
                    "first_violation_frame": 10,
                    "last_violation_frame": 12,
                }
            ],
            "track_count": 3,
            "confirmed_track_count": 2,
            "moving_track_count": 2,
            "traffic_count_total": 1,
            "traffic_count_forward": 1,
            "traffic_count_backward": 0,
            "frames_with_count_event": 1,
            "count_events": [
                {
                    "track_id": "T0002",
                    "rule_name": "northbound",
                    "direction": "forward",
                    "frame_index": 16,
                    "timestamp_s": 3.2,
                }
            ],
            "any_violation": True,
        }
        manager.complete_record_success(record_id, result)
    finally:
        manager.close()

    conn = sqlite3.connect(db_path)
    try:
        record_row = conn.execute(
            "SELECT media_type, status, processed_path FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
        video_row = conn.execute(
            """
            SELECT preview_metadata_path, frame_count, count_line_names_json, region_rule_event_count, any_violation
            FROM video_results
            WHERE record_id = ?
            """,
            (record_id,),
        ).fetchone()
        lane_count = conn.execute(
            "SELECT COUNT(*) FROM lane_segments WHERE record_id = ?",
            (record_id,),
        ).fetchone()[0]
        track_event_rows = conn.execute(
            "SELECT event_type, track_id, start_frame, end_frame FROM track_events WHERE record_id = ? ORDER BY id",
            (record_id,),
        ).fetchall()
        plate_row = conn.execute(
            "SELECT plate_text, confidence, source_frame_index FROM plate_reads",
        ).fetchone()
    finally:
        conn.close()

    assert record_row == (DatabaseManager.MEDIA_VIDEO, DatabaseManager.STATUS_DONE, str(processed_path))
    assert video_row[0] == str(preview_metadata_path)
    assert video_row[1] == 120
    assert video_row[2] == '["northbound"]'
    assert video_row[3] == 1
    assert video_row[4] == 1
    assert lane_count == 1
    assert track_event_rows == [
        ("region_rule:no_parking", "T0001", 12, 12),
        ("traffic_count:forward", "T0002", 16, 16),
        ("violation_track", "T0001", 10, 12),
    ]
    assert plate_row == ("TEST123", 0.93, 11)
