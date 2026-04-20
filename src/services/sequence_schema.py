from __future__ import annotations


def ensure_sequence_schema(conn) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL UNIQUE,
            fps REAL DEFAULT 0,
            source_path TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_regions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_profile_id INTEGER NOT NULL,
            region_name TEXT NOT NULL,
            region_type TEXT NOT NULL,
            points_json TEXT NOT NULL,
            region_source TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(scene_profile_id) REFERENCES scene_profiles(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS count_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_profile_id INTEGER NOT NULL,
            rule_name TEXT NOT NULL,
            direction_mode TEXT NOT NULL,
            start_x REAL NOT NULL,
            start_y REAL NOT NULL,
            end_x REAL NOT NULL,
            end_y REAL NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(scene_profile_id) REFERENCES scene_profiles(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER NOT NULL UNIQUE,
            preview_frame_path TEXT,
            preview_metadata_path TEXT,
            frame_count INTEGER NOT NULL DEFAULT 0,
            processed_frame_count INTEGER NOT NULL DEFAULT 0,
            expected_processed_frame_count INTEGER NOT NULL DEFAULT 0,
            fps REAL DEFAULT 0,
            output_fps REAL DEFAULT 0,
            duration_s REAL DEFAULT 0,
            image_width INTEGER DEFAULT 0,
            image_height INTEGER DEFAULT 0,
            preview_frame_index INTEGER,
            codec TEXT,
            frame_stride INTEGER NOT NULL DEFAULT 1,
            violation_plate_window_frames INTEGER NOT NULL DEFAULT 0,
            violation_plate_window_s REAL DEFAULT 0,
            lane_source TEXT,
            lane_polygon_frame_index INTEGER,
            count_line_source TEXT,
            count_line_count INTEGER NOT NULL DEFAULT 0,
            count_line_names_json TEXT NOT NULL DEFAULT '[]',
            scene_region_count INTEGER NOT NULL DEFAULT 0,
            region_rule_event_count INTEGER NOT NULL DEFAULT 0,
            region_rule_no_parking_count INTEGER NOT NULL DEFAULT 0,
            region_rule_no_non_motor_count INTEGER NOT NULL DEFAULT 0,
            region_rule_no_wrong_way_count INTEGER NOT NULL DEFAULT 0,
            frames_with_region_rule_violation INTEGER NOT NULL DEFAULT 0,
            total_vehicle_instances INTEGER NOT NULL DEFAULT 0,
            max_vehicle_count INTEGER NOT NULL DEFAULT 0,
            frames_with_violation INTEGER NOT NULL DEFAULT 0,
            total_violation_instances INTEGER NOT NULL DEFAULT 0,
            total_plate_ocr_attempt_count INTEGER NOT NULL DEFAULT 0,
            total_plate_ocr_success_count INTEGER NOT NULL DEFAULT 0,
            violating_track_count INTEGER NOT NULL DEFAULT 0,
            violating_track_plate_count INTEGER NOT NULL DEFAULT 0,
            unread_violating_track_count INTEGER NOT NULL DEFAULT 0,
            track_count INTEGER NOT NULL DEFAULT 0,
            confirmed_track_count INTEGER NOT NULL DEFAULT 0,
            moving_track_count INTEGER NOT NULL DEFAULT 0,
            traffic_count_total INTEGER NOT NULL DEFAULT 0,
            traffic_count_forward INTEGER NOT NULL DEFAULT 0,
            traffic_count_backward INTEGER NOT NULL DEFAULT 0,
            frames_with_count_event INTEGER NOT NULL DEFAULT 0,
            any_violation INTEGER NOT NULL DEFAULT 0 CHECK(any_violation IN (0,1)),
            summary_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS track_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            track_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            start_frame INTEGER,
            end_frame INTEGER,
            snapshot_path TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS plate_reads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            track_event_id INTEGER,
            plate_text TEXT,
            confidence REAL DEFAULT 0,
            crop_path TEXT,
            source_frame_index INTEGER,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(detection_id) REFERENCES detections(id) ON DELETE CASCADE,
            FOREIGN KEY(track_event_id) REFERENCES track_events(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scene_regions_profile ON scene_regions(scene_profile_id, region_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_count_rules_profile ON count_rules(scene_profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_results_record ON video_results(record_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_results_violation ON video_results(any_violation, created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_events_record ON track_events(record_id, track_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_events_type ON track_events(record_id, event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_detection ON plate_reads(detection_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_track_event ON plate_reads(track_event_id)")
