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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_events_record ON track_events(record_id, track_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_detection ON plate_reads(detection_id)")
