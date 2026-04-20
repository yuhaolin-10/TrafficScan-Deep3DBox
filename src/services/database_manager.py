import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from .sequence_schema import ensure_sequence_schema
except Exception:
    try:
        from services.sequence_schema import ensure_sequence_schema
    except Exception:
        from sequence_schema import ensure_sequence_schema


class DatabaseManager:
    """
    Database manager for persisted processing results.

    records stores run lifecycle for both image and video tasks.
    Image details live in lane_segments / detections.
    Video details live in video_results / track_events / plate_reads.
    """

    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"

    MEDIA_IMAGE = "image"
    MEDIA_VIDEO = "video"

    VIDEO_EVENT_REGION_RULE = "region_rule"
    VIDEO_EVENT_TRAFFIC_COUNT = "traffic_count"
    VIDEO_EVENT_VIOLATION_TRACK = "violation_track"

    VIOLATION_TYPE_META = {
        "emergency lane occupation": ("EMERGENCY_LANE", "Emergency Lane Occupation"),
        "占用应急车道": ("EMERGENCY_LANE", "Emergency Lane Occupation"),
        "no non motor": ("NO_NON_MOTOR", "No Non-Motor Vehicles"),
        "no non-motor": ("NO_NON_MOTOR", "No Non-Motor Vehicles"),
        "禁止非机动车": ("NO_NON_MOTOR", "No Non-Motor Vehicles"),
        "wrong way": ("NO_WRONG_WAY", "No Wrong Way"),
        "no wrong way": ("NO_WRONG_WAY", "No Wrong Way"),
        "禁止逆行": ("NO_WRONG_WAY", "No Wrong Way"),
        "no parking": ("NO_PARKING", "No Parking"),
        "parking": ("NO_PARKING", "No Parking"),
        "禁止停车": ("NO_PARKING", "No Parking"),
    }

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            cursor = self.conn.cursor()

            # Legacy table kept for backward compatibility with old scripts.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS violation_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    location TEXT,
                    violation_type TEXT,
                    image_path TEXT,
                    is_reviewed INTEGER DEFAULT 0
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    original_path TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    media_type TEXT NOT NULL DEFAULT 'image' CHECK(media_type IN ('image','video')),
                    status TEXT NOT NULL CHECK(status IN ('running','done','failed')),
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    processed_path TEXT,
                    error_message TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(id)
                )
                """
            )
            self._ensure_column(cursor, "records", "media_type", "TEXT NOT NULL DEFAULT 'image'")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS lane_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    lane_type_code TEXT NOT NULL,
                    polygons_json TEXT NOT NULL,
                    mask_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS violation_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    footprint_json TEXT NOT NULL,
                    corners_json TEXT,
                    is_violating INTEGER NOT NULL CHECK(is_violating IN (0,1)),
                    violation_ratio REAL NOT NULL,
                    violation_type_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE,
                    FOREIGN KEY(violation_type_id) REFERENCES violation_types(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    result TEXT NOT NULL CHECK(result IN ('pass','reject')),
                    reviewer TEXT,
                    reviewed_at TEXT,
                    comment TEXT,
                    FOREIGN KEY(detection_id) REFERENCES detections(id) ON DELETE CASCADE
                )
                """
            )

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_image_id ON records(image_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_status_created ON records(status, created_at)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_media_status_created ON records(media_type, status, created_at)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lane_segments_record_id ON lane_segments(record_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_record_id ON detections(record_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_violation ON detections(is_violating, violation_type_id)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_detection_id ON reviews(detection_id)")

            self._ensure_violation_type(cursor, "EMERGENCY_LANE", "Emergency Lane Occupation")
            self._ensure_violation_type(cursor, "NO_NON_MOTOR", "No Non-Motor Vehicles")
            self._ensure_violation_type(cursor, "NO_WRONG_WAY", "No Wrong Way")
            self._ensure_violation_type(cursor, "NO_PARKING", "No Parking")
            ensure_sequence_schema(self.conn)

            self.conn.commit()
            print(f"[DB] Connected: {self.db_path}")
        except Exception as exc:
            print(f"[DB] Initialization failed: {exc}")

    def _ensure_column(self, cursor, table_name: str, column_name: str, definition: str) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {str(row[1]) for row in cursor.fetchall()}
        if column_name in existing_columns:
            return
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _normalize_media_type(self, media_type) -> str:
        if str(media_type or "").strip().lower() == self.MEDIA_VIDEO:
            return self.MEDIA_VIDEO
        return self.MEDIA_IMAGE

    def _safe_int(self, value, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _json_text(self, payload, *, default: str) -> str:
        if payload is None:
            return str(default)
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(default)

    def _compute_content_hash(self, file_path):
        path = Path(file_path)
        hasher = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            # Fallback keeps path mappable even when file cannot be read.
            hasher.update(f"path:{path}".encode("utf-8", errors="ignore"))
            return hasher.hexdigest()

    def _ensure_violation_type(self, cursor, code, name):
        cursor.execute("SELECT id FROM violation_types WHERE code = ?", (code,))
        row = cursor.fetchone()
        if row:
            return int(row[0])
        cursor.execute(
            "INSERT INTO violation_types (code, name) VALUES (?, ?)",
            (str(code), str(name)),
        )
        return int(cursor.lastrowid)

    def _violation_type_meta_for_detection(self, detection):
        labels = []
        for item in list(detection.get("rule_violations", []) or []):
            label = str(dict(item or {}).get("rule_label", "") or "").strip()
            if label:
                labels.append(label)
        raw_violation_type = str(detection.get("violation_type", "") or "").strip()
        if raw_violation_type:
            labels.extend([part.strip() for part in raw_violation_type.split(",") if part.strip()])
        for label in labels:
            normalized = label.lower().replace("-", " ").replace("_", " ")
            normalized = " ".join(normalized.split())
            meta = self.VIOLATION_TYPE_META.get(normalized)
            if meta is not None:
                return meta
        return ("EMERGENCY_LANE", "Emergency Lane Occupation")

    def _upsert_image(self, cursor, original_path):
        now = self._now()
        normalized = str(Path(original_path))
        content_hash = self._compute_content_hash(normalized)
        cursor.execute("SELECT id FROM images WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        if row:
            image_id = int(row[0])
            cursor.execute(
                "UPDATE images SET original_path = ?, imported_at = ? WHERE id = ?",
                (normalized, now, image_id),
            )
            return image_id
        cursor.execute(
            "INSERT INTO images (content_hash, original_path, imported_at) VALUES (?, ?, ?)",
            (content_hash, normalized, now),
        )
        return int(cursor.lastrowid)

    def _insert_lane_segments(self, cursor, record_id: int, lane_polygons, *, now: str) -> None:
        polygons = list(lane_polygons or [])
        if not polygons:
            return
        cursor.execute(
            """
            INSERT INTO lane_segments (record_id, lane_type_code, polygons_json, mask_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(record_id),
                "EMERGENCY_LANE",
                self._json_text(polygons, default="[]"),
                None,
                now,
            ),
        )

    def _insert_detection(self, cursor, record_id: int, detection: dict, *, now: str) -> None:
        bbox = list(detection.get("bbox", [0, 0, 0, 0]))
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        is_violating = 1 if detection.get("is_violating") else 0
        violation_type_id = None
        if is_violating:
            violation_code, violation_name = self._violation_type_meta_for_detection(detection)
            violation_type_id = self._ensure_violation_type(cursor, violation_code, violation_name)

        cursor.execute(
            """
            INSERT INTO detections (
                record_id, vehicle_type, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                footprint_json, corners_json,
                is_violating, violation_ratio, violation_type_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(record_id),
                str(detection.get("vehicle_type", "vehicle")),
                self._safe_float(detection.get("confidence", 0.0)),
                self._safe_float(bbox[0]),
                self._safe_float(bbox[1]),
                self._safe_float(bbox[2]),
                self._safe_float(bbox[3]),
                self._json_text(detection.get("footprint", []), default="[]"),
                self._json_text(detection.get("corners_2d", []), default="[]"),
                int(is_violating),
                self._safe_float(detection.get("violation_ratio", 0.0)),
                violation_type_id,
                now,
            ),
        )

    def _insert_track_event(
        self,
        cursor,
        *,
        record_id: int,
        track_id: str,
        event_type: str,
        start_frame,
        end_frame,
        snapshot_path,
        metadata,
        now: str,
    ) -> int:
        cursor.execute(
            """
            INSERT INTO track_events (
                record_id, track_id, event_type, start_frame, end_frame, snapshot_path, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(record_id),
                str(track_id or ""),
                str(event_type or ""),
                None if start_frame is None else self._safe_int(start_frame),
                None if end_frame is None else self._safe_int(end_frame),
                str(snapshot_path or "").strip() or None,
                self._json_text(metadata, default="{}"),
                now,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_plate_read(self, cursor, *, track_event_id: int, payload: dict, now: str) -> None:
        source_frame_indices = [self._safe_int(item) for item in list(payload.get("source_frame_indices", []) or [])]
        cursor.execute(
            """
            INSERT INTO plate_reads (
                detection_id, track_event_id, plate_text, confidence, crop_path, source_frame_index, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                int(track_event_id),
                str(payload.get("plate_text", "") or ""),
                self._safe_float(payload.get("plate_confidence", 0.0)),
                None,
                source_frame_indices[0]
                if source_frame_indices
                else (
                    None
                    if payload.get("first_violation_frame") is None
                    else self._safe_int(payload.get("first_violation_frame"))
                ),
                self._json_text(
                    {
                        "track_id": str(payload.get("track_id", "") or ""),
                        "plate_support_count": self._safe_int(payload.get("plate_support_count", 0)),
                        "plate_type": str(payload.get("plate_type", "") or ""),
                        "source_frame_indices": source_frame_indices,
                        "violation_labels": [
                            str(item)
                            for item in list(payload.get("violation_labels", []) or [])
                            if str(item).strip()
                        ],
                        "max_violation_ratio": self._safe_float(payload.get("max_violation_ratio", 0.0)),
                        "first_violation_frame": (
                            None
                            if payload.get("first_violation_frame") is None
                            else self._safe_int(payload.get("first_violation_frame"))
                        ),
                        "last_violation_frame": (
                            None
                            if payload.get("last_violation_frame") is None
                            else self._safe_int(payload.get("last_violation_frame"))
                        ),
                    },
                    default="{}",
                ),
                now,
            ),
        )

    def _complete_image_record_success(self, cursor, record_id: int, result: dict, *, now: str) -> None:
        detections = list(result.get("detections", []))
        lane_polygons = list(result.get("violation_lane_polygons", []) or [])
        processed_path = str(result.get("processed_path", "")).strip()

        self._insert_lane_segments(cursor, record_id, lane_polygons, now=now)
        for detection in detections:
            self._insert_detection(cursor, record_id, dict(detection or {}), now=now)

        cursor.execute(
            """
            UPDATE records
            SET media_type = ?, status = ?, processed_at = ?, processed_path = ?, error_message = NULL
            WHERE id = ?
            """,
            (self.MEDIA_IMAGE, self.STATUS_DONE, now, processed_path, int(record_id)),
        )

    def _complete_video_record_success(self, cursor, record_id: int, result: dict, *, now: str) -> None:
        preview_frame_path = str(result.get("preview_frame_path", "") or "").strip()
        preview_metadata_path = str(result.get("preview_metadata_path", "") or "").strip()
        processed_video_path = str(result.get("processed_video_path", "") or "").strip()
        lane_polygons = list(result.get("lane_polygons", []) or [])
        self._insert_lane_segments(cursor, record_id, lane_polygons, now=now)

        cursor.execute(
            """
            INSERT INTO video_results (
                record_id,
                preview_frame_path,
                preview_metadata_path,
                frame_count,
                processed_frame_count,
                expected_processed_frame_count,
                fps,
                output_fps,
                duration_s,
                image_width,
                image_height,
                preview_frame_index,
                codec,
                frame_stride,
                violation_plate_window_frames,
                violation_plate_window_s,
                lane_source,
                lane_polygon_frame_index,
                count_line_source,
                count_line_count,
                count_line_names_json,
                scene_region_count,
                region_rule_event_count,
                region_rule_no_parking_count,
                region_rule_no_non_motor_count,
                region_rule_no_wrong_way_count,
                frames_with_region_rule_violation,
                total_vehicle_instances,
                max_vehicle_count,
                frames_with_violation,
                total_violation_instances,
                total_plate_ocr_attempt_count,
                total_plate_ocr_success_count,
                violating_track_count,
                violating_track_plate_count,
                unread_violating_track_count,
                track_count,
                confirmed_track_count,
                moving_track_count,
                traffic_count_total,
                traffic_count_forward,
                traffic_count_backward,
                frames_with_count_event,
                any_violation,
                summary_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(record_id),
                preview_frame_path,
                preview_metadata_path,
                self._safe_int(result.get("frame_count", 0)),
                self._safe_int(result.get("processed_frame_count", 0)),
                self._safe_int(result.get("expected_processed_frame_count", 0)),
                self._safe_float(result.get("fps", 0.0)),
                self._safe_float(result.get("output_fps", 0.0)),
                self._safe_float(result.get("duration_s", 0.0)),
                self._safe_int(result.get("image_width", 0)),
                self._safe_int(result.get("image_height", 0)),
                None if result.get("preview_frame_index") is None else self._safe_int(result.get("preview_frame_index")),
                str(result.get("codec", "") or ""),
                max(1, self._safe_int(result.get("frame_stride", 1), default=1)),
                self._safe_int(result.get("violation_plate_window_frames", 0)),
                self._safe_float(result.get("violation_plate_window_s", 0.0)),
                str(result.get("lane_source", "") or ""),
                None
                if result.get("lane_polygon_frame_index") is None
                else self._safe_int(result.get("lane_polygon_frame_index")),
                str(result.get("count_line_source", "") or ""),
                self._safe_int(result.get("count_line_count", 0)),
                self._json_text(result.get("count_line_names", []), default="[]"),
                self._safe_int(result.get("scene_region_count", 0)),
                self._safe_int(result.get("region_rule_event_count", 0)),
                self._safe_int(result.get("region_rule_no_parking_count", 0)),
                self._safe_int(result.get("region_rule_no_non_motor_count", 0)),
                self._safe_int(result.get("region_rule_no_wrong_way_count", 0)),
                self._safe_int(result.get("frames_with_region_rule_violation", 0)),
                self._safe_int(result.get("total_vehicle_instances", 0)),
                self._safe_int(result.get("max_vehicle_count", 0)),
                self._safe_int(result.get("frames_with_violation", 0)),
                self._safe_int(result.get("total_violation_instances", 0)),
                self._safe_int(result.get("total_plate_ocr_attempt_count", 0)),
                self._safe_int(result.get("total_plate_ocr_success_count", 0)),
                self._safe_int(result.get("violating_track_count", 0)),
                self._safe_int(result.get("violating_track_plate_count", 0)),
                self._safe_int(result.get("unread_violating_track_count", 0)),
                self._safe_int(result.get("track_count", 0)),
                self._safe_int(result.get("confirmed_track_count", 0)),
                self._safe_int(result.get("moving_track_count", 0)),
                self._safe_int(result.get("traffic_count_total", 0)),
                self._safe_int(result.get("traffic_count_forward", 0)),
                self._safe_int(result.get("traffic_count_backward", 0)),
                self._safe_int(result.get("frames_with_count_event", 0)),
                1 if bool(result.get("any_violation", False)) else 0,
                self._json_text(result, default="{}"),
                now,
            ),
        )

        for event in list(result.get("region_rule_events", []) or []):
            payload = dict(event or {})
            rule_type = str(payload.get("rule_type", "") or "").strip()
            event_type = (
                f"{self.VIDEO_EVENT_REGION_RULE}:{rule_type}"
                if rule_type
                else self.VIDEO_EVENT_REGION_RULE
            )
            frame_index = payload.get("frame_index")
            self._insert_track_event(
                cursor,
                record_id=int(record_id),
                track_id=str(payload.get("track_id", "") or ""),
                event_type=event_type,
                start_frame=frame_index,
                end_frame=frame_index,
                snapshot_path=None,
                metadata=payload,
                now=now,
            )

        for event in list(result.get("count_events", []) or []):
            payload = dict(event or {})
            direction = str(payload.get("direction", "") or "").strip()
            event_type = (
                f"{self.VIDEO_EVENT_TRAFFIC_COUNT}:{direction}"
                if direction
                else self.VIDEO_EVENT_TRAFFIC_COUNT
            )
            frame_index = payload.get("frame_index")
            self._insert_track_event(
                cursor,
                record_id=int(record_id),
                track_id=str(payload.get("track_id", "") or ""),
                event_type=event_type,
                start_frame=frame_index,
                end_frame=frame_index,
                snapshot_path=None,
                metadata=payload,
                now=now,
            )

        for payload in list(result.get("violating_track_plates", []) or []):
            track_payload = dict(payload or {})
            track_event_id = self._insert_track_event(
                cursor,
                record_id=int(record_id),
                track_id=str(track_payload.get("track_id", "") or ""),
                event_type=self.VIDEO_EVENT_VIOLATION_TRACK,
                start_frame=track_payload.get("first_violation_frame"),
                end_frame=track_payload.get("last_violation_frame"),
                snapshot_path=preview_frame_path or None,
                metadata=track_payload,
                now=now,
            )
            self._insert_plate_read(
                cursor,
                track_event_id=track_event_id,
                payload=track_payload,
                now=now,
            )

        cursor.execute(
            """
            UPDATE records
            SET media_type = ?, status = ?, processed_at = ?, processed_path = ?, error_message = NULL
            WHERE id = ?
            """,
            (self.MEDIA_VIDEO, self.STATUS_DONE, now, processed_video_path, int(record_id)),
        )

    def start_record(self, original_path, media_type="image"):
        """
        Create a running record for one processing task.
        """
        if not self.conn:
            return -1
        try:
            cursor = self.conn.cursor()
            image_id = self._upsert_image(cursor, original_path)
            cursor.execute(
                "INSERT INTO records (image_id, media_type, status, created_at) VALUES (?, ?, ?, ?)",
                (image_id, self._normalize_media_type(media_type), self.STATUS_RUNNING, self._now()),
            )
            self.conn.commit()
            return int(cursor.lastrowid)
        except Exception as exc:
            print(f"[DB] Failed to create record: {exc}")
            return -1

    def mark_record_failed(self, record_id, error_message):
        if not self.conn or record_id is None or int(record_id) <= 0:
            return
        try:
            self.conn.execute(
                """
                UPDATE records
                SET status = ?, processed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (self.STATUS_FAILED, self._now(), str(error_message), int(record_id)),
            )
            self.conn.commit()
        except Exception as exc:
            print(f"[DB] Failed to mark record as failed: {exc}")

    def complete_record_success(self, record_id, result):
        """
        Persist success payload and mark record done.
        """
        if not self.conn or record_id is None or int(record_id) <= 0:
            return

        record_id = int(record_id)
        now = self._now()
        media_type = self._normalize_media_type(dict(result or {}).get("media_type"))

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            if media_type == self.MEDIA_VIDEO:
                self._complete_video_record_success(cursor, record_id, dict(result or {}), now=now)
            else:
                self._complete_image_record_success(cursor, record_id, dict(result or {}), now=now)
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self.mark_record_failed(record_id, exc)
            raise

    def persist_result(self, original_path, result, media_type=None):
        """
        Convenience API: start running record -> persist success payload.
        """
        inferred_media_type = media_type
        if inferred_media_type is None:
            inferred_media_type = dict(result or {}).get("media_type")
        record_id = self.start_record(original_path, media_type=inferred_media_type)
        if record_id <= 0:
            raise RuntimeError("Failed to create running record")
        self.complete_record_success(record_id, result)
        return record_id

    def add_record(self, image_path, violation_type="Emergency Lane Occupation", location="Camera 01"):
        """
        Legacy compatibility API used by older scripts.
        """
        if not self.conn:
            return -1
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO violation_records (timestamp, location, violation_type, image_path, is_reviewed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._now(), location, violation_type, str(image_path), 0),
            )
            self.conn.commit()
            return int(cursor.lastrowid)
        except Exception as exc:
            print(f"[DB] Failed to insert legacy record: {exc}")
            return -1

    def close(self):
        if self.conn:
            self.conn.close()
            print("[DB] Connection closed")
