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
    Database manager for M2 schema:
    images -> records -> lane_segments/detections -> reviews
    """

    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"

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
                    status TEXT NOT NULL CHECK(status IN ('running','done','failed')),
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    processed_path TEXT,
                    error_message TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(id)
                )
                """
            )
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

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_image_id ON records(image_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_status_created ON records(status, created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_lane_segments_record_id ON lane_segments(record_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_record_id ON detections(record_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_violation ON detections(is_violating, violation_type_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_reviews_detection_id ON reviews(detection_id)"
            )
            self._ensure_violation_type(cursor, "EMERGENCY_LANE", "Emergency Lane Occupation")
            ensure_sequence_schema(self.conn)

            self.conn.commit()
            print(f"[DB] 鏁版嵁搴撳凡杩炴帴: {self.db_path}")
        except Exception as exc:
            print(f"[DB] 鏁版嵁搴撳垵濮嬪寲澶辫触: {exc}")

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    def start_record(self, original_path):
        """
        Create a running record for one processing task.
        """
        if not self.conn:
            return -1
        try:
            cursor = self.conn.cursor()
            image_id = self._upsert_image(cursor, original_path)
            cursor.execute(
                "INSERT INTO records (image_id, status, created_at) VALUES (?, ?, ?)",
                (image_id, self.STATUS_RUNNING, self._now()),
            )
            self.conn.commit()
            return int(cursor.lastrowid)
        except Exception as exc:
            print(f"[DB] 鍒涘缓 record 澶辫触: {exc}")
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
            print(f"[DB] 鏍囪澶辫触 record 澶辫触: {exc}")

    def complete_record_success(self, record_id, result):
        """
        Persist lane segments + detections and mark record done.
        """
        if not self.conn or record_id is None or int(record_id) <= 0:
            return

        record_id = int(record_id)
        detections = list(result.get("detections", []))
        lane_polygons = list(result.get("lane_polygons", []))
        processed_path = str(result.get("processed_path", "")).strip()
        now = self._now()

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute(
                """
                INSERT INTO lane_segments (record_id, lane_type_code, polygons_json, mask_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    "EMERGENCY_LANE",
                    json.dumps(lane_polygons, ensure_ascii=False),
                    None,
                    now,
                ),
            )

            violation_type_id = self._ensure_violation_type(
                cursor, "EMERGENCY_LANE", "Emergency Lane Occupation"
            )
            for det in detections:
                bbox = list(det.get("bbox", [0, 0, 0, 0]))
                if len(bbox) != 4:
                    bbox = [0, 0, 0, 0]

                is_violating = 1 if det.get("is_violating") else 0
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
                        record_id,
                        str(det.get("vehicle_type", "vehicle")),
                        float(det.get("confidence", 0.0)),
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        json.dumps(det.get("footprint", []), ensure_ascii=False),
                        json.dumps(det.get("corners_2d", []), ensure_ascii=False),
                        is_violating,
                        float(det.get("violation_ratio", 0.0)),
                        violation_type_id if is_violating else None,
                        now,
                    ),
                )

            cursor.execute(
                """
                UPDATE records
                SET status = ?, processed_at = ?, processed_path = ?, error_message = NULL
                WHERE id = ?
                """,
                (self.STATUS_DONE, now, processed_path, record_id),
            )
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self.mark_record_failed(record_id, exc)
            raise

    def persist_result(self, original_path, result):
        """
        Convenience API: start running record -> persist success payload.
        """
        record_id = self.start_record(original_path)
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
            print(f"[DB] 鎻掑叆 legacy 璁板綍澶辫触: {exc}")
            return -1

    def close(self):
        if self.conn:
            self.conn.close()
            print("[DB] 鏁版嵁搴撹繛鎺ュ凡鍏抽棴")

