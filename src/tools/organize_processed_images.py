import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


DATE_PREFIX_RE = re.compile(r"^(\d{8})_")
RUN_DIR_RE = re.compile(r".+_run_\d{8}_\d{6}$")


@dataclass
class MoveItem:
    old_path: Path
    new_path: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_moves(images_dir: Path) -> List[MoveItem]:
    moves: List[MoveItem] = []

    records_root = images_dir / "records"
    runs_root = images_dir / "runs"
    _ensure_dir(records_root)
    _ensure_dir(runs_root)

    for child in sorted(images_dir.iterdir()):
        if child.name in {"records", "runs"}:
            continue

        if child.is_dir():
            if RUN_DIR_RE.match(child.name):
                moves.append(MoveItem(old_path=child, new_path=runs_root / child.name))
            continue

        match = DATE_PREFIX_RE.match(child.name)
        if not match:
            continue
        date_prefix = match.group(1)
        moves.append(
            MoveItem(
                old_path=child,
                new_path=records_root / date_prefix / child.name,
            )
        )

    return moves


def _apply_moves(moves: List[MoveItem]) -> List[Tuple[str, str]]:
    moved_pairs: List[Tuple[str, str]] = []
    for item in moves:
        if not item.old_path.exists():
            continue
        if item.new_path.exists():
            continue
        _ensure_dir(item.new_path.parent)
        item.old_path.rename(item.new_path)
        moved_pairs.append((str(item.old_path.resolve()), str(item.new_path.resolve())))
    return moved_pairs


def _update_database(db_path: Path, moved_pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    if not db_path.exists() or not moved_pairs:
        return {"records_updated": 0, "legacy_updated": 0}

    conn = sqlite3.connect(db_path)
    records_updated = 0
    legacy_updated = 0
    try:
        cur = conn.cursor()
        for old_path, new_path in moved_pairs:
            cur.execute(
                "UPDATE records SET processed_path = ? WHERE processed_path = ?",
                (new_path, old_path),
            )
            records_updated += int(cur.rowcount)

            cur.execute(
                "UPDATE violation_records SET image_path = ? WHERE image_path = ?",
                (new_path, old_path),
            )
            legacy_updated += int(cur.rowcount)

        conn.commit()
    finally:
        conn.close()

    return {
        "records_updated": records_updated,
        "legacy_updated": legacy_updated,
    }


def _write_manifest(images_dir: Path, moved_pairs: List[Tuple[str, str]], db_update_stats: Dict[str, int]) -> Path:
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "move_count": len(moved_pairs),
        "db_update_stats": db_update_stats,
        "moves": [{"old": old, "new": new} for old, new in moved_pairs],
    }
    manifest_path = images_dir / "runs" / f"organize_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    root = _project_root()
    images_dir = root / "data" / "images"
    db_path = root / "data" / "db" / "traffic_scan.db"

    if not images_dir.exists():
        raise SystemExit(f"Missing images dir: {images_dir}")

    moves = _build_moves(images_dir)
    moved_pairs = _apply_moves(moves)
    db_update_stats = _update_database(db_path, moved_pairs)
    manifest_path = _write_manifest(images_dir, moved_pairs, db_update_stats)

    print(f"images_dir={images_dir}")
    print(f"move_candidates={len(moves)}")
    print(f"moved={len(moved_pairs)}")
    print(f"records_updated={db_update_stats['records_updated']}")
    print(f"legacy_updated={db_update_stats['legacy_updated']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
