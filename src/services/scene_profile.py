from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


Point = List[float]


@dataclass
class RegionRuleBinding:
    rule_type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolygonRegion:
    name: str
    region_type: str
    points: List[Point]
    region_id: str = ""
    source: str = "manual"
    enabled: bool = True
    direction_line: List[Point] = field(default_factory=list)
    rule_bindings: List[RegionRuleBinding] = field(default_factory=list)


@dataclass
class CountLine:
    name: str
    start: Point
    end: Point
    direction_mode: str = "any"
    enabled: bool = True


@dataclass
class SceneProfile:
    camera_id: str
    fps: float
    source_path: str = ""
    notes: str = ""
    parking_regions: List[PolygonRegion] = field(default_factory=list)
    count_lines: List[CountLine] = field(default_factory=list)


def normalize_polygon_points(points: Sequence[Sequence[float]]) -> List[Point]:
    normalized: List[Point] = []
    for point in points:
        if len(point) != 2:
            raise ValueError(f"Polygon point must contain 2 values, got: {point}")
        normalized.append([float(point[0]), float(point[1])])
    if len(normalized) < 3:
        raise ValueError("Polygon must contain at least 3 points")
    return normalized


def normalize_line_points(start: Sequence[float], end: Sequence[float]) -> tuple[Point, Point]:
    if len(start) != 2 or len(end) != 2:
        raise ValueError("Count line endpoints must each contain 2 values")
    return [float(start[0]), float(start[1])], [float(end[0]), float(end[1])]


def normalize_optional_direction_line(points) -> List[Point]:
    if not points:
        return []
    if len(points) != 2:
        raise ValueError("Direction line must contain exactly 2 points")
    start, end = normalize_line_points(points[0], points[1])
    return [start, end]


def scene_profile_to_dict(profile: SceneProfile) -> dict:
    return asdict(profile)


def scene_profile_from_dict(payload: dict) -> SceneProfile:
    parking_regions = [
        PolygonRegion(
            name=str(item.get("name", "manual_roi")),
            region_type=str(item.get("region_type", "parking_roi")),
            points=normalize_polygon_points(item.get("points", [])),
            region_id=str(item.get("region_id", "") or ""),
            source=str(item.get("source", "manual")),
            enabled=bool(item.get("enabled", True)),
            direction_line=normalize_optional_direction_line(item.get("direction_line", [])),
            rule_bindings=[
                RegionRuleBinding(
                    rule_type=str(binding.get("rule_type", "") or "").strip(),
                    enabled=bool(binding.get("enabled", True)),
                    params=dict(binding.get("params", {}) or {}),
                )
                for binding in item.get("rule_bindings", [])
                if str(binding.get("rule_type", "") or "").strip()
            ],
        )
        for item in payload.get("parking_regions", [])
    ]
    count_lines = []
    for item in payload.get("count_lines", []):
        start, end = normalize_line_points(item.get("start", []), item.get("end", []))
        count_lines.append(
            CountLine(
                name=str(item.get("name", "count_line")),
                start=start,
                end=end,
                direction_mode=str(item.get("direction_mode", "any")),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return SceneProfile(
        camera_id=str(payload.get("camera_id", "camera_01")),
        fps=float(payload.get("fps", 0.0)),
        source_path=str(payload.get("source_path", "")),
        notes=str(payload.get("notes", "")),
        parking_regions=parking_regions,
        count_lines=count_lines,
    )


def save_scene_profile(path: Path | str, profile: SceneProfile) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = scene_profile_to_dict(profile)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_scene_profile(path: Path | str) -> SceneProfile:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return scene_profile_from_dict(payload)
