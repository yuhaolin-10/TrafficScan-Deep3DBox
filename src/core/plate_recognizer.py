from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from pathlib import Path
import sys
from typing import Iterable, List, Optional

import cv2
import numpy as np


PLATE_TYPE_LABELS = {
    -1: "unknown",
    0: "blue",
    1: "yellow_single",
    2: "white_single",
    3: "green_new_energy",
    4: "black_hk_macao",
    5: "hk_single",
    6: "hk_double",
    7: "macao_single",
    8: "macao_double",
    9: "yellow_double",
}

PLATE_VEHICLE_TYPES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class PlateCandidate:
    text: str
    confidence: float
    frame_index: Optional[int] = None
    crop_path: str = ""
    plate_type: str = ""
    plate_type_id: int = -1
    detect_confidence: float = 0.0
    crop_strategy: str = ""
    box: List[int] = field(default_factory=list)


@dataclass
class PlateRecognitionResult:
    text: str
    confidence: float
    support_count: int
    source_frame_indices: List[int]


class NullPlateRecognizer:
    enabled = False

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = str(reason or "disabled")
        self.min_confidence = 0.0
        self.support_count_hint = 0

    def recognize(self, _image) -> List[PlateCandidate]:
        return []


class HyperLPRPlateRecognizer:
    enabled = True

    def __init__(
        self,
        *,
        detect_level: str = "high",
        min_confidence: float = 0.55,
        min_vehicle_box_area: int = 7000,
        min_vehicle_box_side: int = 72,
    ) -> None:
        self._ensure_vendor_path()
        import hyperlpr3 as lpr3  # type: ignore

        level_name = str(detect_level or "high").strip().lower()
        level = lpr3.DETECT_LEVEL_LOW if level_name == "low" else lpr3.DETECT_LEVEL_HIGH

        self._lpr3 = lpr3
        self.min_confidence = float(min_confidence)
        self.min_vehicle_box_area = int(min_vehicle_box_area)
        self.min_vehicle_box_side = int(min_vehicle_box_side)
        self.support_count_hint = 2
        self.catcher = lpr3.LicensePlateCatcher(detect_level=level)

    def _ensure_vendor_path(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        vendor_candidates = [
            project_root / "experiments" / "license_plate_poc" / "vendor",
        ]
        for vendor_path in vendor_candidates:
            if vendor_path.exists():
                vendor_str = str(vendor_path)
                if vendor_str not in sys.path:
                    # Append instead of prepend to avoid shadowing environment packages such as numpy/torch.
                    sys.path.append(vendor_str)
                return
        raise ImportError(
            "HyperLPR vendor directory was not found. Expected experiments/license_plate_poc/vendor to exist."
        )

    def recognize(self, image) -> List[PlateCandidate]:
        if image is None:
            return []
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []
        results = self.catcher(arr)
        candidates: list[PlateCandidate] = []
        for raw in results or []:
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            code, confidence, plate_type_id, box = raw[:4]
            detect_confidence = float(raw[4]) if len(raw) >= 5 else 0.0
            candidates.append(
                PlateCandidate(
                    text=str(code),
                    confidence=float(confidence),
                    plate_type=plate_type_label(plate_type_id),
                    plate_type_id=int(plate_type_id),
                    detect_confidence=detect_confidence,
                    box=[int(round(float(v))) for v in list(box or [])[:4]],
                )
            )
        return candidates


def create_plate_recognizer() -> NullPlateRecognizer | HyperLPRPlateRecognizer:
    try:
        return HyperLPRPlateRecognizer()
    except Exception as exc:
        return NullPlateRecognizer(reason=str(exc))


def plate_type_label(plate_type_id: int) -> str:
    try:
        key = int(plate_type_id)
    except Exception:
        key = -1
    return PLATE_TYPE_LABELS.get(key, "unknown")


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def normalize_plate_text(text: str) -> str:
    return _normalize_text(text)


def _plate_core_text(text: str) -> tuple[str, bool]:
    norm = _normalize_text(text)
    if len(norm) >= 2 and not norm[0].isascii():
        return norm[1:], True
    return norm, False


def _plate_tail_key(text: str) -> str:
    core, _has_prefix = _plate_core_text(text)
    return core


def _plate_pattern_score(text: str, plate_type: str | int = "") -> float:
    norm = _normalize_text(text)
    if not norm:
        return 0.0

    core, has_prefix = _plate_core_text(norm)
    if isinstance(plate_type, int):
        plate_key = plate_type_label(plate_type)
    else:
        plate_key = str(plate_type or "").strip().lower()

    score = 0.0
    if has_prefix:
        score += 0.18
    if core and core[0].isascii() and core[0].isalpha():
        score += 0.18

    digit_count = sum(1 for ch in core if ch.isdigit())
    score += min(0.35, 0.05 * digit_count)

    if plate_key == "green_new_energy":
        if len(core) == 7:
            score += 0.65
        elif len(core) == 6:
            score += 0.25
        if core.endswith(("D", "F")):
            score += 0.42
        middle = core[1:-1] if len(core) >= 3 else core[1:]
        middle_digit_count = sum(1 for ch in middle if ch.isdigit())
        score += 0.20 * middle_digit_count
        if middle and middle_digit_count == len(middle):
            score += 0.35
    else:
        if len(core) == 6:
            score += 0.60
        elif len(core) == 7:
            score += 0.35
        rest = core[1:]
        if rest and all(ch.isalnum() for ch in rest):
            score += 0.35
        if sum(1 for ch in rest if ch.isdigit()) >= 3:
            score += 0.12

    return score


def fuse_plate_candidates(
    candidates: Iterable[PlateCandidate],
    min_confidence: float = 0.0,
) -> Optional[PlateRecognitionResult]:
    grouped = {}
    for item in candidates:
        norm_text = _normalize_text(item.text)
        if not norm_text:
            continue
        conf = float(item.confidence)
        if conf < float(min_confidence):
            continue
        bucket = grouped.setdefault(norm_text, {"score": 0.0, "count": 0, "frames": []})
        bucket["score"] += conf
        bucket["count"] += 1
        if item.frame_index is not None:
            bucket["frames"].append(int(item.frame_index))
    if not grouped:
        return None
    text, stats = max(grouped.items(), key=lambda kv: (kv[1]["score"], kv[1]["count"], kv[0]))
    confidence = float(stats["score"]) / max(1, int(stats["count"]))
    frames = sorted(set(stats["frames"]))
    return PlateRecognitionResult(
        text=text,
        confidence=confidence,
        support_count=int(stats["count"]),
        source_frame_indices=frames,
    )


def summarize_plate_candidates(
    candidates: Iterable[PlateCandidate],
    *,
    min_confidence: float = 0.0,
) -> Optional[dict]:
    filtered: list[dict] = []
    for item in list(candidates):
        norm_text = _normalize_text(item.text)
        conf = float(item.confidence)
        if not norm_text or conf < float(min_confidence):
            continue
        plate_type_name = str(item.plate_type or plate_type_label(item.plate_type_id))
        filtered.append(
            {
                "item": item,
                "normalized_text": norm_text,
                "tail_key": _plate_tail_key(norm_text),
                "pattern_score": _plate_pattern_score(norm_text, plate_type_name),
                "plate_type": plate_type_name,
                "has_prefix": bool(len(norm_text) >= 1 and not norm_text[0].isascii()),
            }
        )

    if not filtered:
        return None

    exact_groups: dict[str, dict] = {}
    for entry in filtered:
        item = entry["item"]
        norm_text = entry["normalized_text"]
        group = exact_groups.setdefault(
            norm_text,
            {
                "plate_text": str(item.text),
                "normalized_text": norm_text,
                "tail_key": entry["tail_key"],
                "plate_type": entry["plate_type"],
                "has_prefix": bool(entry["has_prefix"]),
                "items": [],
                "support_count": 0,
                "confidence_sum": 0.0,
                "frame_indices": set(),
                "best_item": item,
                "best_pattern_score": 0.0,
            },
        )
        group["items"].append(item)
        group["support_count"] += 1
        group["confidence_sum"] += float(item.confidence)
        if item.frame_index is not None:
            group["frame_indices"].add(int(item.frame_index))
        group["best_pattern_score"] = max(float(group["best_pattern_score"]), float(entry["pattern_score"]))
        if (
            float(item.confidence),
            float(item.detect_confidence),
            float(len(item.box)),
            float(item.frame_index if item.frame_index is not None else -1),
        ) > (
            float(group["best_item"].confidence),
            float(group["best_item"].detect_confidence),
            float(len(group["best_item"].box)),
            float(group["best_item"].frame_index if group["best_item"].frame_index is not None else -1),
        ):
            group["best_item"] = item

    for group in exact_groups.values():
        group["avg_confidence"] = float(group["confidence_sum"]) / max(1, int(group["support_count"]))
        group["exact_score"] = (
            float(group["best_pattern_score"])
            + (float(group["avg_confidence"]) * 0.75)
            + (0.12 * log1p(int(group["support_count"])))
        )

    tail_groups: dict[str, dict] = {}
    for group in exact_groups.values():
        tail_group = tail_groups.setdefault(
            str(group["tail_key"]),
            {
                "tail_key": str(group["tail_key"]),
                "items": [],
                "exact_groups": [],
                "support_count": 0,
                "confidence_sum": 0.0,
                "frame_indices": set(),
                "best_item": group["best_item"],
                "best_pattern_score": 0.0,
            },
        )
        tail_group["items"].extend(group["items"])
        tail_group["exact_groups"].append(group)
        tail_group["support_count"] += int(group["support_count"])
        tail_group["confidence_sum"] += float(group["confidence_sum"])
        tail_group["frame_indices"].update(group["frame_indices"])
        tail_group["best_pattern_score"] = max(float(tail_group["best_pattern_score"]), float(group["best_pattern_score"]))
        if (
            float(group["best_item"].confidence),
            float(group["best_item"].detect_confidence),
            float(len(group["best_item"].box)),
            float(group["best_item"].frame_index if group["best_item"].frame_index is not None else -1),
        ) > (
            float(tail_group["best_item"].confidence),
            float(tail_group["best_item"].detect_confidence),
            float(len(tail_group["best_item"].box)),
            float(tail_group["best_item"].frame_index if tail_group["best_item"].frame_index is not None else -1),
        ):
            tail_group["best_item"] = group["best_item"]

    for tail_group in tail_groups.values():
        tail_group["avg_confidence"] = float(tail_group["confidence_sum"]) / max(1, int(tail_group["support_count"]))
        tail_group["tail_score"] = (
            float(tail_group["best_pattern_score"])
            + (float(tail_group["avg_confidence"]) * 0.75)
            + (0.10 * log1p(int(tail_group["support_count"])))
        )

    selected_tail = max(
        tail_groups.values(),
        key=lambda group: (
            float(group["tail_score"]),
            int(group["support_count"]),
            float(group["avg_confidence"]),
        ),
    )
    selected_exact_groups = list(selected_tail["exact_groups"])

    bare_groups = [group for group in selected_exact_groups if not bool(group["has_prefix"])]
    prefixed_groups = [group for group in selected_exact_groups if bool(group["has_prefix"])]
    chosen_group = None

    prefixed_items = [
        item
        for item in selected_tail["items"]
        if _plate_core_text(item.text)[1]
    ]
    if prefixed_items:
        best_prefixed = max(
            prefixed_items,
            key=lambda item: (
                float(item.confidence),
                float(item.detect_confidence),
                float(item.frame_index if item.frame_index is not None else -1),
            ),
        )
        best_key = _normalize_text(best_prefixed.text)
        for group in selected_exact_groups:
            if str(group["normalized_text"]) == str(best_key):
                chosen_group = group
                break

    if prefixed_groups:
        prefix_support: dict[str, int] = {}
        for group in prefixed_groups:
            prefix = str(group["normalized_text"])[0]
            prefix_support[prefix] = prefix_support.get(prefix, 0) + int(group["support_count"])
        dominant_prefix, dominant_support = max(prefix_support.items(), key=lambda item: (item[1], item[0]))
        tie_count = sum(1 for value in prefix_support.values() if int(value) == int(dominant_support))
        bare_support = sum(int(group["support_count"]) for group in bare_groups)
        if int(dominant_support) >= 2 and tie_count == 1 and int(dominant_support) > int(bare_support):
            chosen_group = max(
                [group for group in prefixed_groups if str(group["normalized_text"]).startswith(dominant_prefix)],
                key=lambda group: (
                    float(group["exact_score"]),
                    int(group["support_count"]),
                    float(group["avg_confidence"]),
                ),
            )

    if chosen_group is None and bare_groups:
        chosen_group = max(
            bare_groups,
            key=lambda group: (
                float(group["best_pattern_score"]),
                float(group["exact_score"]),
                int(group["support_count"]),
                float(group["avg_confidence"]),
            ),
        )

    if chosen_group is None:
        chosen_group = max(
            selected_exact_groups,
            key=lambda group: (
                float(group["exact_score"]),
                int(group["support_count"]),
                float(group["avg_confidence"]),
            ),
        )

    representative = chosen_group["best_item"]
    selected_items = list(selected_tail["items"])
    return {
        "plate_text": str(chosen_group["plate_text"]),
        "plate_confidence": float(selected_tail["avg_confidence"]),
        "plate_support_count": int(selected_tail["support_count"]),
        "plate_source_frame_indices": [int(v) for v in sorted(selected_tail["frame_indices"])],
        "plate_type": str(representative.plate_type or plate_type_label(representative.plate_type_id)),
        "plate_type_id": int(representative.plate_type_id),
        "plate_detect_confidence": float(representative.detect_confidence),
        "plate_box": [int(v) for v in representative.box],
        "plate_crop_strategy": str(representative.crop_strategy or ""),
        "plate_candidates": [
            {
                "text": str(item.text),
                "confidence": float(item.confidence),
                "frame_index": None if item.frame_index is None else int(item.frame_index),
                "plate_type": str(item.plate_type or plate_type_label(item.plate_type_id)),
                "plate_type_id": int(item.plate_type_id),
                "detect_confidence": float(item.detect_confidence),
                "crop_strategy": str(item.crop_strategy or ""),
                "box": [int(v) for v in item.box],
            }
            for item in sorted(
                selected_items,
                key=lambda item: (
                    float(item.confidence),
                    float(item.detect_confidence),
                    float(item.frame_index if item.frame_index is not None else -1),
                ),
                reverse=True,
            )
        ],
    }


def _clip_box(box_xyxy, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in list(box_xyxy or [])[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def _expand_box(box: list[int], width: int, height: int, x_pad_ratio: float = 0.08, y_pad_ratio: float = 0.08) -> list[int]:
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * x_pad_ratio))
    pad_y = int(round((y2 - y1) * y_pad_ratio))
    return _clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def _vehicle_crop_variants(image: np.ndarray, bbox: list[int], *, vehicle_type: str = "") -> list[dict]:
    height, width = image.shape[:2]
    expanded = _expand_box(bbox, width, height)
    x1, y1, x2, y2 = expanded
    base_crop = image[y1:y2, x1:x2]
    if base_crop.size == 0:
        return []

    vehicle_key = str(vehicle_type or "").strip().lower()
    variants = [
        {
            "name": "vehicle_full",
            "image": base_crop,
            "origin": (x1, y1),
        }
    ]

    crop_h, crop_w = base_crop.shape[:2]
    if crop_h >= 80 and crop_w >= 120:
        def append_local_variant(rx1: float, ry1: float, rx2: float, ry2: float, name: str) -> None:
            local_x1 = int(round(crop_w * rx1))
            local_y1 = int(round(crop_h * ry1))
            local_x2 = int(round(crop_w * rx2))
            local_y2 = int(round(crop_h * ry2))
            local_x1 = max(0, min(crop_w - 1, local_x1))
            local_y1 = max(0, min(crop_h - 1, local_y1))
            local_x2 = max(1, min(crop_w, local_x2))
            local_y2 = max(1, min(crop_h, local_y2))
            if local_x2 <= local_x1 or local_y2 <= local_y1:
                return
            focus_crop = base_crop[local_y1:local_y2, local_x1:local_x2]
            if not focus_crop.size:
                return
            variants.append(
                {
                    "name": name,
                    "image": focus_crop,
                    "origin": (x1 + local_x1, y1 + local_y1),
                }
            )

        append_local_variant(0.08, 0.35, 0.92, 1.00, "vehicle_lower_focus")
        if vehicle_key in {"truck", "bus"}:
            append_local_variant(0.00, 0.74, 0.58, 1.00, "bottom_left_wide")
            append_local_variant(0.00, 0.80, 0.40, 1.00, "bottom_left_tight")
            append_local_variant(0.00, 0.82, 1.00, 1.00, "bottom_strip")
    return variants


def _resize_to_longest(image: np.ndarray, target_longest: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(int(height), int(width))
    if longest <= 0:
        return image, 1.0
    scale = max(1.0, float(target_longest) / float(longest))
    if scale <= 1.01:
        return image, 1.0
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, float(scale)


def _preprocess_lpr_variants(image: np.ndarray, *, vehicle_type: str = "", crop_name: str = "") -> list[dict]:
    long640, scale640 = _resize_to_longest(image, 640)
    variants = [
        {
            "name": "long640",
            "image": long640,
            "scale": scale640,
        }
    ]

    vehicle_key = str(vehicle_type or "").strip().lower()
    needs_strong_variant = vehicle_key in {"truck", "bus"} or str(crop_name).startswith("bottom_")
    if not needs_strong_variant:
        return variants

    long960, scale960 = _resize_to_longest(image, 960)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    long960_bilateral = cv2.bilateralFilter(long960, 7, 40, 40)
    long960_bilateral_sharp = cv2.filter2D(long960_bilateral, -1, sharpen_kernel)
    variants.append(
        {
            "name": "long960_bilateral_sharp",
            "image": long960_bilateral_sharp,
            "scale": scale960,
        }
    )
    return variants


def _translate_box(box: list[int], origin: tuple[int, int], scale: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in list(box or [])[:4]]
    ox, oy = origin
    translated = [
        ox + x1 / max(scale, 1e-6),
        oy + y1 / max(scale, 1e-6),
        ox + x2 / max(scale, 1e-6),
        oy + y2 / max(scale, 1e-6),
    ]
    return _clip_box(translated, width, height)


def recognize_vehicle_plate(
    image: np.ndarray,
    bbox_xyxy,
    recognizer,
    *,
    vehicle_type: str = "",
    frame_index: Optional[int] = None,
) -> Optional[dict]:
    if recognizer is None or not bool(getattr(recognizer, "enabled", False)):
        return None
    if image is None:
        return None

    vehicle_key = str(vehicle_type or "").strip().lower()
    if vehicle_key and vehicle_key not in PLATE_VEHICLE_TYPES:
        return None

    frame_h, frame_w = image.shape[:2]
    bbox = _clip_box(bbox_xyxy, frame_w, frame_h)
    box_w = max(1, bbox[2] - bbox[0])
    box_h = max(1, bbox[3] - bbox[1])
    box_area = box_w * box_h
    min_box_area = int(getattr(recognizer, "min_vehicle_box_area", 0) or 0)
    min_box_side = int(getattr(recognizer, "min_vehicle_box_side", 0) or 0)
    if box_area < max(1, min_box_area) or min(box_w, box_h) < max(1, min_box_side):
        return None

    candidates: list[PlateCandidate] = []
    for variant in _vehicle_crop_variants(image, bbox, vehicle_type=vehicle_key):
        crop_image = variant["image"]
        if crop_image.size == 0:
            continue
        for prepared in _preprocess_lpr_variants(crop_image, vehicle_type=vehicle_key, crop_name=str(variant["name"])):
            for item in recognizer.recognize(prepared["image"]):
                translated_box = _translate_box(item.box, variant["origin"], prepared["scale"], frame_w, frame_h)
                candidates.append(
                    PlateCandidate(
                        text=str(item.text),
                        confidence=float(item.confidence),
                        frame_index=frame_index,
                        crop_path=str(item.crop_path or ""),
                        plate_type=str(item.plate_type or ""),
                        plate_type_id=int(item.plate_type_id),
                        detect_confidence=float(item.detect_confidence),
                        crop_strategy=f"{variant['name']}:{prepared['name']}",
                        box=translated_box,
                    )
                )

    summary = summarize_plate_candidates(
        candidates,
        min_confidence=float(getattr(recognizer, "min_confidence", 0.0) or 0.0),
    )
    if summary is None:
        return None
    summary["vehicle_bbox"] = [int(v) for v in bbox]
    return summary
