from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from core.plate_recognizer import (  # noqa: E402
    PLATE_TYPE_LABELS,
    create_plate_recognizer,
)


DEFAULT_VIDEO = PROJECT_ROOT / "images" / "vedio" / "illegal_2.mp4"
DEFAULT_YOLO = PROJECT_ROOT / "src" / "models" / "yolo11l-seg.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs_illegal_2_truck_roi"
TARGET_VEHICLE_CLASSES = {"truck", "bus"}


@dataclass
class VehicleCandidate:
    bbox: list[int]
    confidence: float
    class_name: str
    area: int


@dataclass
class PlateHit:
    frame_index: int
    timestamp_sec: float
    vehicle_class: str
    vehicle_confidence: float
    vehicle_box: list[int]
    crop_strategy: str
    preprocess: str
    plate_text: str
    plate_confidence: float
    plate_detect_confidence: float
    plate_type: str
    plate_box_in_frame: list[int]
    frame_path: str
    annotated_path: str
    crop_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truck-oriented Chinese LPR experiment with ROI crops and temporal fusion.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO, help="Vehicle detector weights.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--frame-step", type=int, default=5, help="Coarse pass frame step.")
    parser.add_argument("--refine-radius", type=int, default=15, help="Dense pass radius around trigger frames.")
    parser.add_argument("--max-vehicles", type=int, default=6, help="Max vehicles to inspect per frame.")
    parser.add_argument("--min-area", type=int, default=18000, help="Minimum vehicle box area.")
    parser.add_argument("--target-plate", type=str, default="津A38869D", help="Optional target plate for ranking.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def load_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}.")
    return frame


def detect_large_vehicles(model: YOLO, frame: np.ndarray, min_area: int, imgsz: int = 960) -> list[VehicleCandidate]:
    result = model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        imgsz=imgsz,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )[0]
    boxes = result.boxes
    if boxes is None:
        return []
    frame_h, frame_w = frame.shape[:2]
    names = result.names
    candidates: list[VehicleCandidate] = []
    xyxy_all = boxes.xyxy.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)
    conf_all = boxes.conf.cpu().numpy()
    for xyxy, cls_id, score in zip(xyxy_all, cls_all, conf_all):
        class_name = str(names[int(cls_id)])
        if class_name not in TARGET_VEHICLE_CLASSES:
            continue
        bbox = clip_box(xyxy.tolist(), frame_w, frame_h)
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area < int(min_area):
            continue
        candidates.append(
            VehicleCandidate(
                bbox=bbox,
                confidence=float(score),
                class_name=class_name,
                area=area,
            )
        )
    candidates.sort(key=lambda item: (item.area, item.confidence), reverse=True)
    return candidates


def crop_variants(frame: np.ndarray, vehicle: VehicleCandidate) -> list[dict]:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(vehicle.bbox, frame_w, frame_h)
    base = frame[y1:y2, x1:x2]
    if base.size == 0:
        return []

    variants = [{"name": "vehicle_full", "image": base, "origin": (x1, y1)}]
    crop_h, crop_w = base.shape[:2]
    if crop_h < 80 or crop_w < 120:
        return variants

    def local_box(rx1: float, ry1: float, rx2: float, ry2: float, name: str) -> None:
        bx1 = int(round(crop_w * rx1))
        by1 = int(round(crop_h * ry1))
        bx2 = int(round(crop_w * rx2))
        by2 = int(round(crop_h * ry2))
        bx1 = max(0, min(crop_w - 1, bx1))
        by1 = max(0, min(crop_h - 1, by1))
        bx2 = max(1, min(crop_w, bx2))
        by2 = max(1, min(crop_h, by2))
        if bx2 <= bx1 or by2 <= by1:
            return
        patch = base[by1:by2, bx1:bx2]
        if patch.size == 0:
            return
        variants.append(
            {
                "name": name,
                "image": patch,
                "origin": (x1 + bx1, y1 + by1),
            }
        )

    local_box(0.08, 0.35, 0.92, 1.00, "vehicle_lower_focus")
    local_box(0.00, 0.76, 0.48, 1.00, "bottom_left")
    local_box(0.00, 0.80, 0.40, 1.00, "bottom_left_tight")
    local_box(0.00, 0.74, 0.58, 1.00, "bottom_left_wide")
    local_box(0.00, 0.82, 1.00, 1.00, "bottom_strip")
    return variants


def resize_to_longest(image: np.ndarray, target_longest: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(int(height), int(width))
    if longest <= 0:
        return image, 1.0
    scale = max(1.0, float(target_longest) / float(longest))
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, float(scale)


def preprocess_variants(image: np.ndarray) -> list[dict]:
    long640, scale640 = resize_to_longest(image, 640)
    long960, scale960 = resize_to_longest(image, 960)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    long960_bilateral = cv2.bilateralFilter(long960, 7, 40, 40)
    long960_bilateral_sharp = cv2.filter2D(long960_bilateral, -1, sharpen_kernel)
    return [
        {"name": "long640", "image": long640, "scale": scale640},
        {"name": "long960_bilateral_sharp", "image": long960_bilateral_sharp, "scale": scale960},
    ]


def translate_box(box: Iterable[float], origin: tuple[int, int], scale: float, frame_w: int, frame_h: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    ox, oy = origin
    translated = [
        ox + x1 / max(scale, 1e-6),
        oy + y1 / max(scale, 1e-6),
        ox + x2 / max(scale, 1e-6),
        oy + y2 / max(scale, 1e-6),
    ]
    return clip_box(translated, frame_w, frame_h)


def normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def plate_tail_key(text: str) -> str:
    norm = normalize_text(text)
    if len(norm) >= 2 and not norm[0].isascii():
        return norm[1:]
    return norm


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def tail_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=plate_tail_key(a), b=plate_tail_key(b)).ratio()


def annotate_frame(frame: np.ndarray, vehicles: list[VehicleCandidate], hits: list[PlateHit]) -> np.ndarray:
    canvas = frame.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(
            canvas,
            f"{vehicle.class_name} {vehicle.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    for hit in hits:
        px1, py1, px2, py2 = hit.plate_box_in_frame
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (40, 220, 40), 3)
        cv2.putText(
            canvas,
            f"{hit.plate_text} {hit.plate_confidence:.2f}",
            (px1, min(canvas.shape[0] - 12, py2 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return canvas


def run_plate_recognition(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    vehicles: list[VehicleCandidate],
    recognizer,
    crop_dir: Path,
) -> list[PlateHit]:
    frame_h, frame_w = frame.shape[:2]
    hits: list[PlateHit] = []
    timestamp_sec = round(float(frame_index) / max(fps, 1.0), 3)
    for vehicle_rank, vehicle in enumerate(vehicles, start=1):
        for crop_variant in crop_variants(frame, vehicle):
            for prep in preprocess_variants(crop_variant["image"]):
                results = recognizer.recognize(prep["image"])
                if not results:
                    continue
                crop_path = crop_dir / f"frame_{frame_index:04d}_v{vehicle_rank:02d}_{crop_variant['name']}_{prep['name']}.jpg"
                if not crop_path.exists():
                    cv2.imwrite(str(crop_path), prep["image"])
                for raw in results:
                    plate_box = translate_box(
                        raw.box,
                        crop_variant["origin"],
                        prep["scale"],
                        frame_w,
                        frame_h,
                    )
                    hits.append(
                        PlateHit(
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            vehicle_class=vehicle.class_name,
                            vehicle_confidence=round(vehicle.confidence, 4),
                            vehicle_box=[int(v) for v in vehicle.bbox],
                            crop_strategy=str(crop_variant["name"]),
                            preprocess=str(prep["name"]),
                            plate_text=str(raw.text),
                            plate_confidence=round(float(raw.confidence), 4),
                            plate_detect_confidence=round(float(raw.detect_confidence), 4),
                            plate_type=PLATE_TYPE_LABELS.get(int(raw.plate_type_id), "unknown"),
                            plate_box_in_frame=plate_box,
                            frame_path="",
                            annotated_path="",
                            crop_path=str(crop_path),
                        )
                    )
    return hits


def frame_indices(frame_count: int, step: int) -> list[int]:
    step = max(1, int(step))
    return list(range(0, max(1, int(frame_count)), step))


def select_trigger_frames(hits: list[PlateHit], target: str, frame_count: int, radius: int) -> list[int]:
    if not hits:
        return []
    target_norm = normalize_text(target)
    target_tail = plate_tail_key(target)
    trigger_frames: set[int] = set()
    for hit in hits:
        norm = normalize_text(hit.plate_text)
        tail = plate_tail_key(hit.plate_text)
        if (
            tail == target_tail
            or target_tail in tail
            or tail in target_tail
            or text_similarity(norm, target_norm) >= 0.72
            or tail_similarity(tail, target_tail) >= 0.84
        ):
            start = max(0, int(hit.frame_index) - int(radius))
            end = min(int(frame_count) - 1, int(hit.frame_index) + int(radius))
            trigger_frames.update(range(start, end + 1))
    return sorted(trigger_frames)


def summarize_hits(hits: list[PlateHit], target_plate: str) -> dict:
    target_norm = normalize_text(target_plate)
    target_tail = plate_tail_key(target_plate)
    groups: dict[str, dict] = {}
    tail_groups: dict[str, dict] = {}

    for hit in hits:
        norm = normalize_text(hit.plate_text)
        tail = plate_tail_key(hit.plate_text)
        if not norm:
            continue

        exact_bucket = groups.setdefault(
            norm,
            {
                "plate_text": hit.plate_text,
                "support_count": 0,
                "avg_confidence_sum": 0.0,
                "best_hit": hit,
                "frame_indices": set(),
            },
        )
        exact_bucket["support_count"] += 1
        exact_bucket["avg_confidence_sum"] += float(hit.plate_confidence)
        exact_bucket["frame_indices"].add(int(hit.frame_index))
        if float(hit.plate_confidence) > float(exact_bucket["best_hit"].plate_confidence):
            exact_bucket["best_hit"] = hit

        tail_bucket = tail_groups.setdefault(
            tail,
            {
                "tail_key": tail,
                "texts": {},
                "support_count": 0,
                "avg_confidence_sum": 0.0,
                "best_hit": hit,
                "frame_indices": set(),
                "best_similarity": 0.0,
            },
        )
        tail_bucket["support_count"] += 1
        tail_bucket["avg_confidence_sum"] += float(hit.plate_confidence)
        tail_bucket["frame_indices"].add(int(hit.frame_index))
        tail_bucket["texts"][norm] = tail_bucket["texts"].get(norm, 0) + 1
        tail_bucket["best_similarity"] = max(
            float(tail_bucket["best_similarity"]),
            float(text_similarity(norm, target_norm)),
            float(tail_similarity(tail, target_tail)),
        )
        if float(hit.plate_confidence) > float(tail_bucket["best_hit"].plate_confidence):
            tail_bucket["best_hit"] = hit

    exact_results = []
    for value in groups.values():
        avg_conf = float(value["avg_confidence_sum"]) / max(1, int(value["support_count"]))
        best_hit = value["best_hit"]
        exact_results.append(
            {
                "plate_text": str(value["plate_text"]),
                "normalized_text": normalize_text(value["plate_text"]),
                "tail_key": plate_tail_key(value["plate_text"]),
                "support_count": int(value["support_count"]),
                "avg_confidence": round(avg_conf, 4),
                "frame_indices": sorted(int(v) for v in value["frame_indices"]),
                "best_similarity": round(text_similarity(value["plate_text"], target_norm), 4),
                "best_hit": asdict(best_hit),
            }
        )

    tail_results = []
    for value in tail_groups.values():
        avg_conf = float(value["avg_confidence_sum"]) / max(1, int(value["support_count"]))
        best_hit = value["best_hit"]
        sorted_texts = sorted(value["texts"].items(), key=lambda item: (-item[1], item[0]))
        tail_results.append(
            {
                "tail_key": str(value["tail_key"]),
                "texts": [{"text": key, "count": count} for key, count in sorted_texts],
                "support_count": int(value["support_count"]),
                "avg_confidence": round(avg_conf, 4),
                "frame_indices": sorted(int(v) for v in value["frame_indices"]),
                "best_similarity": round(float(value["best_similarity"]), 4),
                "best_hit": asdict(best_hit),
            }
        )

    exact_results.sort(key=lambda item: (item["best_similarity"], item["support_count"], item["avg_confidence"]), reverse=True)
    tail_results.sort(key=lambda item: (item["best_similarity"], item["support_count"], item["avg_confidence"]), reverse=True)

    return {
        "target_plate": target_plate,
        "target_plate_normalized": target_norm,
        "target_tail_key": target_tail,
        "exact_groups": exact_results,
        "tail_groups": tail_results,
    }


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.yolo.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo}")

    output_dir = ensure_dir(args.output_dir)
    frame_dir = ensure_dir(output_dir / "frames")
    annotated_dir = ensure_dir(output_dir / "annotated")
    crop_dir = ensure_dir(output_dir / "crops")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    model = YOLO(str(args.yolo))
    recognizer = create_plate_recognizer()
    if not bool(getattr(recognizer, "enabled", False)):
        raise RuntimeError(f"Plate recognizer unavailable: {getattr(recognizer, 'reason', 'unknown error')}")

    coarse_indices = frame_indices(frame_count, args.frame_step)
    processed_frames: dict[int, dict] = {}
    coarse_hits: list[PlateHit] = []

    for frame_index in coarse_indices:
        frame = load_frame(capture, frame_index)
        vehicles = detect_large_vehicles(model, frame, min_area=args.min_area)
        if not vehicles:
            continue
        hits = run_plate_recognition(
            frame,
            frame_index,
            fps,
            vehicles[: args.max_vehicles],
            recognizer,
            crop_dir,
        )
        if not hits:
            continue
        frame_name = f"frame_{frame_index:04d}_{frame_index / max(fps, 1.0):06.2f}s"
        frame_path = frame_dir / f"{frame_name}.jpg"
        annotated_path = annotated_dir / f"{frame_name}_annotated.jpg"
        cv2.imwrite(str(frame_path), frame)
        annotated = annotate_frame(frame, vehicles[: args.max_vehicles], hits)
        cv2.imwrite(str(annotated_path), annotated)
        for hit in hits:
            hit.frame_path = str(frame_path)
            hit.annotated_path = str(annotated_path)
        processed_frames[frame_index] = {
            "frame_path": str(frame_path),
            "annotated_path": str(annotated_path),
            "vehicles": [asdict(vehicle) for vehicle in vehicles[: args.max_vehicles]],
            "hits": [asdict(hit) for hit in hits],
        }
        coarse_hits.extend(hits)

    refine_indices = select_trigger_frames(coarse_hits, args.target_plate, frame_count, args.refine_radius)
    refine_indices = [index for index in refine_indices if index not in processed_frames]
    refined_hits: list[PlateHit] = []

    for frame_index in refine_indices:
        frame = load_frame(capture, frame_index)
        vehicles = detect_large_vehicles(model, frame, min_area=args.min_area)
        if not vehicles:
            continue
        hits = run_plate_recognition(
            frame,
            frame_index,
            fps,
            vehicles[: args.max_vehicles],
            recognizer,
            crop_dir,
        )
        if not hits:
            continue
        frame_name = f"frame_{frame_index:04d}_{frame_index / max(fps, 1.0):06.2f}s"
        frame_path = frame_dir / f"{frame_name}.jpg"
        annotated_path = annotated_dir / f"{frame_name}_annotated.jpg"
        cv2.imwrite(str(frame_path), frame)
        annotated = annotate_frame(frame, vehicles[: args.max_vehicles], hits)
        cv2.imwrite(str(annotated_path), annotated)
        for hit in hits:
            hit.frame_path = str(frame_path)
            hit.annotated_path = str(annotated_path)
        processed_frames[frame_index] = {
            "frame_path": str(frame_path),
            "annotated_path": str(annotated_path),
            "vehicles": [asdict(vehicle) for vehicle in vehicles[: args.max_vehicles]],
            "hits": [asdict(hit) for hit in hits],
        }
        refined_hits.extend(hits)

    capture.release()

    all_hits = coarse_hits + refined_hits
    summary = {
        "video": str(args.video),
        "yolo": str(args.yolo),
        "video_meta": {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": round(frame_count / max(fps, 1.0), 3),
        },
        "coarse_frame_step": int(args.frame_step),
        "refine_radius": int(args.refine_radius),
        "coarse_frames_scanned": len(coarse_indices),
        "refine_frames_scanned": len(refine_indices),
        "frames_with_hits": sorted(int(index) for index in processed_frames.keys()),
        "total_hits": len(all_hits),
        "target_summary": summarize_hits(all_hits, args.target_plate),
        "frames": {str(key): value for key, value in sorted(processed_frames.items())},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    print(f"Total hits: {len(all_hits)}")
    print(f"Frames with hits: {len(processed_frames)}")
    top_tail = summary["target_summary"]["tail_groups"][:3]
    if top_tail:
        print("Top tail groups:")
        for item in top_tail:
            best_hit = item["best_hit"]
            print(
                f"  tail={item['tail_key']} support={item['support_count']} avg={item['avg_confidence']:.4f} "
                f"text={best_hit['plate_text']} frame={best_hit['frame_index']}"
            )


if __name__ == "__main__":
    main()
