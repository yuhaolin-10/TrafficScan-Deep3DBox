from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import hyperlpr3 as lpr3


PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "images" / "vedio" / "365A2046.MP4"
DEFAULT_YOLO = PROJECT_ROOT / "src" / "models" / "yolo11l-seg.pt"
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
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class VehicleCandidate:
    bbox: list[int]
    confidence: float
    class_name: str
    area: int


@dataclass
class PlateDetection:
    frame_index: int
    timestamp_sec: float
    frame_path: str
    annotated_path: str
    vehicle_box: list[int]
    vehicle_confidence: float
    vehicle_class: str
    crop_strategy: str
    plate_text: str
    plate_confidence: float
    plate_detect_confidence: float
    plate_type: str
    plate_box_in_frame: list[int]
    crop_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone Chinese LPR proof-of-concept on sampled video frames.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO, help="Vehicle detector weights.")
    parser.add_argument("--num-frames", type=int, default=5, help="How many frames to keep for detailed analysis.")
    parser.add_argument("--scan-count", type=int, default=12, help="How many candidate timestamps to scan before choosing frames.")
    parser.add_argument("--max-vehicles", type=int, default=6, help="How many large vehicles to inspect per selected frame.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs", help="Directory for generated files.")
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


def expand_box(box: list[int], width: int, height: int, x_pad_ratio: float = 0.08, y_pad_ratio: float = 0.08) -> list[int]:
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * x_pad_ratio))
    pad_y = int(round((y2 - y1) * y_pad_ratio))
    return clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def load_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}.")
    return frame


def detect_vehicles(model: YOLO, frame: np.ndarray, imgsz: int = 1280) -> list[VehicleCandidate]:
    result = model.predict(source=frame, conf=0.25, iou=0.45, imgsz=imgsz, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
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
        if class_name not in VEHICLE_CLASSES:
            continue
        bbox = clip_box(xyxy.tolist(), frame_w, frame_h)
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area < 18_000:
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


def candidate_indices(frame_count: int, scan_count: int) -> list[int]:
    frame_count = max(1, int(frame_count))
    scan_count = max(1, int(scan_count))
    positions = np.linspace(0.08, 0.92, scan_count)
    return sorted({int(round((frame_count - 1) * float(pos))) for pos in positions})


def choose_frames(capture: cv2.VideoCapture, model: YOLO, frame_count: int, num_frames: int, scan_count: int) -> list[int]:
    frame_area = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scored: list[tuple[float, int, int]] = []
    for index in candidate_indices(frame_count, scan_count):
        frame = load_frame(capture, index)
        vehicles = detect_vehicles(model, frame, imgsz=960)
        top_area = sum(item.area for item in vehicles[:3])
        score = float(top_area) / max(1, frame_area)
        scored.append((score, len(vehicles), index))
    scored.sort(reverse=True)

    selected: list[int] = []
    min_gap = max(20, frame_count // max(8, num_frames * 3))
    for _score, _count, index in scored:
        if any(abs(index - existing) < min_gap for existing in selected):
            continue
        selected.append(index)
        if len(selected) >= num_frames:
            break
    if len(selected) < num_frames:
        for _score, _count, index in scored:
            if index not in selected:
                selected.append(index)
                if len(selected) >= num_frames:
                    break
    return sorted(selected[:num_frames])


def iter_crop_variants(frame: np.ndarray, vehicle: VehicleCandidate) -> list[dict]:
    frame_h, frame_w = frame.shape[:2]
    expanded = expand_box(vehicle.bbox, frame_w, frame_h)
    x1, y1, x2, y2 = expanded
    base_crop = frame[y1:y2, x1:x2]

    variants = [
        {
            "name": "vehicle_full",
            "image": base_crop,
            "origin": (x1, y1),
        }
    ]

    crop_h, crop_w = base_crop.shape[:2]
    if crop_h >= 80 and crop_w >= 120:
        focus_x1 = int(round(crop_w * 0.08))
        focus_x2 = int(round(crop_w * 0.92))
        focus_y1 = int(round(crop_h * 0.35))
        focus_y2 = crop_h
        focus_crop = base_crop[focus_y1:focus_y2, focus_x1:focus_x2]
        if focus_crop.size:
            variants.append(
                {
                    "name": "vehicle_lower_focus",
                    "image": focus_crop,
                    "origin": (x1 + focus_x1, y1 + focus_y1),
                }
            )
    return variants


def upscale_for_lpr(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= 0:
        return image, 1.0
    scale = min(3.0, max(1.0, 960.0 / float(longest)))
    if scale <= 1.01:
        return image, 1.0
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_CUBIC)
    return resized, float(scale)


def run_lpr_on_crop(catcher: lpr3.LicensePlateCatcher, crop: np.ndarray) -> tuple[list, float]:
    upscaled, scale = upscale_for_lpr(crop)
    results = catcher(upscaled)
    return results, scale


def translate_plate_box(box: Iterable[float], origin: tuple[int, int], scale: float, frame_w: int, frame_h: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    ox, oy = origin
    translated = [
        ox + x1 / max(scale, 1e-6),
        oy + y1 / max(scale, 1e-6),
        ox + x2 / max(scale, 1e-6),
        oy + y2 / max(scale, 1e-6),
    ]
    return clip_box(translated, frame_w, frame_h)


def unpack_plate_result(raw: object) -> dict[str, object]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        code, confidence, plate_type, box = raw[:4]
        detect_confidence = raw[4] if len(raw) >= 5 else 0.0
        return {
            "plate_text": str(code),
            "plate_confidence": float(confidence),
            "plate_type": int(plate_type),
            "plate_box": box,
            "plate_detect_confidence": float(detect_confidence),
        }
    raise TypeError(f"Unsupported HyperLPR result format: {type(raw)!r}")


def annotate_frame(frame: np.ndarray, vehicles: list[VehicleCandidate], detections: list[PlateDetection]) -> np.ndarray:
    canvas = frame.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3)
        label = f"{vehicle.class_name} {vehicle.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
    for item in detections:
        px1, py1, px2, py2 = item.plate_box_in_frame
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (40, 220, 40), 3)
        text = f"{item.plate_text} {item.plate_confidence:.2f}"
        cv2.putText(canvas, text, (px1, min(canvas.shape[0] - 12, py2 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 40), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame_dir = ensure_dir(output_dir / "frames")
    annotated_dir = ensure_dir(output_dir / "annotated")
    crop_dir = ensure_dir(output_dir / "crops")

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.yolo.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yolo = YOLO(str(args.yolo))
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

    selected_frames = choose_frames(capture, yolo, frame_count, args.num_frames, args.scan_count)
    summary: dict[str, object] = {
        "video": str(args.video),
        "video_meta": {
            "frame_count": frame_count,
            "fps": fps,
            "width": frame_w,
            "height": frame_h,
            "duration_sec": round(frame_count / max(fps, 1.0), 3),
        },
        "selected_frame_indices": selected_frames,
        "frames": [],
    }

    for frame_index in selected_frames:
        frame = load_frame(capture, frame_index)
        timestamp_sec = float(frame_index) / max(fps, 1.0)
        frame_name = f"frame_{frame_index:04d}_{timestamp_sec:06.2f}s"
        frame_path = frame_dir / f"{frame_name}.jpg"
        annotated_path = annotated_dir / f"{frame_name}_annotated.jpg"
        cv2.imwrite(str(frame_path), frame)

        vehicles = detect_vehicles(yolo, frame)
        frame_detections: list[PlateDetection] = []
        saved_crops = 0
        for vehicle_rank, vehicle in enumerate(vehicles[: args.max_vehicles], start=1):
            for variant in iter_crop_variants(frame, vehicle):
                crop_img = variant["image"]
                if crop_img.size == 0:
                    continue
                results, scale = run_lpr_on_crop(catcher, crop_img)
                for plate_rank, plate in enumerate(results, start=1):
                    plate_data = unpack_plate_result(plate)
                    plate_box = translate_plate_box(
                        plate_data["plate_box"],
                        variant["origin"],
                        scale,
                        frame_w,
                        frame_h,
                    )
                    crop_name = f"{frame_name}_v{vehicle_rank:02d}_{variant['name']}_p{plate_rank:02d}.jpg"
                    crop_path = crop_dir / crop_name
                    if not crop_path.exists():
                        cv2.imwrite(str(crop_path), crop_img)
                        saved_crops += 1
                    frame_detections.append(
                        PlateDetection(
                            frame_index=frame_index,
                            timestamp_sec=round(timestamp_sec, 3),
                            frame_path=str(frame_path),
                            annotated_path=str(annotated_path),
                            vehicle_box=vehicle.bbox,
                            vehicle_confidence=round(vehicle.confidence, 4),
                            vehicle_class=vehicle.class_name,
                            crop_strategy=str(variant["name"]),
                            plate_text=str(plate_data["plate_text"]),
                            plate_confidence=round(float(plate_data["plate_confidence"]), 4),
                            plate_detect_confidence=round(float(plate_data["plate_detect_confidence"]), 4),
                            plate_type=PLATE_TYPE_LABELS.get(int(plate_data["plate_type"]), "unknown"),
                            plate_box_in_frame=plate_box,
                            crop_path=str(crop_path),
                        )
                    )
            if not frame_detections and saved_crops < 2:
                crop_name = f"{frame_name}_v{vehicle_rank:02d}_vehicle.jpg"
                crop_path = crop_dir / crop_name
                x1, y1, x2, y2 = expand_box(vehicle.bbox, frame_w, frame_h)
                cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
                saved_crops += 1

        annotated = annotate_frame(frame, vehicles[: args.max_vehicles], frame_detections)
        cv2.imwrite(str(annotated_path), annotated)
        summary["frames"].append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "frame_path": str(frame_path),
                "annotated_path": str(annotated_path),
                "vehicle_candidates": [asdict(vehicle) for vehicle in vehicles[: args.max_vehicles]],
                "detections": [asdict(item) for item in frame_detections],
            }
        )

    capture.release()

    summary["total_plate_detections"] = sum(len(item["detections"]) for item in summary["frames"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    print(f"Selected frames: {selected_frames}")
    print(f"Total detections: {summary['total_plate_detections']}")


if __name__ == "__main__":
    main()
