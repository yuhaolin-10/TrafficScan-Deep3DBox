from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_source = project_root / "images"
    default_output = project_root / "data" / "obb_preview"

    parser = argparse.ArgumentParser(
        description="Run YOLO11 OBB inference on a directory of images and save preview overlays."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Image file or directory to process. Default: {default_source}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for rendered images and JSON summary. Default: {default_output}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11l-obb.pt",
        help="YOLO OBB model path or weight name. Default: yolo11l-obb.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold passed to Ultralytics.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size passed to Ultralytics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Inference device. Use "auto", "cpu", "0", "0,1", etc.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images to process. 0 means no limit.",
    )
    parser.add_argument(
        "--class-contains",
        action="append",
        default=[],
        help="Keep only detections whose class name contains this text. Repeatable.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def collect_images(source: Path, output_dir: Path) -> list[Path]:
    source = source.resolve()
    output_dir = output_dir.resolve()

    if source.is_file():
        return [source]

    images: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass
        images.append(path)
    return sorted(images)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 220, 255),
        (0, 180, 255),
        (70, 200, 70),
        (255, 180, 70),
        (120, 160, 255),
        (200, 120, 255),
        (255, 120, 140),
        (140, 255, 160),
    ]
    return palette[class_id % len(palette)]


def draw_label(image: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    padding = 4
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = anchor
    x = max(4, x)
    y = max(th + 8, y)
    top_left = (x, y - th - padding * 2)
    bottom_right = (x + tw + padding * 2, y)
    cv2.rectangle(image, top_left, bottom_right, color, -1)
    cv2.putText(
        image,
        text,
        (x + padding, y - padding),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(
    image: np.ndarray,
    result,
    names: dict[int, str],
    class_filters: list[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rendered = image.copy()
    detections: list[dict[str, object]] = []
    filters = [token.lower() for token in (class_filters or []) if token.strip()]

    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        draw_label(rendered, "No OBB detections", (10, 28), (90, 90, 90))
        return rendered, detections

    polygons = obb.xyxyxyxy.cpu().numpy()
    boxes = obb.xywhr.cpu().numpy()
    confidences = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)

    for polygon, box, conf, cls_id in zip(polygons, boxes, confidences, classes):
        class_name = names.get(int(cls_id), str(cls_id))
        class_name_lower = class_name.lower()
        if filters and not any(token in class_name_lower for token in filters):
            continue

        pts = np.round(polygon).astype(np.int32).reshape(-1, 2)
        color = color_for_class(int(cls_id))
        cv2.polylines(rendered, [pts], True, color, 2, cv2.LINE_AA)

        center = np.round(box[:2]).astype(np.int32)
        w, h, angle = float(box[2]), float(box[3]), float(box[4])
        direction_len = max(18.0, min(max(w, h) * 0.35, 48.0))
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        end = np.round(center.astype(np.float32) + direction * direction_len).astype(np.int32)
        cv2.arrowedLine(
            rendered,
            tuple(center.tolist()),
            tuple(end.tolist()),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.22,
        )

        label = f"{class_name} {float(conf):.2f}"
        top_left = tuple(np.min(pts, axis=0).tolist())
        draw_label(rendered, label, (int(top_left[0]), int(top_left[1]) - 4), color)

        detections.append(
            {
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(conf),
                "xywhr": [float(v) for v in box.tolist()],
                "polygon": pts.astype(int).tolist(),
            }
        )

    if not detections:
        draw_label(rendered, "No filtered OBB detections", (10, 28), (90, 90, 90))

    return rendered, detections


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source, output_dir)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise SystemExit(f"No images found under: {source}")

    device = resolve_device(args.device)
    print(f"[OBB] Loading model: {args.model}")
    print(f"[OBB] Device: {device}")
    print(f"[OBB] Images: {len(images)}")
    model = YOLO(args.model)

    summary: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print("  -> skip: failed to read")
            summary.append(
                {
                    "image": str(image_path),
                    "output": None,
                    "error": "failed_to_read",
                    "detections": [],
                }
            )
            continue

        results = model.predict(
            source=image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )
        result = results[0]
        rendered, detections = annotate_image(
            image,
            result,
            model.names,
            class_filters=args.class_contains,
        )

        relative = image_path.relative_to(source) if source.is_dir() else Path(image_path.name)
        save_path = output_dir / relative
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), rendered):
            raise RuntimeError(f"Failed to write rendered preview: {save_path}")

        print(f"  -> detections: {len(detections)} | saved: {save_path}")
        summary.append(
            {
                "image": str(image_path),
                "output": str(save_path),
                "detections": detections,
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OBB] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
