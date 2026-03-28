from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.tensorflow_runtime import configure_tensorflow_runtime

configure_tensorflow_runtime()

import tensorflow.compat.v1 as tf
import tf_slim as slim
import torch
from ultralytics import YOLO

tf.disable_v2_behavior()

BIN = 2
NORM_H = 224
NORM_W = 224

DIMS_AVG = {
    "Cyclist": np.array([1.73532436, 0.58028152, 1.77413709], dtype=np.float32),
    "Van": np.array([2.18928571, 1.90979592, 5.07087755], dtype=np.float32),
    "Tram": np.array([3.56092896, 2.39601093, 18.34125683], dtype=np.float32),
    "Car": np.array([1.52159147, 1.64443089, 3.85813679], dtype=np.float32),
    "Pedestrian": np.array([1.75554637, 0.66860882, 0.87623049], dtype=np.float32),
    "Truck": np.array([3.07392252, 2.63079903, 11.2190799], dtype=np.float32),
}

COCO_TO_DEEP3DBOX = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Van",
    "person": "Pedestrian",
}

MEAN_BGR = np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)


@dataclass
class Detection3D:
    coco_class: str
    deep3dbox_class: str
    score: float
    box_2d: list[tuple[int, int]]
    dimensions: list[float]
    alpha: float
    theta_ray: float
    location: list[float]


def rotation_matrix(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    tx = roll
    ty = yaw
    tz = pitch

    _rx = np.array(
        [[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]],
        dtype=np.float64,
    )
    _rz = np.array(
        [[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return ry.reshape((3, 3))


def create_corners(
    dimension: np.ndarray,
    location: list[float] | None = None,
    rotation: np.ndarray | None = None,
) -> list[list[float]]:
    dx = dimension[2] / 2.0
    dy = dimension[0] / 2.0
    dz = dimension[1] / 2.0

    x_corners: list[float] = []
    y_corners: list[float] = []
    z_corners: list[float] = []

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                x_corners.append(dx * i)
                y_corners.append(dy * j)
                z_corners.append(dz * k)

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float64)

    if rotation is not None:
        corners = np.dot(rotation, corners)

    if location is not None:
        for axis_index, axis_loc in enumerate(location):
            corners[axis_index, :] = corners[axis_index, :] + axis_loc

    return [[corners[0][i], corners[1][i], corners[2][i]] for i in range(8)]


def calc_location(
    dimension: np.ndarray,
    proj_matrix: np.ndarray,
    box_2d: list[tuple[int, int]],
    alpha: float,
    theta_ray: float,
) -> tuple[list[float], list[list[float]]]:
    orient = alpha + theta_ray
    rotation = rotation_matrix(orient)

    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]
    box_corners = [xmin, ymin, xmax, ymax]

    constraints: list[list[list[float]]] = []
    left_constraints: list[list[float]] = []
    right_constraints: list[list[float]] = []
    top_constraints: list[list[float]] = []
    bottom_constraints: list[list[float]] = []

    dx = dimension[2] / 2.0
    dy = dimension[0] / 2.0
    dz = dimension[1] / 2.0

    left_mult = 1
    right_mult = -1

    if np.deg2rad(88) < alpha < np.deg2rad(92):
        left_mult = 1
        right_mult = 1
    elif -np.deg2rad(92) < alpha < -np.deg2rad(88):
        left_mult = -1
        right_mult = -1
    elif -np.deg2rad(90) < alpha < np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    switch_mult = 1 if alpha > 0 else -1

    for i in (-1, 1):
        left_constraints.append([left_mult * dx, i * dy, -switch_mult * dz])
        right_constraints.append([right_mult * dx, i * dy, switch_mult * dz])

    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i * dx, -dy, j * dz])
            bottom_constraints.append([i * dx, dy, j * dz])

    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    corners = [left, top, right, bottom]
                    if len(corners) == len(set(tuple(point) for point in corners)):
                        constraints.append(corners)

    pre_m = np.eye(4, dtype=np.float64)
    best_loc = None
    best_error = np.array([1e09], dtype=np.float64)
    best_x = None

    for constraint in constraints:
        xa, xb, xc, xd = constraint
        x_array = [xa, xb, xc, xd]
        m_array = [np.copy(pre_m) for _ in range(4)]

        a = np.zeros((4, 3), dtype=np.float64)
        b = np.zeros((4, 1), dtype=np.float64)

        for row, index in enumerate([0, 1, 0, 1]):
            x = x_array[row]
            m = m_array[row]
            rx = np.dot(rotation, x)
            m[:3, 3] = rx.reshape(3)
            m = np.dot(proj_matrix, m)

            a[row, :] = m[index, :3] - box_corners[row] * m[2, :3]
            b[row] = box_corners[row] * m[2, 3] - m[index, 3]

        loc, error, _rank, _s = np.linalg.lstsq(a, b, rcond=None)

        if error.size == 0:
            error = np.array([0.0], dtype=np.float64)

        if error < best_error:
            best_loc = loc
            best_error = error
            best_x = x_array

    if best_loc is None or best_x is None:
        raise RuntimeError("3D location solve failed")

    return [best_loc[0][0], best_loc[1][0], best_loc[2][0]], best_x


def project_3d_pt(pt: list[float], cam_to_img: np.ndarray) -> np.ndarray:
    point = np.array(pt, dtype=np.float64)
    point = np.append(point, 1.0)
    point = np.dot(cam_to_img, point)
    point = point[:2] / point[2]
    return point.astype(np.int16)


def plot_2d_box(img: np.ndarray, box_2d: list[tuple[int, int]]) -> None:
    (x1, y1), (x2, y2) = box_2d
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def plot_3d_box(
    img: np.ndarray,
    cam_to_img: np.ndarray,
    ry: float,
    dimension: np.ndarray,
    center: list[float],
) -> None:
    rotation = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, rotation=rotation)
    box_3d = [project_3d_pt(corner, cam_to_img) for corner in corners]

    green = (0, 255, 0)
    blue = (255, 0, 0)

    cv2.line(img, tuple(box_3d[0]), tuple(box_3d[2]), green, 1)
    cv2.line(img, tuple(box_3d[4]), tuple(box_3d[6]), green, 1)
    cv2.line(img, tuple(box_3d[0]), tuple(box_3d[4]), green, 1)
    cv2.line(img, tuple(box_3d[2]), tuple(box_3d[6]), green, 1)

    cv2.line(img, tuple(box_3d[1]), tuple(box_3d[3]), green, 1)
    cv2.line(img, tuple(box_3d[1]), tuple(box_3d[5]), green, 1)
    cv2.line(img, tuple(box_3d[7]), tuple(box_3d[3]), green, 1)
    cv2.line(img, tuple(box_3d[7]), tuple(box_3d[5]), green, 1)

    for idx in range(0, 7, 2):
        cv2.line(img, tuple(box_3d[idx]), tuple(box_3d[idx + 1]), green, 1)

    front_mark = [tuple(box_3d[idx]) for idx in range(4)]
    cv2.line(img, front_mark[0], front_mark[3], blue, 1)
    cv2.line(img, front_mark[1], front_mark[2], blue, 1)


def build_projection_matrix(width: int, height: int, focal_scale: float) -> np.ndarray:
    focal = focal_scale * float(width)
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal, 0.0, cx, 0.0], [0.0, focal, cy, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def calc_theta_ray(
    image: np.ndarray, box_2d: list[tuple[int, int]], proj_matrix: np.ndarray
) -> float:
    width = image.shape[1]
    fovx = 2.0 * np.arctan(width / (2.0 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2.0
    dx = center - (width / 2.0)
    mult = -1.0 if dx < 0 else 1.0
    dx = abs(dx)
    angle = np.arctan((2.0 * dx * np.tan(fovx / 2.0)) / width)
    return float(angle * mult)


def decode_alpha(orientation: np.ndarray, confidence: np.ndarray) -> float:
    max_anc = int(np.argmax(confidence[0]))
    anchors = orientation[0][max_anc]

    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])

    wedge = 2.0 * np.pi / BIN
    angle_offset = angle_offset + max_anc * wedge
    angle_offset = angle_offset % (2.0 * np.pi)
    angle_offset = angle_offset - np.pi / 2.0
    if angle_offset > np.pi:
        angle_offset = angle_offset - (2.0 * np.pi)

    return float(angle_offset)


def preprocess_patch(image: np.ndarray, box_2d: list[tuple[int, int]]) -> np.ndarray:
    (x1, y1), (x2, y2) = box_2d
    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (NORM_W, NORM_H), interpolation=cv2.INTER_CUBIC)
    crop = crop.astype(np.float32, copy=False)
    crop = crop - MEAN_BGR
    return np.expand_dims(crop, 0)


def clip_box(
    box_xyxy: np.ndarray, width: int, height: int, min_box_side: int, min_box_area: int
) -> list[tuple[int, int]] | None:
    x1, y1, x2, y2 = box_xyxy.astype(np.int32).tolist()
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box_side or box_h < min_box_side or box_w * box_h < min_box_area:
        return None

    return [(x1, y1), (x2, y2)]


class Deep3DBoxModel:
    def __init__(self, checkpoint_path: Path) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="inputs")
            self.dimension, self.orientation, self.confidence = self._build_model(self.inputs)

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(graph=self.graph, config=config)
            saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, str(checkpoint_path))

    def _build_model(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        def leaky_relu(x: tf.Tensor, alpha: float) -> tf.Tensor:
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
            weights_regularizer=slim.l2_regularizer(0.0005),
        ):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
            net = slim.max_pool2d(net, [2, 2], scope="pool2")
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
            net = slim.max_pool2d(net, [2, 2], scope="pool3")
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv4")
            net = slim.max_pool2d(net, [2, 2], scope="pool4")
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
            net = slim.max_pool2d(net, [2, 2], scope="pool5")
            conv5 = slim.flatten(net)

            dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope="fc7_d")
            dimension = leaky_relu(dimension, 0.1)
            dimension = slim.dropout(dimension, 0.5, is_training=False, scope="dropout7_d")
            dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope="fc8_d")

            orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope="fc7_o")
            orientation = leaky_relu(orientation, 0.1)
            orientation = slim.dropout(orientation, 0.5, is_training=False, scope="dropout7_o")
            orientation = slim.fully_connected(orientation, BIN * 2, activation_fn=None, scope="fc8_o")
            orientation = tf.reshape(orientation, [-1, BIN, 2])
            orientation = tf.nn.l2_normalize(orientation, axis=2)

            confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope="fc7_c")
            confidence = leaky_relu(confidence, 0.1)
            confidence = slim.dropout(confidence, 0.5, is_training=False, scope="dropout7_c")
            confidence = slim.fully_connected(confidence, BIN, activation_fn=None, scope="fc8_c")
            confidence = tf.nn.softmax(confidence)

        return dimension, orientation, confidence

    def predict(self, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.session.run(
            [self.dimension, self.orientation, self.confidence],
            feed_dict={self.inputs: patch},
        )


def load_yolo(yolo_weights: Path) -> YOLO:
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    return YOLO(str(yolo_weights))


def get_default_paths(script_dir: Path) -> dict[str, Path]:
    project_root = script_dir.parent.parent
    return {
        "input_dir": project_root / "images" / "test_images" / "illegal",
        "output_dir": project_root / "data" / "deep3dbox_outputs" / "illegal",
        "checkpoint": project_root / "external" / "deep3dbox_demo_model" / "demo_model",
        "yolo_weights": project_root / "src" / "models" / "yolo11l.pt",
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    defaults = get_default_paths(script_dir)

    parser = argparse.ArgumentParser(description="TrafficScan Deep3DBox illegal-scene test runner")
    parser.add_argument("--input-dir", type=Path, default=defaults["input_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["checkpoint"])
    parser.add_argument("--yolo-weights", type=Path, default=defaults["yolo_weights"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--min-box-side", type=int, default=15)
    parser.add_argument("--min-box-area", type=int, default=280)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    vis_dir = output_dir / "visualizations"
    json_dir = output_dir / "json"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir, json_dir


def image_paths(input_dir: Path) -> list[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes])


def predict_image(
    image_path: Path,
    image: np.ndarray,
    yolo_model: YOLO,
    deep3dbox_model: Deep3DBoxModel,
    conf: float,
    iou: float,
    focal_scale: float,
    min_box_side: int,
    min_box_area: int,
) -> tuple[np.ndarray, list[Detection3D]]:
    height, width = image.shape[:2]
    proj_matrix = build_projection_matrix(width, height, focal_scale)
    draw = image.copy()
    detections_3d: list[Detection3D] = []

    device = 0 if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=640,
        device=device,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for xyxy, cls_id, score in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            coco_class = result.names[int(cls_id)]
            if coco_class not in COCO_TO_DEEP3DBOX:
                continue

            box_2d = clip_box(xyxy, width, height, min_box_side, min_box_area)
            if box_2d is None:
                continue

            patch = preprocess_patch(image, box_2d)
            dim_delta, orientation, confidence = deep3dbox_model.predict(patch)
            deep_class = COCO_TO_DEEP3DBOX[coco_class]
            dimensions = DIMS_AVG[deep_class] + dim_delta[0]
            alpha = decode_alpha(orientation, confidence)
            theta_ray = calc_theta_ray(image, box_2d, proj_matrix)

            try:
                location, _ = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)
            except Exception:
                continue

            if not np.all(np.isfinite(location)) or location[2] <= 0:
                continue

            plot_2d_box(draw, box_2d)
            plot_3d_box(draw, proj_matrix, alpha + theta_ray, dimensions, location)

            text = f"{deep_class} {score:.2f}"
            cv2.putText(
                draw,
                text,
                (box_2d[0][0], max(18, box_2d[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            detections_3d.append(
                Detection3D(
                    coco_class=coco_class,
                    deep3dbox_class=deep_class,
                    score=float(score),
                    box_2d=box_2d,
                    dimensions=[float(x) for x in dimensions.tolist()],
                    alpha=float(alpha),
                    theta_ray=float(theta_ray),
                    location=[float(x) for x in location],
                )
            )

    return draw, detections_3d


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint.with_suffix(".index").exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    vis_dir, json_dir = ensure_dirs(args.output_dir)
    images = image_paths(args.input_dir)
    if args.max_images > 0:
        images = images[: args.max_images]

    yolo_model = load_yolo(args.yolo_weights)
    deep3dbox_model = Deep3DBoxModel(args.checkpoint)

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "yolo_weights": str(args.yolo_weights),
        "focal_scale": args.focal_scale,
        "min_box_side": args.min_box_side,
        "min_box_area": args.min_box_area,
        "images": [],
    }

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        visualization, detections = predict_image(
            image_path=image_path,
            image=image,
            yolo_model=yolo_model,
            deep3dbox_model=deep3dbox_model,
            conf=args.conf,
            iou=args.iou,
            focal_scale=args.focal_scale,
            min_box_side=args.min_box_side,
            min_box_area=args.min_box_area,
        )

        output_image = vis_dir / image_path.name
        cv2.imwrite(str(output_image), visualization)

        image_record = {
            "image": image_path.name,
            "visualization": str(output_image),
            "detections": [
                {
                    "coco_class": det.coco_class,
                    "deep3dbox_class": det.deep3dbox_class,
                    "score": det.score,
                    "box_2d": det.box_2d,
                    "dimensions_hwl": det.dimensions,
                    "alpha": det.alpha,
                    "theta_ray": det.theta_ray,
                    "location_xyz": det.location,
                }
                for det in detections
            ],
        }
        summary["images"].append(image_record)

        image_json = json_dir / f"{image_path.stem}.json"
        image_json.write_text(json.dumps(image_record, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to: {args.output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()


