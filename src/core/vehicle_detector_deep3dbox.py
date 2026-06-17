from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .tensorflow_runtime import configure_tensorflow_runtime

tf = None
slim = None


def _ensure_tensorflow():
    global tf, slim
    if tf is not None and slim is not None:
        return tf, slim
    try:
        configure_tensorflow_runtime()
        import tensorflow.compat.v1 as tf_mod
        import tf_slim as slim_mod
    except ImportError as exc:
        raise ImportError(
            "Deep3DBox requires tensorflow and tf_slim. Activate the environment that installs them before running this detector."
        ) from exc
    tf_mod.disable_v2_behavior()
    tf = tf_mod
    slim = slim_mod
    return tf, slim

BIN = 2
NORM_H = 224
NORM_W = 224
MEAN_BGR = np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)

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
    "motorcycle": "Cyclist",
    "bicycle": "Cyclist",
}


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


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
) -> Optional[list[tuple[int, int]]]:
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


class Deep3DBoxModel:
    def __init__(self, checkpoint_path: Path) -> None:
        _ensure_tensorflow()
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

    def _build_model(self, inputs):
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

    def _run_batch(self, batch: np.ndarray):
        return self.session.run(
            [self.dimension, self.orientation, self.confidence],
            feed_dict={self.inputs: batch},
        )

    def predict_batch(self, patches: np.ndarray):
        batch = np.asarray(patches, dtype=np.float32)
        if batch.ndim == 3:
            batch = np.expand_dims(batch, 0)
        if batch.ndim != 4:
            raise ValueError(f"Deep3DBox batch must have shape [N, 224, 224, 3], got {batch.shape}")
        try:
            return self._run_batch(batch)
        except tf.errors.ResourceExhaustedError:
            if batch.shape[0] <= 1:
                raise

            midpoint = batch.shape[0] // 2
            left = self.predict_batch(batch[:midpoint])
            right = self.predict_batch(batch[midpoint:])
            return tuple(np.concatenate([lhs, rhs], axis=0) for lhs, rhs in zip(left, right))

    def predict(self, patch: np.ndarray):
        return self.predict_batch(patch)


class VehicleDetector3D:
    def __init__(
        self,
        model_path,
        checkpoint_path=None,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        min_box_side=15,
        min_box_area=280,
        focal_scale=1.2,
        depth_estimator=None,
    ):
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.min_box_side = int(min_box_side)
        self.min_box_area = int(min_box_area)
        self.focal_scale = float(focal_scale)
        self.depth_estimator = depth_estimator
        self.device = 0 if torch.cuda.is_available() else "cpu"

        project_root = Path(__file__).resolve().parents[2]
        default_checkpoint = project_root / "external" / "deep3dbox_demo_model" / "demo_model"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint
        if not self.checkpoint_path.with_suffix(".index").exists():
            raise FileNotFoundError(f"Deep3DBox checkpoint not found: {self.checkpoint_path}")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.model_path}")

        self.yolo_model = YOLO(self.model_path)
        self.deep3dbox_model = Deep3DBoxModel(self.checkpoint_path)

    def _release_torch_gpu_cache(self) -> None:
        if self.device == "cpu" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            return

    def _project_box(
        self,
        dimensions: np.ndarray,
        location: list[float],
        ry: float,
        proj_matrix: np.ndarray,
        box_2d: list[tuple[int, int]],
        mask_polygon=None,
    ):
        h, w, l = [float(v) for v in dimensions.tolist()]
        dx = l / 2.0
        dy = h / 2.0
        dz = w / 2.0

        corners_local = np.array(
            [
                [dx, -dy, dz],
                [dx, dy, dz],
                [-dx, dy, dz],
                [-dx, -dy, dz],
                [dx, -dy, -dz],
                [dx, dy, -dz],
                [-dx, dy, -dz],
                [-dx, -dy, -dz],
            ],
            dtype=np.float64,
        )
        rotation = rotation_matrix(float(ry))
        corners_3d = corners_local @ rotation.T
        corners_3d += np.asarray(location, dtype=np.float64)

        corners_2d = []
        for x, y, z in corners_3d:
            point = np.array([x, y, z, 1.0], dtype=np.float64)
            projected = proj_matrix @ point
            if projected[2] <= 1e-6:
                projected_xy = np.array([(box_2d[0][0] + box_2d[1][0]) * 0.5, box_2d[1][1]], dtype=np.float64)
            else:
                projected_xy = projected[:2] / projected[2]
            corners_2d.append(projected_xy)

        corners_2d = np.asarray(corners_2d, dtype=np.float32)
        footprint_2d = corners_2d[[1, 2, 6, 5]].astype(np.float32)
        if abs(cv2.contourArea(footprint_2d.astype(np.float32))) < 1.0:
            x1, y1 = box_2d[0]
            x2, y2 = box_2d[1]
            strip_h = max(2.0, float(y2 - y1) * 0.18)
            footprint_2d = np.array(
                [[float(x1), float(y2)], [float(x2), float(y2)], [float(x2), float(y2) - strip_h], [float(x1), float(y2) - strip_h]],
                dtype=np.float32,
            )

        bbox_center = np.array(
            [
                (float(box_2d[0][0]) + float(box_2d[1][0])) * 0.5,
                (float(box_2d[0][1]) + float(box_2d[1][1])) * 0.5,
            ],
            dtype=np.float32,
        )
        geometry_debug = {
            "vehicle_center_2d": bbox_center,
            "location_3d": np.asarray(location, dtype=np.float32),
        }
        return corners_2d, footprint_2d, geometry_debug

    def detect(self, image, lane_polygons=None):
        height, width = image.shape[:2]
        proj_matrix = build_projection_matrix(width, height, self.focal_scale)
        results = self.yolo_model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            mask_polygons = []
            if result.masks is not None:
                try:
                    masks_xy = result.masks.xy
                    if masks_xy is not None and len(masks_xy) > 0:
                        mask_polygons = list(masks_xy)
                except (AttributeError, ValueError):
                    # 如果masks.xy不存在或有数组比较问题，跳过
                    pass

            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()

            candidates = []
            for idx, (xyxy, cls_id, score) in enumerate(zip(xyxy_all, cls_all, conf_all)):
                coco_class = result.names[int(cls_id)]
                deep3dbox_class = COCO_TO_DEEP3DBOX.get(coco_class)
                if deep3dbox_class is None:
                    continue

                box_2d = clip_box(xyxy, width, height, self.min_box_side, self.min_box_area)
                if box_2d is None:
                    continue

                mask_polygon = None
                if idx < len(mask_polygons):
                    candidate_polygon = np.asarray(mask_polygons[idx], dtype=np.float32)
                    if candidate_polygon.ndim == 2 and candidate_polygon.shape[0] >= 3 and candidate_polygon.shape[1] == 2:
                        mask_polygon = candidate_polygon

                candidates.append(
                    {
                        "xyxy": xyxy,
                        "score": float(score),
                        "coco_class": str(coco_class),
                        "deep3dbox_class": deep3dbox_class,
                        "box_2d": box_2d,
                        "mask_polygon": mask_polygon,
                        "patch": preprocess_patch(image, box_2d),
                    }
                )

            if not candidates:
                continue

            # Torch and TensorFlow share the same GPU in this pipeline, so release
            # cached YOLO memory before running Deep3DBox inference.
            self._release_torch_gpu_cache()
            patch_batch = np.concatenate([item["patch"] for item in candidates], axis=0)
            dim_delta_batch, orientation_batch, confidence_batch = self.deep3dbox_model.predict_batch(patch_batch)

            for candidate, dim_delta, orientation, confidence in zip(candidates, dim_delta_batch, orientation_batch, confidence_batch):
                dimensions = DIMS_AVG[candidate["deep3dbox_class"]] + dim_delta
                alpha = decode_alpha(np.expand_dims(orientation, 0), np.expand_dims(confidence, 0))
                theta_ray = calc_theta_ray(image, candidate["box_2d"], proj_matrix)

                try:
                    location, _ = calc_location(dimensions, proj_matrix, candidate["box_2d"], alpha, theta_ray)
                except Exception:
                    continue

                if not np.all(np.isfinite(location)) or float(location[2]) <= 0.0:
                    continue

                ry = _normalize_angle(alpha + theta_ray)
                corners_2d, footprint_2d, geometry_debug = self._project_box(
                    dimensions=dimensions,
                    location=location,
                    ry=ry,
                    proj_matrix=proj_matrix,
                    box_2d=candidate["box_2d"],
                    mask_polygon=candidate["mask_polygon"],
                )

                det = {
                    "bbox": [float(v) for v in candidate["xyxy"].tolist()],
                    "type": candidate["coco_class"],
                    "conf": float(candidate["score"]),
                    "dimensions_hwl": [float(v) for v in dimensions.tolist()],
                    "location_3d": [float(v) for v in location],
                    "alpha": float(alpha),
                    "theta_ray": float(theta_ray),
                    "yaw": float(ry),
                    "corners_2d": np.round(corners_2d).astype(np.int32),
                    "footprint_2d": np.round(footprint_2d).astype(np.int32),
                    "geometry_debug": geometry_debug,
                    "yaw_debug": {
                        "magnitude": float(abs(ry)),
                    },
                }
                if candidate["mask_polygon"] is not None:
                    det["mask_polygon_2d"] = np.round(candidate["mask_polygon"]).astype(np.int32)
                detections.append(det)

        return detections