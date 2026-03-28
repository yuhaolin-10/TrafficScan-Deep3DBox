from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.depth_estimator import DepthEstimatorBase, DepthMapResult


class ZoeDepthAdapter(DepthEstimatorBase):
    """
    ZoeDepth adapter for the pluggable DepthEstimator interface.

    Example:
        depth_estimator = ZoeDepthAdapter(
            model_id="Intel/zoedepth-nyu-kitti",
            device="cuda",
            depth_scale=1.0,
            output_is_metric=True,
        )
    """

    def __init__(
        self,
        model_id: str = "Intel/zoedepth-nyu-kitti",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        depth_scale: float = 1.0,
        output_is_metric: bool = True,
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(
                "ZoeDepthAdapter requires `torch` and `transformers`. "
                "Install with: pip install torch transformers"
            ) from exc

        self._torch = torch
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._depth_scale = float(depth_scale)
        self._output_is_metric = bool(output_is_metric)
        self._device = self._resolve_device(device)

        self._processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=cache_dir)
        self._model.to(self._device)
        self._model.eval()

    def _resolve_device(self, device: Optional[str]) -> str:
        if device:
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _to_inputs(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        return {k: v.to(self._device) for k, v in inputs.items()}

    def estimate_depth(self, image_bgr: np.ndarray) -> DepthMapResult:
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be a valid HxWx3 BGR image")

        h, w = image_bgr.shape[:2]
        inputs = self._to_inputs(image_bgr)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            # (B,H,W) -> (B,1,H,W) for interpolation
            pred = outputs.predicted_depth.unsqueeze(1)
            pred = self._torch.nn.functional.interpolate(
                pred,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            depth_map = pred.squeeze(1).squeeze(0).detach().cpu().numpy().astype(np.float32)

        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_map = np.maximum(depth_map, 0.0)

        return DepthMapResult(
            depth_map=depth_map,
            is_metric=self._output_is_metric,
            scale=self._depth_scale,
            meta={
                "adapter": "ZoeDepthAdapter",
                "model_id": self._model_id,
                "device": self._device,
            },
        )

