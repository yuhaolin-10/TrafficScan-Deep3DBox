import os
from typing import Callable, Optional

DEFAULT_ENABLE_DEPTH = True
DEFAULT_DEPTH_MODEL_ID = "Intel/zoedepth-nyu-kitti"
DEFAULT_DEPTH_DEVICE = "cuda"
DEFAULT_DEPTH_CACHE_DIR = None
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_DEPTH_IS_METRIC = True


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_depth_estimator_from_env(
    *,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None,
):
    """
    Build a pluggable depth estimator from defaults + environment variables.
    Environment variables (if provided) override defaults.

    Environment variables:
      - TRAFFICSCAN_ENABLE_DEPTH: 1/0, true/false
      - TRAFFICSCAN_DEPTH_MODEL_ID: HuggingFace model id
      - TRAFFICSCAN_DEPTH_DEVICE: cuda/cpu (empty means auto)
      - TRAFFICSCAN_DEPTH_CACHE_DIR: optional cache directory
      - TRAFFICSCAN_DEPTH_SCALE: float scale multiplier
      - TRAFFICSCAN_DEPTH_IS_METRIC: 1/0, true/false
    """
    info_logger = info_logger or (lambda msg: None)
    warn_logger = warn_logger or (lambda msg: None)

    enabled = _as_bool(os.getenv("TRAFFICSCAN_ENABLE_DEPTH"), DEFAULT_ENABLE_DEPTH)
    if not enabled:
        info_logger("[DepthEstimator] disabled")
        return None

    model_id = os.getenv("TRAFFICSCAN_DEPTH_MODEL_ID", DEFAULT_DEPTH_MODEL_ID)
    device = os.getenv("TRAFFICSCAN_DEPTH_DEVICE", DEFAULT_DEPTH_DEVICE)
    cache_dir = os.getenv("TRAFFICSCAN_DEPTH_CACHE_DIR", DEFAULT_DEPTH_CACHE_DIR)
    depth_scale_raw = os.getenv("TRAFFICSCAN_DEPTH_SCALE")
    depth_is_metric = _as_bool(os.getenv("TRAFFICSCAN_DEPTH_IS_METRIC"), DEFAULT_DEPTH_IS_METRIC)

    depth_scale = float(DEFAULT_DEPTH_SCALE)
    if depth_scale_raw is not None:
        try:
            depth_scale = float(depth_scale_raw)
        except Exception:
            warn_logger(
                f"[DepthEstimator] invalid TRAFFICSCAN_DEPTH_SCALE='{depth_scale_raw}', "
                f"fallback to {DEFAULT_DEPTH_SCALE}"
            )

    device = device.strip() if isinstance(device, str) else device
    cache_dir = cache_dir.strip() if isinstance(cache_dir, str) else cache_dir
    device = device or None
    cache_dir = cache_dir or None

    try:
        from core.depth_zoedepth_adapter import ZoeDepthAdapter

        estimator = ZoeDepthAdapter(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir,
            depth_scale=depth_scale,
            output_is_metric=depth_is_metric,
        )
        info_logger(
            f"[DepthEstimator] enabled model={model_id} device={device or 'auto'} "
            f"scale={depth_scale} metric={depth_is_metric}"
        )
        return estimator
    except Exception as exc:
        warn_logger(f"[DepthEstimator] init failed, fallback to prior-only depth: {exc}")
        return None
