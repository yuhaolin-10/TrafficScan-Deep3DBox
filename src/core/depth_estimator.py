from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Union, runtime_checkable

import numpy as np


@dataclass
class DepthMapResult:
    """
    Depth estimator standard output.

    Fields:
        depth_map: 2D depth map, expected in meters when is_metric=True.
        is_metric: Whether depth values are already metric.
        scale: Optional scale factor applied before downstream use.
        meta: Extra metadata for debugging/inspection.
    """
    depth_map: np.ndarray
    is_metric: bool = True
    scale: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DepthEstimator(Protocol):
    """
    Pluggable depth estimator interface.

    Implementations should provide `estimate_depth(image_bgr)` and return
    either:
      1) DepthMapResult
      2) np.ndarray depth map (treated as metric depth map)
    """

    def estimate_depth(self, image_bgr: np.ndarray) -> Union[DepthMapResult, np.ndarray]:
        ...


class DepthEstimatorBase:
    """Convenience base class for depth estimators."""

    def estimate_depth(self, image_bgr: np.ndarray) -> Union[DepthMapResult, np.ndarray]:
        raise NotImplementedError

