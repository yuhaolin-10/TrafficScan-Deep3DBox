from __future__ import annotations

import warnings

warnings.warn(
    "core.vehicle_detector_3d has been retired. Use core.vehicle_detector_deep3dbox instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.vehicle_detector_deep3dbox import VehicleDetector3D

__all__ = ["VehicleDetector3D"]
