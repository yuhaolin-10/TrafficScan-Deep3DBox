import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.vehicle_detector_3d import VehicleDetector3D as LegacyVehicleDetector3D
from core.vehicle_detector_deep3dbox import VehicleDetector3D as Deep3DBoxVehicleDetector3D


def test_legacy_vehicle_detector_module_proxies_to_deep3dbox_detector():
    assert LegacyVehicleDetector3D is Deep3DBoxVehicleDetector3D
