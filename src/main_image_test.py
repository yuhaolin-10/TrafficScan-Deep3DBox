import os
from pathlib import Path

from core.lane_segmenter import LaneSegmenter
from core.vehicle_detector_deep3dbox import VehicleDetector3D
from core.violation_checker import ViolationChecker
from services.pipeline import process_image


def main():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    lane_model_path = project_root / "src" / "models" / "best.pt"
    vehicle_model_path = project_root / "src" / "models" / "yolo11l.pt"
    image_dir = project_root / "images" / "test_images" / "illegal"
    output_dir = project_root / "images" / "integrated_test_results"
    os.makedirs(output_dir, exist_ok=True)

    print("=== TrafficScan Deep3DBox image test ===")
    try:
        lane_detector = LaneSegmenter(str(lane_model_path))
        vehicle_detector = VehicleDetector3D(str(vehicle_model_path))
        violation_checker = ViolationChecker(threshold=0.3)
    except Exception as exc:
        print(f"Model initialization failed: {exc}")
        return

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in: {image_dir}")
        return

    print(f"Found {len(image_files)} images")
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        try:
            result = process_image(
                image_path=img_path,
                lane_detector=lane_detector,
                vehicle_detector=vehicle_detector,
                violation_checker=violation_checker,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        print(
            f"  vehicles={result['vehicle_count']} violations={result['violation_count']} saved={result['processed_path']}"
        )

    print("Done")


if __name__ == "__main__":
    main()
