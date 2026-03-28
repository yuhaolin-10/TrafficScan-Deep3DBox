from pathlib import Path

from core.lane_segmenter import LaneSegmenter
from core.vehicle_detector_deep3dbox import VehicleDetector3D
from core.violation_checker import ViolationChecker
from services.database_manager import DatabaseManager
from services.pipeline import process_image


def main():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    lane_model_path = project_root / "src" / "models" / "best.pt"
    vehicle_model_path = project_root / "src" / "models" / "yolo11l.pt"
    input_dir = project_root / "images" / "test_images" / "illegal"
    output_dir = project_root / "data" / "images"
    db_path = project_root / "data" / "db" / "traffic_scan.db"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    lane_detector = LaneSegmenter(str(lane_model_path))
    vehicle_detector = VehicleDetector3D(str(vehicle_model_path))
    violation_checker = ViolationChecker(threshold=0.3)
    db_manager = DatabaseManager(str(db_path))

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Start processing {len(image_files)} images")
    try:
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
                record_id = db_manager.persist_result(str(img_path), result)
                print(
                    f"  record={record_id} vehicles={result['vehicle_count']} violations={result['violation_count']}"
                )
            except Exception as exc:
                print(f"  Failed: {exc}")
    finally:
        db_manager.close()

    print("All tasks completed")


if __name__ == "__main__":
    main()
