import argparse
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "external" / "MonoGround"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "monoground_probe"
DEFAULT_IMAGE_DIRS = [
    PROJECT_ROOT / "images" / "test_images",
    PROJECT_ROOT / "data" / "images",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ProbeImage:
    source_path: str
    staged_name: str
    width: Optional[int]
    height: Optional[int]


def _try_import_module(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, str(exc)


def _gather_runtime_info() -> Dict[str, object]:
    torch, torch_err = _try_import_module("torch")
    cv2, cv2_err = _try_import_module("cv2")

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch_version": None if torch is None else getattr(torch, "__version__", "unknown"),
        "torch_cuda_available": None if torch is None else bool(torch.cuda.is_available()),
        "cv2_version": None if cv2 is None else getattr(cv2, "__version__", "unknown"),
        "torch_import_error": torch_err,
        "cv2_import_error": cv2_err,
    }
    return info


def _find_images(paths: Sequence[Path]) -> List[Path]:
    discovered: List[Path] = []
    seen = set()

    for base in paths:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = str(path.resolve()).lower()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(path)

    return discovered


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _probe_image_shape(image_path: Path) -> Dict[str, Optional[int]]:
    cv2, _ = _try_import_module("cv2")
    if cv2 is None:
        return {"width": None, "height": None}

    image = cv2.imread(str(image_path))
    if image is None:
        return {"width": None, "height": None}

    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def _find_checkpoint_candidates(external_root: Path) -> List[Path]:
    if not external_root.exists():
        return []

    candidates: List[Path] = []
    for path in external_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
            continue
        candidates.append(path)
    return sorted(candidates)


def _find_repo_markers(external_root: Path) -> Dict[str, bool]:
    if not external_root.exists():
        return {
            "repo_exists": False,
            "has_readme": False,
            "has_requirements": False,
            "has_tools_dir": False,
            "has_lib_dir": False,
            "has_configs_dir": False,
        }

    return {
        "repo_exists": True,
        "has_readme": (external_root / "README.md").exists(),
        "has_requirements": any((external_root / name).exists() for name in ("requirements.txt", "requirement.txt")),
        "has_tools_dir": (external_root / "tools").exists(),
        "has_lib_dir": (external_root / "lib").exists(),
        "has_configs_dir": any((external_root / name).exists() for name in ("configs", "config", "experiments")),
    }


def _write_calibration_template(output_root: Path, images: Sequence[ProbeImage]) -> Path:
    template_path = output_root / "calibration_template.json"
    if template_path.exists():
        return template_path

    payload = {
        "notes": [
            "MonoGround and similar monocular 3D detectors normally expect camera intrinsics.",
            "If your images come from multiple cameras, add one camera profile per camera or per image group.",
            "If you do not know exact intrinsics, start with an approximate fx/fy and image center, then compare results visually.",
        ],
        "camera_profiles": {
            "fill_me_default": {
                "fx": None,
                "fy": None,
                "cx": None,
                "cy": None,
                "comment": "Example: for a 1920x1080 image you can start from cx=960, cy=540 and estimate fx/fy.",
            }
        },
        "images": [
            {
                "staged_name": image.staged_name,
                "source_path": image.source_path,
                "width": image.width,
                "height": image.height,
                "camera_profile": "fill_me_default",
            }
            for image in images
        ],
    }
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return template_path


def _stage_images(
    image_paths: Sequence[Path],
    output_root: Path,
    *,
    limit: int,
    overwrite: bool,
) -> List[ProbeImage]:
    stage_dir = output_root / "input_images"
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged: List[ProbeImage] = []
    for index, image_path in enumerate(image_paths[:limit], start=1):
        staged_name = f"{index:03d}_{_safe_stem(image_path)}{image_path.suffix.lower()}"
        target_path = stage_dir / staged_name
        if overwrite or not target_path.exists():
            shutil.copy2(image_path, target_path)

        shape = _probe_image_shape(image_path)
        staged.append(
            ProbeImage(
                source_path=str(image_path.resolve()),
                staged_name=staged_name,
                width=shape["width"],
                height=shape["height"],
            )
        )

    manifest_path = output_root / "manifest.json"
    manifest_payload = {
        "project_root": str(PROJECT_ROOT),
        "stage_dir": str(stage_dir.resolve()),
        "image_count": len(staged),
        "images": [asdict(item) for item in staged],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_calibration_template(output_root, staged)
    return staged


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def doctor(args: argparse.Namespace) -> int:
    runtime = _gather_runtime_info()
    image_dirs = [Path(p).resolve() for p in args.image_dirs]
    external_root = Path(args.external_root).resolve()
    output_root = Path(args.output_root).resolve()
    images = _find_images(image_dirs)
    repo_markers = _find_repo_markers(external_root)
    checkpoints = _find_checkpoint_candidates(external_root)

    _print_header("Runtime")
    print(f"Python: {runtime['python']}")
    print(f"Platform: {runtime['platform']}")
    print(f"Executable: {runtime['executable']}")
    print(f"Torch: {runtime['torch_version']}")
    print(f"CUDA Available: {runtime['torch_cuda_available']}")
    print(f"OpenCV: {runtime['cv2_version']}")
    if runtime["torch_import_error"]:
        print(f"Torch Import Error: {runtime['torch_import_error']}")
    if runtime["cv2_import_error"]:
        print(f"OpenCV Import Error: {runtime['cv2_import_error']}")

    _print_header("Inputs")
    for image_dir in image_dirs:
        status = "OK" if image_dir.exists() else "MISSING"
        print(f"{status}: {image_dir}")
    print(f"Discovered images: {len(images)}")
    if images:
        for image in images[: min(5, len(images))]:
            print(f"  - {image}")

    _print_header("MonoGround Repo")
    print(f"Expected repo path: {external_root}")
    for key, value in repo_markers.items():
        print(f"{key}: {value}")
    print(f"Checkpoint candidates: {len(checkpoints)}")
    for ckpt in checkpoints[: min(5, len(checkpoints))]:
        print(f"  - {ckpt}")

    _print_header("Output Workspace")
    print(f"Probe output dir: {output_root}")
    print(f"Exists: {output_root.exists()}")
    print(f"Calibration template exists: {(output_root / 'calibration_template.json').exists()}")
    print(f"Manifest exists: {(output_root / 'manifest.json').exists()}")

    _print_header("Next Steps")
    if not repo_markers["repo_exists"]:
        print("1. Clone or unzip the official MonoGround repo into external/MonoGround.")
    if not checkpoints:
        print("2. Download a MonoGround checkpoint and place it under external/MonoGround.")
    if images and not (output_root / "manifest.json").exists():
        print("3. Run the prepare command to stage a small image set for the probe.")
    print("4. Fill in data/monoground_probe/calibration_template.json with approximate camera intrinsics.")
    print("5. Prefer Linux/WSL for the official MonoGround code path if Windows compilation becomes unstable.")

    return 0


def prepare(args: argparse.Namespace) -> int:
    image_dirs = [Path(p).resolve() for p in args.image_dirs]
    output_root = Path(args.output_root).resolve()
    images = _find_images(image_dirs)
    if not images:
        print("No images found in the selected directories.")
        return 1

    staged = _stage_images(images, output_root, limit=args.limit, overwrite=args.overwrite)

    _print_header("Prepared Probe Workspace")
    print(f"Output root: {output_root}")
    print(f"Staged images: {len(staged)}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    print(f"Calibration template: {output_root / 'calibration_template.json'}")
    print("Recommended next command:")
    print("  conda.bat run -n depth-probe python src/tools/monoground_probe.py doctor")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and diagnose a standalone MonoGround experiment workspace."
    )
    parser.add_argument(
        "--external-root",
        default=str(DEFAULT_EXTERNAL_ROOT),
        help="Path where the official MonoGround repository is expected.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory used to stage MonoGround probe assets.",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_IMAGE_DIRS],
        help="One or more directories to search for source images.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime, repo, checkpoint and image readiness.")
    doctor_parser.set_defaults(handler=doctor)

    prepare_parser = subparsers.add_parser("prepare", help="Copy a small image subset into a dedicated probe folder.")
    prepare_parser.add_argument("--limit", type=int, default=12, help="Maximum number of images to stage.")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite staged files if they already exist.",
    )
    prepare_parser.set_defaults(handler=prepare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
