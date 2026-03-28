from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_RUNTIME_READY = False
_DLL_DIRECTORY_HANDLES: list[object] = []


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.normcase(str(path.resolve(strict=False)))
        if normalized in seen or not path.is_dir():
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _find_torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent / "lib"


def _candidate_dll_dirs() -> list[Path]:
    prefix = Path(sys.prefix).resolve()
    candidates = [
        prefix / "Library" / "bin",
        prefix / "bin",
        prefix / "Lib" / "site-packages" / "torch" / "lib",
    ]
    torch_lib_dir = _find_torch_lib_dir()
    if torch_lib_dir is not None:
        candidates.append(torch_lib_dir)

    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value) / "bin")

    return _unique_existing_paths(candidates)


def configure_tensorflow_runtime() -> list[str]:
    global _RUNTIME_READY
    if _RUNTIME_READY or os.name != "nt":
        return []

    added_paths: list[str] = []
    current_path_entries = {
        os.path.normcase(entry)
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    }

    for dll_dir in _candidate_dll_dirs():
        dll_dir_str = str(dll_dir)
        normalized = os.path.normcase(dll_dir_str)

        if normalized not in current_path_entries:
            os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
            current_path_entries.add(normalized)
            added_paths.append(dll_dir_str)

        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir_str))
            except (FileNotFoundError, OSError):
                continue

    _RUNTIME_READY = True
    return added_paths
