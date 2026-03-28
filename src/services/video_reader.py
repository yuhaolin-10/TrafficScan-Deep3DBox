from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VideoInfo:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_s: float
    codec: str = ""
    preview_frame_index: int = 0


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_s: float
    frame: np.ndarray


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def supported_media_path(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in VIDEO_EXTS or suffix in IMAGE_EXTS


def _candidate_video_paths(path: str | Path):
    source = Path(path).expanduser()
    candidates = []
    seen = set()
    for item in [source, source.resolve(strict=False)]:
        for text_value in [str(item), item.as_posix()]:
            if text_value and text_value not in seen:
                seen.add(text_value)
                candidates.append(text_value)
    return source, candidates


def _open_capture(path: str | Path):
    source, candidate_paths = _candidate_video_paths(path)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source.resolve(strict=False)}")

    backend_names = ["CAP_ANY", "CAP_FFMPEG", "CAP_MSMF"]
    tried = []
    for candidate in candidate_paths:
        for backend_name in backend_names:
            backend = getattr(cv2, backend_name, None)
            if backend is None:
                continue
            capture = cv2.VideoCapture(candidate, backend)
            if capture.isOpened():
                return capture
            tried.append(f"{backend_name}:{candidate}")
            capture.release()

        capture = cv2.VideoCapture(candidate)
        if capture.isOpened():
            return capture
        tried.append(f"default:{candidate}")
        capture.release()

    tried_text = "; ".join(tried[:6])
    if len(tried) > 6:
        tried_text += "; ..."
    raise RuntimeError(
        f"Failed to open video: {source.resolve(strict=False)} | tried={tried_text or 'none'}"
    )


def read_video_info(path: str | Path) -> VideoInfo:
    source = Path(path)
    capture = _open_capture(source)
    try:
        frame_count = int(max(0.0, float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(max(0.0, float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        height = int(max(0.0, float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        capture.release()

    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    duration_s = (frame_count / fps) if fps > 1e-6 and frame_count > 0 else 0.0
    preview_frame_index = 0
    if frame_count > 1:
        preview_frame_index = min(frame_count - 1, frame_count // 2)

    return VideoInfo(
        path=str(source),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_s=duration_s,
        codec=codec,
        preview_frame_index=preview_frame_index,
    )


def read_video_frame(path: str | Path, frame_index: Optional[int] = None) -> np.ndarray:
    capture = _open_capture(path)
    try:
        if frame_index is not None and frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from video: {path}")
        return frame
    finally:
        capture.release()


def read_preview_frame(path: str | Path, info: Optional[VideoInfo] = None) -> tuple[VideoInfo, np.ndarray]:
    video_info = info or read_video_info(path)
    frame = read_video_frame(path, frame_index=video_info.preview_frame_index)
    return video_info, frame


def iter_video_frames(
    path: str | Path,
    *,
    start_frame: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[VideoFrame]:
    capture = _open_capture(path)
    emitted = 0
    try:
        start_frame = max(0, int(start_frame))
        stride = max(1, int(stride))
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            timestamp_s = (frame_index / fps) if fps > 1e-6 else 0.0
            yield VideoFrame(frame_index=frame_index, timestamp_s=timestamp_s, frame=frame)
            emitted += 1
            if max_frames is not None and emitted >= int(max_frames):
                break
            if stride > 1:
                for _ in range(stride - 1):
                    skipped_ok = capture.grab()
                    if not skipped_ok:
                        return
                    frame_index += 1
            frame_index += 1
    finally:
        capture.release()


def video_info_to_dict(info: VideoInfo) -> dict:
    return asdict(info)


class VideoFrameSession:
    def __init__(self, path: str | Path, info: Optional[VideoInfo] = None):
        self.path = Path(path).resolve(strict=False)
        self.info = info or read_video_info(self.path)
        self._capture = _open_capture(self.path)
        self._current_frame_index: Optional[int] = None
        self._last_frame: Optional[np.ndarray] = None

    def close(self):
        capture = self._capture
        self._capture = None
        self._current_frame_index = None
        self._last_frame = None
        if capture is not None:
            capture.release()

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _require_capture(self):
        if not self.is_open():
            raise RuntimeError(f"Video session is closed: {self.path}")
        return self._capture

    def _clamp_frame_index(self, frame_index: int) -> int:
        target = max(0, int(frame_index))
        if int(self.info.frame_count or 0) > 0:
            target = min(target, int(self.info.frame_count) - 1)
        return target

    def read_frame(self, frame_index: int) -> np.ndarray:
        capture = self._require_capture()
        target = self._clamp_frame_index(frame_index)

        if self._current_frame_index == target and self._last_frame is not None:
            return self._last_frame.copy()

        sequential = self._current_frame_index is not None and target == (self._current_frame_index + 1)
        if not sequential:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(target))

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {target} from video: {self.path}")

        self._current_frame_index = int(target)
        self._last_frame = frame.copy()
        return frame.copy()

    def read_next_frame(self) -> tuple[int, np.ndarray]:
        target = 0 if self._current_frame_index is None else self._current_frame_index + 1
        frame = self.read_frame(target)
        return int(self._current_frame_index or 0), frame

    def current_frame_index(self) -> int:
        return int(self._current_frame_index or 0)

    def timestamp_s_for_frame(self, frame_index: int) -> float:
        fps = float(self.info.fps or 0.0)
        if fps <= 1e-6:
            return 0.0
        return float(self._clamp_frame_index(frame_index)) / fps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
