"""Memory-conscious reference video sampling and basic visual statistics."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


class ReferenceAnalyzer:
    def __init__(self, sample_seconds: float = 2.0, max_dimension: int = 640) -> None:
        if sample_seconds <= 0 or max_dimension <= 0:
            raise ValueError("Sampling interval and image size must be positive")
        self.sample_seconds = sample_seconds
        self.max_dimension = max_dimension

    def analyze(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg and ffprobe are required for reference analysis")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True, timeout=30)
        duration = float(probe.stdout.strip())
        if duration <= 0:
            raise ValueError(f"Invalid duration for {path}")
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install opencv-python and numpy to analyze visual frames") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {path}")
        brightness: list[float] = []
        contrast: list[float] = []
        saturation: list[float] = []
        try:
            timestamp = 0.0
            while timestamp < duration:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if ok:
                    h, w = frame.shape[:2]
                    scale = min(1.0, self.max_dimension / max(h, w))
                    if scale < 1:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    value = hsv[:, :, 2].astype(np.float32) / 255
                    brightness.append(float(np.mean(value)))
                    contrast.append(float(np.std(value)))
                    saturation.append(float(np.mean(hsv[:, :, 1]) / 255))
                timestamp += self.sample_seconds
        finally:
            capture.release()
        if not brightness:
            raise ValueError(f"No decodable frames in {path}")
        return {"source": path.name, "duration": duration, "sample_count": len(brightness), "brightness": float(np.mean(brightness)), "contrast": float(np.mean(contrast)), "saturation": float(np.mean(saturation)), "transition_style": "unknown", "caption_style": "unknown", "observations": ["Color statistics sampled at fixed intervals; typography and transitions are not inferred by this baseline analyzer."]}
