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
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
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
        motion: list[float] = []
        cuts: list[float] = []
        caption_candidates: list[float] = []
        previous_gray = None
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
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if previous_gray is not None:
                        delta = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255)
                        motion.append(delta)
                        if delta > 0.22:
                            cuts.append(timestamp)
                    previous_gray = gray
                    bottom = gray[int(gray.shape[0] * 0.62) :, :]
                    edges = cv2.Canny(bottom, 80, 180)
                    row_density = np.mean(edges > 0, axis=1)
                    caption_candidates.append(float(np.max(row_density)) if row_density.size else 0.0)
                timestamp += self.sample_seconds
        finally:
            capture.release()
        if not brightness:
            raise ValueError(f"No decodable frames in {path}")
        cut_rate = len(cuts) / duration
        observations = [
            "Fixed-interval frames; abrupt visual differences are treated as scene-change candidates, not confirmed edits.",
            "Bottom-frame edge density is a low-confidence caption-region candidate, not OCR or font recognition.",
        ]
        return {
            "source": path.name,
            "duration": duration,
            "sample_count": len(brightness),
            "brightness": float(np.mean(brightness)),
            "contrast": float(np.mean(contrast)),
            "saturation": float(np.mean(saturation)),
            "cut_rate": cut_rate,
            "scene_change_times": cuts,
            "motion": float(np.mean(motion)) if motion else 0.0,
            "caption_region_score": float(np.mean(caption_candidates)) if caption_candidates else 0.0,
            "transition_style": "unknown",
            "caption_style": "unknown",
            "observations": observations,
        }
