"""Memory-conscious reference video sampling and basic visual statistics."""

from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


class ReferenceAnalyzer:
    def __init__(
        self,
        sample_seconds: float = 2.0,
        max_dimension: int = 640,
        max_ocr_samples: int = 8,
        scene_cut_threshold: float = 0.55,
    ) -> None:
        if sample_seconds <= 0 or max_dimension <= 0 or max_ocr_samples < 0 or not 0 < scene_cut_threshold <= 1:
            raise ValueError("Invalid frame-analysis configuration")
        self.sample_seconds = sample_seconds
        self.max_dimension = max_dimension
        self.max_ocr_samples = max_ocr_samples
        self.scene_cut_threshold = scene_cut_threshold

    def _read_text(self, frame: Any, cv2: Any, frame_time: float) -> list[dict[str, Any]]:
        executable = shutil.which("tesseract")
        if not executable:
            return []
        height, width = frame.shape[:2]
        region_top = int(height * 0.55)
        caption_roi = frame[region_top:, :]
        encoded, image_bytes = cv2.imencode(".png", caption_roi)
        if not encoded:
            return []
        try:
            result = subprocess.run(
                [executable, "stdin", "stdout", "--psm", "6", "tsv"],
                input=image_bytes.tobytes(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if result.returncode:
            return []
        regions = []
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf", -1))
                x, y, w, h = (int(row.get(k, 0)) for k in ("left", "top", "width", "height"))
            except (TypeError, ValueError):
                continue
            if text and confidence >= 40 and w > 0 and h > 0:
                regions.append(
                    {
                        "text": text,
                        "confidence": round(confidence / 100, 3),
                        "time": round(frame_time, 3),
                        "box": {
                            "x": round(x / width, 3),
                            "y": round((region_top + y) / height, 3),
                            "width": round(w / width, 3),
                            "height": round(h / height, 3),
                        },
                    }
                )
        return regions

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
        frame_differences: list[float] = []
        cuts: list[float] = []
        caption_candidates: list[float] = []
        composition_x: list[float] = []
        composition_y: list[float] = []
        hue_histograms = []
        text_regions: list[dict[str, Any]] = []
        previous_gray = None
        previous_histogram = None
        sample_index = 0
        ocr_stride = max(1, math.ceil(duration / self.sample_seconds / max(1, self.max_ocr_samples)))
        ocr_available = self.max_ocr_samples > 0 and shutil.which("tesseract") is not None
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
                    saturation_mask = cv2.inRange(hsv[:, :, 1], 32, 255)
                    histogram = cv2.calcHist([hsv], [0], saturation_mask, [12], [0, 180])
                    cv2.normalize(histogram, histogram)
                    hue_histograms.append(histogram[:, 0].astype(float))
                    if previous_gray is not None:
                        frame_delta = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255)
                        frame_differences.append(frame_delta)
                        flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion.append(float(np.mean(flow_magnitude) / max(gray.shape)))
                        histogram_delta = float(
                            cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
                        )
                        if histogram_delta > self.scene_cut_threshold:
                            cuts.append(timestamp)
                    previous_gray = gray
                    previous_histogram = histogram
                    edges_full = cv2.Canny(gray, 80, 180)
                    edge_y, edge_x = np.nonzero(edges_full)
                    if edge_x.size:
                        composition_x.append(float(np.mean(edge_x) / max(1, gray.shape[1] - 1)))
                        composition_y.append(float(np.mean(edge_y) / max(1, gray.shape[0] - 1)))
                    bottom = gray[int(gray.shape[0] * 0.62) :, :]
                    edges = cv2.Canny(bottom, 80, 180)
                    row_density = np.mean(edges > 0, axis=1)
                    caption_candidates.append(float(np.max(row_density)) if row_density.size else 0.0)
                    if (
                        ocr_available
                        and sample_index % ocr_stride == 0
                        and len(text_regions) < self.max_ocr_samples * 24
                    ):
                        text_regions.extend(self._read_text(frame, cv2, timestamp))
                sample_index += 1
                timestamp += self.sample_seconds
        finally:
            capture.release()
        if not brightness:
            raise ValueError(f"No decodable frames in {path}")
        cut_rate = len(cuts) / duration
        observations = [
            "Fixed-interval frame samples; scene cuts use hue-histogram distance and remain candidates, not confirmed edits.",
            "Motion is estimated with dense optical-flow magnitude; movement is not classified as camera or subject motion.",
        ]
        if ocr_available:
            observations.append(
                "Local Tesseract OCR sampled caption-region frames; its text and boxes are heuristic and may be inaccurate."
            )
        else:
            observations.append(
                "Tesseract was unavailable; caption-region estimates use image edges, with no OCR or font identification."
            )
        hue_totals = np.sum(hue_histograms, axis=0) if hue_histograms else np.zeros(12)
        dominant_hues = [round((int(index) + 0.5) * 15, 1) for index in np.argsort(hue_totals)[-3:][::-1]]
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
            "frame_difference": float(np.mean(frame_differences)) if frame_differences else 0.0,
            "caption_region_score": float(np.mean(caption_candidates)) if caption_candidates else 0.0,
            "text_regions": text_regions,
            "ocr_sample_count": min(self.max_ocr_samples, math.ceil(sample_index / ocr_stride)) if ocr_available else 0,
            "composition_centroid": {
                "x": float(np.mean(composition_x)) if composition_x else None,
                "y": float(np.mean(composition_y)) if composition_y else None,
            },
            "dominant_hues_degrees": dominant_hues,
            "transition_style": "unknown",
            "caption_style": "unknown",
            "observations": observations,
        }
