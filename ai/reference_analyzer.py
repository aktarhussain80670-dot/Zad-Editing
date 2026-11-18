"""Memory-conscious reference video sampling and basic visual statistics."""

from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from services.ffprobe_service import FFprobeService


class ReferenceAnalyzer:
    def __init__(
        self,
        sample_seconds: float = 2.0,
        max_dimension: int = 640,
        max_ocr_samples: int = 8,
        scene_cut_threshold: float = 0.55,
    ) -> None:
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("Sample interval must be positive and finite")

        if max_dimension <= 0:
            raise ValueError("Maximum frame dimension must be positive")

        if max_ocr_samples < 0:
            raise ValueError("Maximum OCR samples cannot be negative")

        if (
            not math.isfinite(scene_cut_threshold)
            or not 0 < scene_cut_threshold <= 1
        ):
            raise ValueError("Scene-cut threshold must be between 0 and 1")

        self.sample_seconds = sample_seconds
        self.max_dimension = max_dimension
        self.max_ocr_samples = max_ocr_samples
        self.scene_cut_threshold = scene_cut_threshold

    def _read_text(
        self,
        frame: Any,
        cv2: Any,
        frame_time: float,
    ) -> list[dict[str, Any]]:
        executable = shutil.which("tesseract")

        if not executable:
            return []

        height, width = frame.shape[:2]

        if height <= 0 or width <= 0:
            return []

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
                text=False,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []

        if result.returncode:
            return []

        output_text = result.stdout.decode("utf-8", errors="replace")
        regions: list[dict[str, Any]] = []

        for row in csv.DictReader(
            io.StringIO(output_text),
            delimiter="\t",
        ):
            text = (row.get("text") or "").strip()

            try:
                confidence = float(row.get("conf", -1))
                x, y, box_width, box_height = (
                    int(row.get(key, 0))
                    for key in ("left", "top", "width", "height")
                )
            except (TypeError, ValueError):
                continue

            if (
                not text
                or not math.isfinite(confidence)
                or confidence < 40
                or box_width <= 0
                or box_height <= 0
                or x < 0
                or y < 0
            ):
                continue

            regions.append(
                {
                    "text": text,
                    "confidence": round(confidence / 100, 3),
                    "time": round(frame_time, 3),
                    "box": {
                        "x": round(x / width, 3),
                        "y": round((region_top + y) / height, 3),
                        "width": round(box_width / width, 3),
                        "height": round(box_height / height, 3),
                    },
                }
            )

        return regions

    def analyze(self, path: Path) -> dict[str, Any]:
        path = Path(path)

        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Reference video is missing or empty: {path}"
            )

        if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
            raise RuntimeError(
                "FFmpeg and FFprobe are required for reference analysis"
            )

        media = FFprobeService().probe(path)

        if not media["has_video"]:
            raise ValueError(
                f"Reference file has no video stream: {path}"
            )

        duration = float(media["duration"])

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Reference video has an invalid duration: {path}"
            )

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and NumPy are required to analyze visual frames"
            ) from exc

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not decode reference video: {path}")

        brightness: list[float] = []
        contrast: list[float] = []
        saturation: list[float] = []
        motion: list[float] = []
        frame_differences: list[float] = []
        cuts: list[float] = []
        caption_candidates: list[float] = []
        composition_x: list[float] = []
        composition_y: list[float] = []
        hue_histograms: list[Any] = []
        text_regions: list[dict[str, Any]] = []

        previous_gray = None
        previous_histogram = None
        sample_index = 0

        estimated_samples = max(
            1,
            math.ceil(duration / self.sample_seconds),
        )

        ocr_stride = max(
            1,
            math.ceil(
                estimated_samples / max(1, self.max_ocr_samples)
            ),
        )

        ocr_available = (
            self.max_ocr_samples > 0
            and shutil.which("tesseract") is not None
        )

        try:
            timestamp = 0.0

            while timestamp < duration:
                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp * 1000,
                )

                ok, frame = capture.read()

                if ok and frame is not None and frame.size:
                    height, width = frame.shape[:2]

                    if height <= 0 or width <= 0:
                        sample_index += 1
                        timestamp += self.sample_seconds
                        continue

                    scale = min(
                        1.0,
                        self.max_dimension / max(height, width),
                    )

                    if scale < 1:
                        frame = cv2.resize(
                            frame,
                            (
                                max(1, int(width * scale)),
                                max(1, int(height * scale)),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )

                    hsv = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2HSV,
                    )

                    value = hsv[:, :, 2].astype(np.float32) / 255

                    brightness.append(float(np.mean(value)))
                    contrast.append(float(np.std(value)))
                    saturation.append(
                        float(np.mean(hsv[:, :, 1]) / 255)
                    )

                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY,
                    )

                    saturation_mask = cv2.inRange(
                        hsv[:, :, 1],
                        32,
                        255,
                    )

                    histogram = cv2.calcHist(
                        [hsv],
                        [0],
                        saturation_mask,
                        [12],
                        [0, 180],
                    )

                    cv2.normalize(
                        histogram,
                        histogram,
                    )

                    hue_histograms.append(
                        histogram[:, 0].astype(float)
                    )

                    if previous_gray is not None:
                        frame_delta = float(
                            np.mean(
                                cv2.absdiff(gray, previous_gray)
                            ) / 255
                        )

                        if math.isfinite(frame_delta):
                            frame_differences.append(frame_delta)

                        flow = cv2.calcOpticalFlowFarneback(
                            previous_gray,
                            gray,
                            None,
                            0.5,
                            3,
                            15,
                            3,
                            5,
                            1.2,
                            0,
                        )

                        flow_magnitude, _ = cv2.cartToPolar(
                            flow[..., 0],
                            flow[..., 1],
                        )

                        motion_value = float(
                            np.mean(flow_magnitude)
                            / max(gray.shape)
                        )

                        if math.isfinite(motion_value):
                            motion.append(motion_value)

                        histogram_delta = float(
                            cv2.compareHist(
                                previous_histogram,
                                histogram,
                                cv2.HISTCMP_BHATTACHARYYA,
                            )
                        )

                        if (
                            math.isfinite(histogram_delta)
                            and histogram_delta > self.scene_cut_threshold
                        ):
                            cuts.append(round(timestamp, 3))

                    previous_gray = gray
                    previous_histogram = histogram

                    edges_full = cv2.Canny(gray, 80, 180)
                    edge_y, edge_x = np.nonzero(edges_full)

                    if edge_x.size:
                        composition_x.append(
                            float(
                                np.mean(edge_x)
                                / max(1, gray.shape[1] - 1)
                            )
                        )

                        composition_y.append(
                            float(
                                np.mean(edge_y)
                                / max(1, gray.shape[0] - 1)
                            )
                        )

                    bottom = gray[
                        int(gray.shape[0] * 0.62):,
                        :,
                    ]

                    edges = cv2.Canny(bottom, 80, 180)
                    row_density = np.mean(edges > 0, axis=1)

                    caption_score = (
                        float(np.max(row_density))
                        if row_density.size
                        else 0.0
                    )

                    if math.isfinite(caption_score):
                        caption_candidates.append(caption_score)

                    if (
                        ocr_available
                        and sample_index % ocr_stride == 0
                        and len(text_regions) < self.max_ocr_samples * 24
                    ):
                        text_regions.extend(
                            self._read_text(
                                frame,
                                cv2,
                                timestamp,
                            )
                        )

                sample_index += 1
                timestamp += self.sample_seconds

        finally:
            capture.release()

        if not brightness:
            raise ValueError(
                f"No decodable video frames were found in {path}"
            )

        cut_rate = len(cuts) / duration

        observations = [
            "Fixed-interval frame samples; scene cuts use hue-histogram distance and remain candidates, not confirmed edits.",
            "Motion is estimated with dense optical-flow magnitude; movement is not classified as camera or subject motion.",
        ]

        if ocr_available:
            observations.append(
                "Local Tesseract OCR sampled caption-region frames; "
                "its text and boxes are heuristic and may be inaccurate."
            )
        else:
            observations.append(
                "Tesseract was unavailable; caption-region estimates "
                "use image edges, with no OCR or font identification."
            )

        hue_totals = (
            np.sum(hue_histograms, axis=0)
            if hue_histograms
            else np.zeros(12)
        )

        dominant_hues = [
            round((int(index) + 0.5) * 15, 1)
            for index in np.argsort(hue_totals)[-3:][::-1]
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
            "frame_difference": (
                float(np.mean(frame_differences))
                if frame_differences
                else 0.0
            ),
            "caption_region_score": (
                float(np.mean(caption_candidates))
                if caption_candidates
                else 0.0
            ),
            "text_regions": text_regions,
            "ocr_sample_count": (
                min(
                    self.max_ocr_samples,
                    math.ceil(sample_index / ocr_stride),
                )
                if ocr_available
                else 0
            ),
            "composition_centroid": {
                "x": (
                    float(np.mean(composition_x))
                    if composition_x
                    else None
                ),
                "y": (
                    float(np.mean(composition_y))
                    if composition_y
                    else None
                ),
            },
            "dominant_hues_degrees": dominant_hues,
            "transition_style": "unknown",
            "caption_style": "unknown",
            "observations": observations,
        }
