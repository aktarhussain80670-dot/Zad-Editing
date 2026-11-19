"""FFprobe metadata extraction and full-decode output validation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from services.ffmpeg_service import MediaToolError


class FFprobeService:
    def __init__(self, executable: str = "ffprobe", timeout: int = 120) -> None:
        self.executable = shutil.which(executable)
        self.timeout = timeout

    @staticmethod
    def _parse_rate(value: str | None) -> float:
        if not value or "/" not in value:
            return 0.0
        try:
            numerator, denominator = map(float, value.split("/", 1))
            result = numerator / denominator if denominator else 0.0
            return result if math.isfinite(result) and result > 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def _run_ffprobe(self, arguments: list[str], error_message: str) -> str:
        if not self.executable:
            raise MediaToolError(
                "FFprobe not found. Install FFmpeg and ensure it is on PATH."
            )
        try:
            result = subprocess.run(
                [self.executable, "-v", "error", *arguments],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError(error_message) from exc

        if result.returncode:
            raise MediaToolError(
                f"{error_message}: {result.stderr[-2000:].strip()}"
            )
        return result.stdout

    def probe(self, path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Media file is missing or empty: {path}")

        output = self._run_ffprobe(
            ["-show_streams", "-show_format", "-of", "json", str(path)],
            f"FFprobe could not read {path.name}",
        )

        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise MediaToolError("FFprobe returned malformed JSON") from exc

        streams = payload.get("streams", [])
        video = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        audio = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"),
            None,
        )

        format_duration = payload.get("format", {}).get("duration")
        stream_durations = []
        for stream in streams:
            value = stream.get("duration")
            if value in (None, "", "N/A"):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                stream_durations.append(parsed)

        try:
            duration = (
                float(format_duration)
                if format_duration not in (None, "", "N/A")
                else max(stream_durations, default=0.0)
            )
        except (TypeError, ValueError) as exc:
            raise MediaToolError("FFprobe returned an invalid media duration") from exc

        if not math.isfinite(duration) or duration <= 0:
            raise MediaToolError("Media duration must be a finite positive number")

        avg_rate = (video or {}).get("avg_frame_rate", "0/1")
        nominal_rate = (video or {}).get("r_frame_rate", "0/1")

        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "duration": duration,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "width": (video or {}).get("width"),
            "height": (video or {}).get("height"),
            "fps": self._parse_rate(avg_rate),
            "avg_frame_rate": avg_rate,
            "r_frame_rate": nominal_rate,
            "nominal_fps": self._parse_rate(nominal_rate),
            "time_base": (video or {}).get("time_base"),
            "nb_frames": (video or {}).get("nb_frames"),
            "video_codec": (video or {}).get("codec_name"),
            "audio_codec": (audio or {}).get("codec_name"),
            "pixel_format": (video or {}).get("pix_fmt"),
            "streams": streams,
        }

    def validate_constant_frame_rate(
        self,
        path: Path,
        expected_fps: float,
    ) -> dict[str, Any]:
        if (
            isinstance(expected_fps, bool)
            or not isinstance(expected_fps, (int, float))
            or not math.isfinite(expected_fps)
            or expected_fps <= 0
        ):
            raise ValueError("Expected FPS must be a finite positive number")

        output = self._run_ffprobe(
            [
                "-select_streams", "v:0",
                "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time",
                "-of", "csv=p=0",
                str(path),
            ],
            "FFprobe could not inspect video frame timestamps",
        )

        timestamps: list[float] = []
        for line in output.splitlines():
            value = line.strip().split(",", 1)[0]
            if not value or value == "N/A":
                raise MediaToolError(
                    "Video contains a frame with a missing timestamp"
                )
            try:
                timestamp = float(value)
            except ValueError as exc:
                raise MediaToolError(
                    "FFprobe returned an invalid frame timestamp"
                ) from exc
            if not math.isfinite(timestamp):
                raise MediaToolError(
                    "Video contains a non-finite frame timestamp"
                )
            timestamps.append(timestamp)

        if len(timestamps) < 2:
            raise MediaToolError(
                "At least two video frames are required for CFR validation"
            )

        # Preserve original frame order; sorting would hide invalid timestamps.
        expected_interval = 1.0 / expected_fps
        tolerance = max(0.0005, expected_interval * 0.01)

        for index in range(1, len(timestamps)):
            interval = timestamps[index] - timestamps[index - 1]
            if interval <= 0 or abs(interval - expected_interval) > tolerance:
                raise MediaToolError(
                    "Video is not constant-frame-rate: "
                    f"frame interval {interval:.7f}s; "
                    f"expected {expected_interval:.7f}s"
                )

        return {
            "constant_frame_rate": True,
            "verified_fps": expected_fps,
            "verified_frame_count": len(timestamps),
        }

    def validate(
        self,
        path: Path,
        require_video: bool = True,
        require_audio: bool = False,
        expected_duration: float | None = None,
        tolerance: float = 0.25,
        expected_fps: float | None = None,
    ) -> dict[str, Any]:
        if expected_duration is not None and (
            not math.isfinite(expected_duration) or expected_duration <= 0
        ):
            raise ValueError("Expected duration must be finite and positive")

        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Duration tolerance must be finite and non-negative")

        info = self.probe(path)

        if require_video and not info["has_video"]:
            raise MediaToolError("Output is missing a video stream")
        if require_audio and not info["has_audio"]:
            raise MediaToolError("Output is missing an audio stream")

        if expected_duration is not None:
            if abs(info["duration"] - expected_duration) > tolerance:
                raise MediaToolError(
                    f"Duration {info['duration']:.3f}s differs from "
                    f"expected {expected_duration:.3f}s"
                )

        if expected_fps is not None:
            if not info["has_video"]:
                raise MediaToolError("CFR validation requires a video stream")
            cfr_result = self.validate_constant_frame_rate(path, expected_fps)
            info.update(cfr_result)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise MediaToolError(
                "Full output integrity check requires FFmpeg; "
                "ensure it is on PATH"
            )

        try:
            check = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError(
                "Output decode integrity check timed out"
            ) from exc

        if check.returncode:
            raise MediaToolError(
                f"Output integrity check failed: {check.stderr[-1000:].strip()}"
            )

        return info
