"""FFprobe metadata extraction and full-decode output validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from services.ffmpeg_service import MediaToolError


class FFprobeService:
    def __init__(self, executable: str = "ffprobe", timeout: int = 120) -> None:
        self.executable = shutil.which(executable)
        self.timeout = timeout

    def probe(self, path: Path) -> dict[str, Any]:
        if not self.executable:
            raise MediaToolError("FFprobe not found. Install FFmpeg (which includes ffprobe) and ensure it is on PATH.")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Media file is missing or empty: {path}")
        try:
            result = subprocess.run(
                [self.executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError("FFprobe timed out while reading media metadata") from exc
        if result.returncode:
            raise MediaToolError(f"FFprobe could not read {path.name}: {result.stderr[-2000:].strip()}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MediaToolError("FFprobe returned malformed JSON") from exc
        streams = payload.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(
            payload.get("format", {}).get("duration", 0)
            or max((float(s.get("duration", 0) or 0) for s in streams), default=0)
        )
        rate = (video or {}).get("avg_frame_rate", "0/1")
        try:
            n, d = map(float, rate.split("/"))
            fps = n / d if d else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "duration": duration,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "width": (video or {}).get("width"),
            "height": (video or {}).get("height"),
            "fps": fps,
            "video_codec": (video or {}).get("codec_name"),
            "audio_codec": (audio or {}).get("codec_name"),
            "pixel_format": (video or {}).get("pix_fmt"),
            "streams": streams,
        }

    def validate(
        self,
        path: Path,
        require_video: bool = True,
        require_audio: bool = False,
        expected_duration: float | None = None,
        tolerance: float = 0.25,
    ) -> dict[str, Any]:
        info = self.probe(path)
        if require_video and not info["has_video"]:
            raise MediaToolError("Output is missing a video stream")
        if require_audio and not info["has_audio"]:
            raise MediaToolError("Output is missing an audio stream")
        if expected_duration is not None and abs(info["duration"] - expected_duration) > tolerance:
            raise MediaToolError(f"Duration {info['duration']:.3f}s differs from expected {expected_duration:.3f}s")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise MediaToolError("Full output integrity check requires FFmpeg; install it and ensure it is on PATH")
        try:
            check = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError("Output decode integrity check timed out") from exc
        if check.returncode:
            raise MediaToolError(f"Output integrity check failed: {check.stderr[-1000:].strip()}")
        return info
