"""Safe FFmpeg wrappers with timeouts and useful diagnostics."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


class MediaToolError(RuntimeError):
    """A media tool is missing or returned a failed result."""


class FFmpegService:
    def __init__(self, executable: str = "ffmpeg", timeout: int = 600) -> None:
        self.executable = shutil.which(executable)
        self.timeout = timeout

    def ensure_available(self) -> str:
        if not self.executable:
            raise MediaToolError("FFmpeg not found. Install FFmpeg and ensure ffmpeg is on PATH.")
        return self.executable

    @staticmethod
    def _destination(destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing media file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _verify_output(destination: Path) -> Path:
        if not destination.is_file() or destination.stat().st_size == 0:
            raise MediaToolError(f"FFmpeg reported success but output is missing or empty: {destination}")
        return destination

    def run(self, arguments: Sequence[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
            raise TypeError("FFmpeg arguments must be a sequence of strings")

        if any(not isinstance(argument, str) or "\\x00" in argument for argument in arguments):
            raise ValueError("FFmpeg arguments must contain valid strings without NUL characters")

        effective_timeout = self.timeout if timeout is None else timeout

        if (
            isinstance(effective_timeout, bool)
            or not isinstance(effective_timeout, (int, float))
            or not math.isfinite(effective_timeout)
            or effective_timeout <= 0
        ):
            raise ValueError("FFmpeg timeout must be a positive number")

        command = [self.ensure_available(), "-hide_banner", "-nostdin", *arguments]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError(
                f"FFmpeg timed out after {exc.timeout} seconds"
            ) from exc
        except OSError as exc:
            raise MediaToolError(f"Could not execute FFmpeg: {exc}") from exc

        if result.returncode:
            raise MediaToolError(
                f"FFmpeg exited with code {result.returncode}: "
                f"{result.stderr[-4000:].strip()}"
            )

        return result

    def trim(self, source: Path, destination: Path, start: float, duration: float) -> Path:
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(start)
            or start < 0
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Trim start and duration must be finite valid numbers")
        self._destination(destination)
        self.run(
            [
                "-y",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(destination),
            ]
        )
        return self._verify_output(destination)

    def scale(self, source: Path, destination: Path, width: int, height: int, fps: int) -> Path:
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height, fps))
            or width < 2
            or height < 2
            or fps <= 0
        ):
            raise ValueError("Output dimensions and frame rate must be positive integers")
        self._destination(destination)
        self.run(
            [
                "-y",
                "-i",
                str(source),
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ]
        )
        return self._verify_output(destination)

    def concatenate(self, sources: list[Path], destination: Path) -> Path:
        if not sources or any(not path.is_file() for path in sources):
            raise ValueError("Concatenation requires existing input files")
        if any("'" in str(path.resolve()) or "\n" in str(path) or "\r" in str(path) for path in sources):
            raise ValueError("Input paths contain unsupported characters")
        self._destination(destination)
        manifest = destination.with_suffix(destination.suffix + ".concat.txt")
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite existing concat manifest: {manifest}")
        manifest.write_text("".join(f"file '{path.resolve()}'\n" for path in sources), encoding="utf-8")
        try:
            self.run(["-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(destination)])
        finally:
            manifest.unlink(missing_ok=True)
        return self._verify_output(destination)

    def extract_audio(self, source: Path, destination: Path) -> Path:
        self._destination(destination)
        self.run(["-y", "-i", str(source), "-vn", "-ac", "2", "-ar", "48000", str(destination)])
        return self._verify_output(destination)

    def prepare_scene(
        self, source: Path, destination: Path, duration: float, width: int, height: int, fps: int
    ) -> Path:
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
            or not source.is_file()
        ):
            raise ValueError("Scene clip must exist and have a finite positive duration")
        self._destination(destination)
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps={fps}"
        )
        self.run(
            [
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                str(source),
                "-t",
                str(duration),
                "-vf",
                video_filter,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ]
        )
        return self._verify_output(destination)
