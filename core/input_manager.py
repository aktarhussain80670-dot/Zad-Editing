"""Validate voiceover and reference inputs before expensive processing."""

from __future__ import annotations

import math
import os
from pathlib import Path

from config.settings import Settings
from services.ffprobe_service import FFprobeService

VOICE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _validate_file(path: Path, max_bytes: int, description: str) -> Path:
    """Validate file existence, readability, size, and canonical path."""
    try:
        resolved = path.expanduser().resolve(strict=True)

        if not resolved.is_file():
            raise ValueError(f"{description} is not a regular file: {path}")

        if not os.access(resolved, os.R_OK):
            raise ValueError(f"{description} is not readable: {path}")

        size = resolved.stat().st_size

        if size <= 0:
            raise ValueError(f"{description} is empty: {path}")

        if size > max_bytes:
            raise ValueError(f"{description} exceeds MAX_INPUT_BYTES: {path}")

        return resolved

    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{description} cannot be accessed: {path}") from exc


def _validate_duration(info: dict, description: str) -> None:
    """Reject media with missing, invalid, or non-finite duration."""
    duration = info.get("duration")

    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"{description} has an invalid or unavailable duration")


def validate_inputs(
    voiceover: Path,
    references: list[Path],
    settings: Settings,
) -> None:
    """Validate one voiceover and between one and four reference videos."""
    if not isinstance(voiceover, Path):
        raise ValueError("Voiceover must be provided as a pathlib Path")

    if not isinstance(references, list):
        raise ValueError("Reference videos must be provided as a list of paths")

    if not settings.min_references <= len(references) <= settings.max_references:
        raise ValueError(
            f"Expected {settings.min_references}-{settings.max_references} reference videos; "
            f"received {len(references)}"
        )

    if voiceover.suffix.lower() not in VOICE_EXTENSIONS:
        raise ValueError(f"Unsupported voiceover format: {voiceover.suffix}")

    voiceover_path = _validate_file(
        voiceover,
        settings.max_input_bytes,
        "Voiceover",
    )

    reference_paths: list[Path] = []

    for reference in references:
        if not isinstance(reference, Path):
            raise ValueError("Every reference video must be a pathlib Path")

        if reference.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported reference video format: {reference.suffix}")

        reference_paths.append(
            _validate_file(
                reference,
                settings.max_input_bytes,
                "Reference video",
            )
        )

    all_paths = [voiceover_path, *reference_paths]

    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Voiceover and reference inputs must be distinct files")

    ffprobe = FFprobeService()

    voiceover_info = ffprobe.probe(voiceover_path)

    if not voiceover_info["has_audio"]:
        raise ValueError("The selected voiceover does not contain an audio stream")

    _validate_duration(voiceover_info, "Voiceover")

    for reference_path in reference_paths:
        reference_info = ffprobe.probe(reference_path)

        if not reference_info["has_video"]:
            raise ValueError(
                f"Reference file does not contain a video stream: {reference_path.name}"
            )

        _validate_duration(reference_info, f"Reference video {reference_path.name}")
