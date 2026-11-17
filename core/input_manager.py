"""Validate voiceover and reference inputs before any expensive processing."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings

VOICE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def validate_inputs(voiceover: Path, references: list[Path], settings: Settings) -> None:
    if not voiceover.is_file() or voiceover.suffix.lower() not in VOICE_EXTENSIONS:
        raise ValueError("Provide exactly one readable supported voiceover file")
    if voiceover.stat().st_size == 0 or voiceover.stat().st_size > settings.max_input_bytes:
        raise ValueError("Voiceover is empty or exceeds MAX_INPUT_BYTES")
    if not settings.min_references <= len(references) <= settings.max_references:
        raise ValueError(f"Expected {settings.min_references}-{settings.max_references} reference videos")
    normalized = [path.resolve() for path in references]
    if len(set(normalized)) != len(normalized) or voiceover.resolve() in normalized:
        raise ValueError("Inputs must be distinct files")
    for path in references:
        if (
            not path.is_file()
            or path.suffix.lower() not in VIDEO_EXTENSIONS
            or path.stat().st_size == 0
            or path.stat().st_size > settings.max_input_bytes
        ):
            raise ValueError(f"Missing, empty, or unsupported reference video: {path}")
