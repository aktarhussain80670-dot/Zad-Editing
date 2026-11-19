"""Generate and atomically persist validated version 2 editing guides."""

from __future__ import annotations

import json
import os
import tempfile
import math
from pathlib import Path
from typing import Any

from core.timeline_validator import validate_timeline


def generate_guide(
    transcript: dict[str, Any],
    style: dict[str, Any],
    scenes: list[dict[str, Any]],
    output: Path = Path("storage/editing_guides/editing_guide.json"),
    audio_analysis: dict[str, Any] | None = None,
    audio_duration: float | None = None,
) -> dict[str, Any]:
    if not isinstance(transcript, dict):
        raise TypeError("Transcript must be an object")

    if not isinstance(style, dict):
        raise TypeError("Style profile must be an object")

    if not isinstance(scenes, list):
        raise TypeError("Scenes must be a list")

    if audio_analysis is not None and not isinstance(audio_analysis, dict):
        raise TypeError("Audio analysis must be an object")

    if audio_duration is not None:
        if (
            isinstance(audio_duration, bool)
            or not isinstance(audio_duration, (int, float))
            or not math.isfinite(audio_duration)
            or audio_duration <= 0
        ):
            raise ValueError("Original audio duration must be finite and positive")

        transcript = dict(transcript)
        transcript["duration"] = float(audio_duration)

    guide = {
        "schema_version": 2,
        "duration": transcript.get("duration"),
        "transcript": transcript,
        "audio_analysis": (
            {} if audio_analysis is None else audio_analysis
        ),
        "style": style,
        "scenes": scenes,
    }

    validate_timeline(guide)

    serialized = (
        json.dumps(
            guide,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return guide
