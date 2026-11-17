"""Structural and timing validation for generated editing guides."""

from __future__ import annotations

from typing import Any


def validate_timeline(guide: dict[str, Any]) -> None:
    duration = float(guide.get("duration", 0))
    scenes = guide.get("scenes")
    if duration <= 0 or not isinstance(scenes, list) or not scenes:
        raise ValueError("Guide requires positive duration and at least one scene")
    cursor = 0.0
    for expected_id, scene in enumerate(scenes, 1):
        if scene.get("scene_id") != expected_id:
            raise ValueError("Scene IDs must be sequential")
        start, end = float(scene["start"]), float(scene["end"])
        if abs(start - cursor) > 0.05 or end <= start:
            raise ValueError(f"Scene {expected_id} has a gap, overlap, or invalid duration")
        if end > duration + 0.05:
            raise ValueError(f"Scene {expected_id} exceeds guide duration")
        cursor = end
    if abs(cursor - duration) > 0.05:
        raise ValueError("Timeline must end at voiceover duration")
