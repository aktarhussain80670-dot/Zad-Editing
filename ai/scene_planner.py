"""Create chronologically continuous scenes from voiceover sentence timing."""

from __future__ import annotations

import re
from typing import Any


def plan_scenes(transcript: dict[str, Any], style: dict[str, Any]) -> list[dict[str, Any]]:
    duration = float(transcript.get("duration", 0))
    segments = transcript.get("segments", [])
    if duration <= 0 or not segments:
        raise ValueError("Transcript must have positive duration and timed segments")
    scenes: list[dict[str, Any]] = []
    boundaries = [0.0]
    for left, right in zip(segments, segments[1:]):
        left_end, right_start = float(left["end"]), float(right["start"])
        boundaries.append(max(boundaries[-1], min(duration, (left_end + right_start) / 2)))
    boundaries.append(duration)
    for index, segment in enumerate(segments, 1):
        start, end = float(segment["start"]), float(segment["end"])
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError(f"Invalid transcript segment timing at index {index - 1}")
        text = str(segment.get("text", "")).strip()
        terms = re.findall(r"[\w'-]+", text)
        scenes.append({
            "scene_id": index,
            "start": round(boundaries[index - 1], 3),
            "end": round(boundaries[index], 3),
            "transcript": text,
            "visual_idea": text or "Narration-led illustrative visual",
            "visual_required": True,
            "visual_search_query": " ".join(terms[:8]),
            "caption": {"enabled": True, "text": text, "animation": "kinetic" if style.get("caption_style") == "kinetic" else "fade"},
            "motion": {"type": "slow_zoom_in"},
            "transition": {"type": style.get("transition_preference", "cut")},
            "sound_effect": {"enabled": False},
        })
    return scenes
