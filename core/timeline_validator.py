"""Structural and timing validation for generated editing guides."""

from __future__ import annotations

from typing import Any


def validate_timeline(guide: dict[str, Any]) -> None:
    duration = float(guide.get("duration", 0))
    if guide.get("schema_version") != 1:
        raise ValueError("Unsupported editing guide schema version")
    transcript = guide.get("transcript")
    if not isinstance(transcript, dict) or not isinstance(transcript.get("text"), str):
        raise TypeError("Guide transcript must contain the preserved original text")
    if not isinstance(transcript.get("segments"), list) or not transcript["segments"]:
        raise ValueError("Guide transcript requires at least one timed segment")
    if abs(float(transcript.get("duration", 0)) - duration) > 0.05:
        raise ValueError("Guide duration must match the original voiceover duration")
    for index, segment in enumerate(transcript["segments"], 1):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise TypeError(f"Transcript segment {index} must contain timed text")
        start, end = float(segment["start"]), float(segment["end"])
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError(f"Transcript segment {index} has invalid timestamps")
        for word in segment.get("words", []):
            word_start, word_end = float(word["start"]), float(word["end"])
            if word_start < start - 0.05 or word_end <= word_start or word_end > end + 0.05:
                raise ValueError(f"Transcript segment {index} has invalid word timestamps")
    if not isinstance(guide.get("style"), dict):
        raise TypeError("Guide style profile must be an object")
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
        if (
            not isinstance(scene.get("caption"), dict)
            or not isinstance(scene.get("motion"), dict)
            or not isinstance(scene.get("transition"), dict)
        ):
            raise TypeError(f"Scene {expected_id} is missing rendering instructions")
        if not isinstance(scene.get("visual_required"), bool) or not isinstance(scene.get("external_media"), dict):
            raise TypeError(f"Scene {expected_id} is missing visual/media requirements")
        required_fields = ("transcript", "visual_idea", "visual_search_query", "typography", "sound_effect", "audio")
        if any(field not in scene for field in required_fields):
            raise ValueError(f"Scene {expected_id} is missing required editing instructions")
        for field in ("transcript", "visual_idea", "visual_search_query"):
            if not isinstance(scene[field], str):
                raise TypeError(f"Scene {expected_id} {field} must be a string")
        for group, keys in (
            (scene["caption"], ("enabled", "text", "animation")),
            (scene["motion"], ("type",)),
            (scene["transition"], ("type",)),
            (scene["external_media"], ("required", "query")),
            (scene["sound_effect"], ("enabled",)),
            (scene["audio"], ("music_enabled", "sound_effect_enabled")),
        ):
            if any(key not in group for key in keys):
                raise ValueError(f"Scene {expected_id} has incomplete rendering instructions")
        cursor = end
    if abs(cursor - duration) > 0.05:
        raise ValueError("Timeline must end at voiceover duration")
