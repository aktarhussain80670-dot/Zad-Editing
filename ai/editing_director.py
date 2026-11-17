"""Transparent rule-based director for voiceover-led scene planning."""

from __future__ import annotations

from typing import Any

from ai.scene_planner import plan_scenes


def direct(
    transcript: dict[str, Any], style: dict[str, Any], features: dict[str, bool] | None = None
) -> list[dict[str, Any]]:
    scenes = plan_scenes(transcript, style)
    enabled = features or {}
    for scene in scenes:
        scene["caption"]["enabled"] = enabled.get("subtitles", True)
        scene["motion"]["enabled"] = enabled.get("motion_graphics", True)
        scene["transition"]["enabled"] = enabled.get("transitions", True)
        scene["sound_effect"]["enabled"] = enabled.get("sound_effects", False)
        scene["audio"]["music_enabled"] = enabled.get("background_music", False)
        scene["audio"]["sound_effect_enabled"] = enabled.get("sound_effects", False)
        scene["external_media"]["required"] = enabled.get("external_clips", False) and scene["visual_required"]
        scene["decision_rationale"] = [
            "Scene timing is based on original voiceover transcript timestamps.",
            "External footage is optional and is gated by global feature configuration.",
        ]
    return scenes
