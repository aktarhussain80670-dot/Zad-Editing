"""Transparent rule-based director for voiceover-led scene planning."""

from __future__ import annotations

from typing import Any

from ai.scene_planner import plan_scenes


def direct(transcript: dict[str, Any], style: dict[str, Any], features: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    scenes = plan_scenes(transcript, style)
    enabled = features or {}
    for scene in scenes:
        scene["caption"]["enabled"] = enabled.get("subtitles", True)
        scene["motion"]["enabled"] = enabled.get("motion_graphics", True)
        scene["transition"]["enabled"] = enabled.get("transitions", True)
        scene["sound_effect"]["enabled"] = enabled.get("sound_effects", False)
    return scenes
