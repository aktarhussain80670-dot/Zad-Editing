"""Deterministically select scenes that qualify for optional external footage."""

from typing import Any

from config.settings import Settings


class MediaSelector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def scenes_requiring_clips(self, guide: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.settings.enable_external_clips or not self.settings.pexels_api_key:
            return []
        return [
            scene
            for scene in guide.get("scenes", [])
            if scene.get("visual_required") and scene.get("external_media", {}).get("required")
        ]
