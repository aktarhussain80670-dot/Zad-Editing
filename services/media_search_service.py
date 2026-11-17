"""Apply feature and scene-level gates before optional media-provider searches."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings
from providers.pexels_provider import PexelsProvider

logger = logging.getLogger(__name__)


class MediaSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search_scene(self, scene: dict[str, Any], orientation: str = "portrait") -> list[dict[str, Any]]:
        if not scene.get("visual_required", True):
            return []
        if not self.settings.enable_external_clips:
            return []
        if not self.settings.pexels_api_key:
            logger.warning("External footage skipped: PEXELS_API_KEY is not configured")
            return []
        query = str(scene.get("visual_search_query", "")).strip()
        if not query:
            return []
        return PexelsProvider(self.settings.pexels_api_key, self.settings.media_timeout_seconds).search_videos(
            query, orientation
        )
