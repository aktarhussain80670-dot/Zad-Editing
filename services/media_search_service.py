"""External media search is disabled in the production editing pipeline."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings

logger = logging.getLogger(__name__)


class MediaSearchService:
    """Compatibility interface that never contacts external media providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search_scene(
        self,
        scene: dict[str, Any],
        orientation: str = "portrait",
    ) -> list[dict[str, Any]]:
        """Return no external media results by production policy."""
        if scene.get("visual_required") and not self.settings.pexels_api_key:
            logger.warning("External media requested without a provider key; using generated graphics")
        return []
