"""External media search is disabled in the production editing pipeline."""

from __future__ import annotations

from typing import Any

from config.settings import Settings


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
        return []
