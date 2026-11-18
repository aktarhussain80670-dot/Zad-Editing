"""Pexels integration disabled by production policy."""

from __future__ import annotations

from typing import Any

from providers.http import ProviderError


class PexelsProvider:
    """Compatibility interface that rejects all Pexels API requests."""

    API = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str = "", timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search_videos(
        self,
        query: str,
        orientation: str = "portrait",
        min_width: int = 0,
        min_height: int = 0,
        per_page: int = 15,
    ) -> list[dict[str, Any]]:
        raise ProviderError(
            "Pexels integration is disabled by production policy."
        )
