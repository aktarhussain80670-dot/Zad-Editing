"""Pixabay integration disabled by production policy."""

from __future__ import annotations

from typing import Any

from providers.http import ProviderError


class PixabayProvider:
    """Compatibility interface that rejects all Pixabay API requests."""

    VIDEO_API = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str = "", timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search_videos(
        self,
        query: str,
        orientation: str = "vertical",
        per_page: int = 20,
    ) -> list[dict[str, Any]]:
        raise ProviderError(
            "Pixabay integration is disabled by production policy."
        )

    def search_audio(self, query: str, kind: str) -> list[dict[str, Any]]:
        raise ProviderError(
            "Pixabay music and sound-effect integration is disabled "
            "by production policy."
        )
