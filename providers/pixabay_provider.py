"""Official Pixabay API adapters; public API currently covers photos and videos."""

from __future__ import annotations

import urllib.parse
from typing import Any

from providers.http import ProviderError, get_json


class PixabayProvider:
    VIDEO_API = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout

    def search_videos(self, query: str, orientation: str = "vertical", per_page: int = 20) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ProviderError("PIXABAY_API_KEY is not configured")
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError("Pixabay video orientation must be vertical or horizontal")
        params = urllib.parse.urlencode(
            {"key": self.api_key, "q": query[:100], "safesearch": "true", "per_page": max(3, min(per_page, 200))}
        )
        payload = get_json(f"{self.VIDEO_API}?{params}", {}, self.timeout)
        results = []
        for hit in payload.get("hits", []):
            renditions = hit.get("videos", {})
            candidates = [
                item
                for item in renditions.values()
                if item.get("url", "").startswith("https://")
                and (item.get("height", 0) >= item.get("width", 0)) == (orientation == "vertical")
            ]
            if candidates:
                chosen = max(candidates, key=lambda item: item.get("width", 0) * item.get("height", 0))
                results.append(
                    {
                        "provider": "pixabay",
                        "id": hit.get("id"),
                        "url": chosen["url"],
                        "width": chosen.get("width"),
                        "height": chosen.get("height"),
                        "duration": hit.get("duration"),
                        "source": hit.get("pageURL"),
                        "creator": hit.get("user"),
                        "license": "Pixabay Content License; review provider terms for intended use",
                    }
                )
        return results

    def search_audio(self, query: str, kind: str) -> list[dict[str, Any]]:
        if kind not in {"music", "sound_effect"}:
            raise ValueError("Audio kind must be music or sound_effect")
        if not self.api_key:
            raise ProviderError("PIXABAY_API_KEY is not configured")
        raise ProviderError(
            "Pixabay's documented public API does not provide music or sound-effect search endpoints; no undocumented endpoint or scraping is used"
        )
