"""Official Pexels video-search API adapter."""

from __future__ import annotations

import urllib.parse
from typing import Any

from providers.http import ProviderError, get_json


class PexelsProvider:
    API = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout

    def search_videos(
        self, query: str, orientation: str = "portrait", min_width: int = 0, min_height: int = 0, per_page: int = 15
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ProviderError("PEXELS_API_KEY is not configured")
        if not query.strip() or orientation not in {"portrait", "landscape", "square"}:
            raise ValueError("A search query and supported orientation are required")
        params = urllib.parse.urlencode(
            {"query": query[:100], "orientation": orientation, "size": "large", "per_page": max(1, min(per_page, 80))}
        )
        payload = get_json(f"{self.API}?{params}", {"Authorization": self.api_key}, self.timeout)
        results = []
        for video in payload.get("videos", []):
            files = [
                f
                for f in video.get("video_files", [])
                if f.get("link", "").startswith("https://")
                and int(f.get("width", 0)) >= min_width
                and int(f.get("height", 0)) >= min_height
            ]
            if files:
                chosen = max(files, key=lambda f: int(f.get("width", 0)) * int(f.get("height", 0)))
                results.append(
                    {
                        "provider": "pexels",
                        "id": video.get("id"),
                        "url": chosen["link"],
                        "width": chosen.get("width"),
                        "height": chosen.get("height"),
                        "duration": video.get("duration"),
                        "photographer": video.get("user", {}).get("name"),
                        "source": video.get("url"),
                        "license": "Pexels License; review provider terms for intended use",
                    }
                )
        return results
