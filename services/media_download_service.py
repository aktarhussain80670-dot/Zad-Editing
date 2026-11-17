"""Download permitted provider assets and record their source metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from providers.http import download_file
from services.ffprobe_service import FFprobeService


class MediaDownloadService:
    def __init__(self, timeout: int = 30, max_bytes: int = 250_000_000) -> None:
        self.timeout, self.max_bytes = timeout, max_bytes

    def download(self, result: dict[str, Any], directory: Path, metadata_directory: Path) -> Path:
        url = str(result.get("url", ""))
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".mp3", ".wav", ".m4a"}:
            suffix = ".mp4"
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(result.get("id", "asset")))[:80] or "asset"
        destination = directory / f"{name}{suffix}"
        if destination.is_symlink():
            raise ValueError("Refusing to reuse a symlink as a downloaded media asset")
        if destination.is_file():
            existing = FFprobeService(timeout=self.timeout).probe(destination)
            if not existing["has_video"] or existing["duration"] <= 0:
                raise ValueError("Existing downloaded clip is not valid video media")
            self._write_metadata(result, name, existing, destination, metadata_directory)
            return destination
        provider = result.get("provider")
        allowed_hosts = (
            {"videos.pexels.com"} if provider == "pexels" else {"cdn.pixabay.com"} if provider == "pixabay" else set()
        )
        download_file(url, destination, self.timeout, self.max_bytes, allowed_hosts)
        try:
            media_info = FFprobeService(timeout=self.timeout).probe(destination)
            if not media_info["has_video"] or media_info["duration"] <= 0:
                raise ValueError("Downloaded clip has no valid video stream")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        result = {
            **result,
            "downloaded_bytes": destination.stat().st_size,
            "validated_duration": media_info["duration"],
        }
        self._write_metadata(result, name, media_info, destination, metadata_directory)
        return destination

    @staticmethod
    def _write_metadata(
        result: dict[str, Any], name: str, media_info: dict[str, Any], destination: Path, metadata_directory: Path
    ) -> None:
        metadata_directory.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_directory / f"{name}.json"
        if metadata_path.exists():
            return
        metadata = {
            **result,
            "downloaded_bytes": destination.stat().st_size,
            "validated_duration": media_info["duration"],
        }
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=2) + "\n")
