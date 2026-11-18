"""External media downloading is disabled by production policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MediaDownloadService:
    """Compatibility interface that rejects all external media downloads."""

    def __init__(
        self,
        timeout: int = 30,
        max_bytes: int = 250_000_000,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes

    def download(
        self,
        result: dict[str, Any],
        directory: Path,
        metadata_directory: Path,
    ) -> Path:
        """Reject external media downloads without touching local files."""
        raise RuntimeError(
            "External media downloading is permanently disabled "
            "by production policy. Pexels and Pixabay are not permitted."
        )
