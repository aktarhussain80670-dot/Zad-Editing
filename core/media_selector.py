"""External media selection is disabled by production policy."""

from typing import Any

from config.settings import Settings


class MediaSelector:
    """Compatibility interface that never selects external footage."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def scenes_requiring_clips(
        self, guide: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return no scenes because external footage is prohibited."""
        return []
