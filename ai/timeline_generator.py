"""Generate and persist a validated JSON editing guide."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.timeline_validator import validate_timeline


def generate_guide(transcript: dict[str, Any], style: dict[str, Any], scenes: list[dict[str, Any]], output: Path = Path("storage/editing_guides/editing_guide.json")) -> dict[str, Any]:
    guide = {"schema_version": 1, "duration": float(transcript["duration"]), "transcript": transcript, "style": style, "scenes": scenes}
    validate_timeline(guide)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(guide, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return guide
