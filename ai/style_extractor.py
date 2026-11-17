"""Aggregate observations from reference analyses without inventing certainty."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def extract_style(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not analyses:
        raise ValueError("At least one reference analysis is required")
    keys = ("brightness", "contrast", "saturation", "cut_rate", "motion")
    metrics: dict[str, Any] = {}
    for key in keys:
        values = [float(item[key]) for item in analyses if isinstance(item.get(key), (int, float))]
        metrics[key] = {"median": round(median(values), 4), "samples": len(values)} if values else None
    transitions = Counter(str(item["transition_style"]) for item in analyses if item.get("transition_style"))
    caption_modes = Counter(str(item["caption_style"]) for item in analyses if item.get("caption_style"))
    return {
        "reference_count": len(analyses),
        "metrics": metrics,
        "transition_preference": transitions.most_common(1)[0][0] if transitions else "unknown",
        "caption_style": caption_modes.most_common(1)[0][0] if caption_modes else "unknown",
        "observations": [item.get("observations", []) for item in analyses],
        "confidence": round(min(1.0, len(analyses) / 4), 2),
    }
