"""Aggregate observations from reference analyses without inventing certainty."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def extract_style(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not analyses:
        raise ValueError("At least one reference analysis is required")
    keys = ("brightness", "contrast", "saturation", "cut_rate", "motion", "caption_region_score")
    metrics: dict[str, Any] = {}
    for key in keys:
        values = [float(item[key]) for item in analyses if isinstance(item.get(key), (int, float))]
        if values:
            center = median(values)
            deviation = median([abs(value - center) for value in values])
            metrics[key] = {
                "median": round(center, 4),
                "mad": round(deviation, 4),
                "samples": len(values),
                "confidence": round(len(values) / len(analyses), 2),
            }
        else:
            metrics[key] = None
    transitions = Counter(str(item["transition_style"]) for item in analyses if item.get("transition_style"))
    caption_modes = Counter(str(item["caption_style"]) for item in analyses if item.get("caption_style"))
    transition = transitions.most_common(1)[0] if transitions else ("unknown", 0)
    caption = caption_modes.most_common(1)[0] if caption_modes else ("unknown", 0)
    return {
        "reference_count": len(analyses),
        "metrics": metrics,
        "transition_preference": transition[0] if transition[1] > len(analyses) / 2 else "unknown",
        "caption_style": caption[0] if caption[1] > len(analyses) / 2 else "unknown",
        "common_characteristics": {
            "transition": transition[0] if transition[1] > len(analyses) / 2 else "unknown",
            "caption": caption[0] if caption[1] > len(analyses) / 2 else "unknown",
        },
        "similarity_scores": [
            round(
                1
                / (
                    1
                    + sum(
                        abs(float(a.get(k, 0)) - float(b.get(k, 0)))
                        for k in ("brightness", "contrast", "saturation", "cut_rate", "motion")
                        if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
                    )
                ),
                4,
            )
            for a in analyses
            for b in analyses
        ],
        "observations": [item.get("observations", []) for item in analyses],
        "confidence": round(min(1.0, len(analyses) / 4), 2),
    }
