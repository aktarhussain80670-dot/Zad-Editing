"""Aggregate observations from reference analyses without inventing certainty."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def extract_style(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not analyses:
        raise ValueError("At least one reference analysis is required")
    keys = (
        "brightness",
        "contrast",
        "saturation",
        "cut_rate",
        "motion",
        "frame_difference",
        "caption_region_score",
    )
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
        "composition_centroid": {
            axis: round(
                median(
                    float(item["composition_centroid"][axis])
                    for item in analyses
                    if isinstance((item.get("composition_centroid") or {}).get(axis), (int, float))
                ),
                4,
            )
            if any(isinstance((item.get("composition_centroid") or {}).get(axis), (int, float)) for item in analyses)
            else None
            for axis in ("x", "y")
        },
        "dominant_hues_degrees": [
            int(hue)
            for hue, count in Counter(
                hue for item in analyses for hue in item.get("dominant_hues_degrees", [])
            ).most_common(3)
            if count >= max(1, len(analyses) // 2)
        ],
        "ocr_text_examples": list(
            dict.fromkeys(
                str(region["text"])
                for item in analyses
                for region in item.get("text_regions", [])
                if region.get("text")
            )
        )[:30],
        "similarity_scores": _similarity_scores(analyses),
        "observations": [item.get("observations", []) for item in analyses],
        "confidence": round(min(1.0, len(analyses) / 4), 2),
    }


def _similarity_scores(analyses: list[dict[str, Any]]) -> list[float]:
    features = ("brightness", "contrast", "saturation", "cut_rate", "motion", "frame_difference")
    ranges = {}
    for key in features:
        values = [float(item[key]) for item in analyses if isinstance(item.get(key), (int, float))]
        ranges[key] = max(max(values) - min(values), 0.1) if values else 1.0
    scores = []
    for left in analyses:
        for right in analyses:
            distances = [
                abs(float(left[key]) - float(right[key])) / ranges[key]
                for key in features
                if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float))
            ]
            scores.append(round(1 / (1 + (sum(distances) / len(distances) if distances else 0)), 4))
    return scores
