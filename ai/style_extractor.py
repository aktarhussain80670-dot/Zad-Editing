"""Extract conservative, measurable visual style profiles from references."""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any


def _valid_numbers(values: list[Any]) -> list[float]:
    result: list[float] = []

    for value in values:
        if isinstance(value, bool):
            continue

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if math.isfinite(number):
            result.append(number)

    return result


def _valid_analyses(
    analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(analyses, list):
        raise TypeError("Reference analyses must be a list")

    if not 1 <= len(analyses) <= 4:
        raise ValueError(
            "Style extraction requires between 1 and 4 reference analyses"
        )

    if any(not isinstance(item, dict) for item in analyses):
        raise ValueError(
            "Every reference analysis must be a dictionary"
        )

    return analyses


def _metric(
    analyses: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    values = _valid_numbers(
        [item.get(key) for item in analyses]
    )

    if not values:
        return None

    center = median(values)
    deviation = median(
        [abs(value - center) for value in values]
    )

    return {
        "median": round(center, 4),
        "mad": round(deviation, 4),
        "samples": len(values),
        "confidence": round(len(values) / len(analyses), 2),
    }


def _normalized_hues(
    analyses: list[dict[str, Any]],
) -> list[int]:
    hues: list[int] = []

    for item in analyses:
        values = item.get("dominant_hues_degrees", [])

        if not isinstance(values, list):
            continue

        for value in values:
            if isinstance(value, bool):
                continue

            try:
                hue = float(value)
            except (TypeError, ValueError, OverflowError):
                continue

            if math.isfinite(hue):
                hues.append(round(hue) % 360)

    return hues


def _hue_palette(
    analyses: list[dict[str, Any]],
) -> list[str]:
    hues = _normalized_hues(analyses)

    if not hues:
        return []

    counts = Counter(hues)

    return [
        f"hsl({hue}, 62%, 52%)"
        for hue, _ in counts.most_common(4)
    ]


def _composition_profile(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    x_values: list[float] = []
    y_values: list[float] = []

    for item in analyses:
        centroid = item.get("composition_centroid")

        if not isinstance(centroid, dict):
            continue

        x = _valid_numbers([centroid.get("x")])
        y = _valid_numbers([centroid.get("y")])

        if x and 0 <= x[0] <= 1:
            x_values.extend(x)

        if y and 0 <= y[0] <= 1:
            y_values.extend(y)

    return {
        "edge_centroid_x": (
            round(median(x_values), 4) if x_values else None
        ),
        "edge_centroid_y": (
            round(median(y_values), 4) if y_values else None
        ),
        "interpretation": (
            "edge_distribution_observation_not_caption_position"
        ),
    }


def _classify_motion(
    metrics: dict[str, Any],
) -> str:
    motion = metrics.get("motion")
    cut_rate = metrics.get("cut_rate")

    motion_value = motion["median"] if motion else None
    cut_value = cut_rate["median"] if cut_rate else None

    if motion_value is None and cut_value is None:
        return "unknown"

    if motion_value is not None and motion_value >= 0.035:
        return "high_motion"

    if cut_value is not None and cut_value >= 0.12:
        return "fast_cut_pacing"

    if motion_value is not None and motion_value <= 0.008:
        return "subtle_motion"

    return "moderate_motion"


def _majority_preference(
    analyses: list[dict[str, Any]],
    key: str,
) -> str:
    values = []

    for item in analyses:
        value = item.get(key)

        if not isinstance(value, str):
            continue

        value = value.strip()

        if value and value.lower() != "unknown":
            values.append(value)

    counts = Counter(values)

    if not counts:
        return "unknown"

    value, count = counts.most_common(1)[0]

    return (
        value
        if count > len(analyses) / 2
        else "unknown"
    )


def _ocr_examples(
    analyses: list[dict[str, Any]],
) -> list[str]:
    examples: list[str] = []

    for item in analyses:
        regions = item.get("text_regions", [])

        if not isinstance(regions, list):
            continue

        for region in regions:
            if not isinstance(region, dict):
                continue

            text = region.get("text")

            if not isinstance(text, str):
                continue

            text = " ".join(text.split())

            if text and text not in examples:
                examples.append(text)

            if len(examples) >= 30:
                return examples

    return examples


def _observations(
    analyses: list[dict[str, Any]],
) -> list[str]:
    result: list[str] = []

    for item in analyses:
        values = item.get("observations", [])

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, list):
            continue

        for value in values:
            if isinstance(value, str):
                value = value.strip()

                if value and value not in result:
                    result.append(value)

    return result


def extract_style(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate available reference measurements without claiming font detection."""

    analyses = _valid_analyses(analyses)

    metric_names = (
        "brightness",
        "contrast",
        "saturation",
        "cut_rate",
        "motion",
        "frame_difference",
        "caption_region_score",
    )

    metrics = {
        name: _metric(analyses, name)
        for name in metric_names
    }

    transition = _majority_preference(
        analyses,
        "transition_style",
    )

    caption = _majority_preference(
        analyses,
        "caption_style",
    )

    palette = _hue_palette(analyses)
    normalized_hues = _normalized_hues(analyses)
    composition = _composition_profile(analyses)
    motion_profile = _classify_motion(metrics)

    caption_score = metrics.get("caption_region_score")

    caption_presence = (
        "possible_caption_region_signal"
        if caption_score and caption_score["median"] > 0
        else "not_detected_or_unavailable"
    )

    hue_counts = Counter(normalized_hues)

    dominant_hues = [
        hue
        for hue, count in hue_counts.most_common(4)
        if count > len(analyses) / 2
    ]

    return {
        "reference_count": len(analyses),
        "metrics": metrics,
        "transition_preference": transition,
        "caption_style": caption,
        "dominant_hues_degrees": dominant_hues,
        "palette_css": palette,
        "composition_centroid": composition,
        "style_profile": {
            "palette": palette,
            "motion": motion_profile,
            "transition": (
                transition if transition != "unknown" else "adaptive"
            ),
            "caption_presence": caption_presence,
            "caption_style": (
                caption if caption != "unknown" else "adaptive"
            ),
            "typography": {
                "family": "licensed_local_font_or_system_fallback",
                "weight": "adaptive",
                "size": "adaptive_to_reference_and_canvas",
                "exact_font_identification": False,
                "animation_identification": False,
            },
            "composition": composition,
            "confidence": round(
                sum(
                    metric["confidence"]
                    for metric in metrics.values()
                    if metric is not None
                )
                / max(
                    1,
                    sum(
                        metric is not None
                        for metric in metrics.values()
                    ),
                ),
                2,
            ),
        },
        "ocr_text_examples": _ocr_examples(analyses),
        "observations": _observations(analyses),
    }
