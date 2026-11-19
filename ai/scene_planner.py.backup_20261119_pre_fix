"""Plan semantic, voiceover-driven scenes for universal motion graphics."""

from __future__ import annotations

import math
import re
from typing import Any


WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)")
YEAR_PATTERN = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

CONNECTORS = {
    "and", "but", "because", "however", "therefore", "although",
    "meanwhile", "then", "next", "finally", "instead", "while",
    "so", "also", "which", "that", "this", "these", "those",
}

COMPARISON_TERMS = {
    "versus", "vs", "compared", "comparison", "difference",
    "different", "similar", "unlike", "whereas", "rather",
}

PROCESS_TERMS = {
    "first", "second", "third", "next", "finally", "step",
    "process", "stage", "method", "sequence", "before", "after",
}

QUESTION_TERMS = {
    "why", "how", "what", "when", "where", "who", "which",
}


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_PATTERN.findall(text)]


def _time(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field} timestamp")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Invalid {field} timestamp") from None

    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} timestamp")

    return result


def _audio_events(
    audio_analysis: dict[str, Any] | None,
    duration: float,
) -> tuple[list[float], list[float]]:
    """Extract finite audio event timestamps within the voiceover duration."""
    if not isinstance(audio_analysis, dict):
        return [], []

    def valid_times(
        container: Any,
        key: str,
        field: str,
    ) -> list[float]:
        if not isinstance(container, dict):
            return []

        events = container.get(key, [])

        if not isinstance(events, list):
            return []

        result: list[float] = []

        for event in events:
            if not isinstance(event, dict):
                continue

            try:
                value = _time(event.get(field), field)
            except ValueError:
                continue

            if 0 <= value < duration:
                result.append(value)

        return result

    pauses = valid_times(
        audio_analysis.get("silence", {}),
        "intervals",
        "start",
    )

    emphasis = valid_times(
        audio_analysis.get("energy", {}),
        "estimated_emphasis",
        "time",
    )

    return sorted(set(pauses)), sorted(set(emphasis))


def _validate_words(
    words: Any,
    segment_start: float,
    segment_end: float,
    index: int,
) -> list[dict[str, Any]]:
    """Validate word timestamps while preserving Whisper word metadata."""
    if words is None:
        return []

    if not isinstance(words, list):
        raise ValueError(
            f"Invalid word timestamp list in segment {index}"
        )

    validated: list[dict[str, Any]] = []
    previous_end = segment_start

    for word_index, word in enumerate(words):
        if not isinstance(word, dict):
            raise ValueError(
                f"Invalid word entry in segment {index}"
            )

        start = _time(
            word.get("start"),
            f"segment {index} word {word_index} start",
        )

        end = _time(
            word.get("end"),
            f"segment {index} word {word_index} end",
        )

        text = str(word.get("word", word.get("text", ""))).strip()

        if not text or start < segment_start - 0.05:
            raise ValueError(
                f"Invalid word timing in segment {index}"
            )

        if end <= start or end > segment_end + 0.05:
            raise ValueError(
                f"Word timestamp outside segment {index}"
            )

        if start < previous_end - 0.05:
            raise ValueError(
                f"Overlapping word timestamps in segment {index}"
            )

        start = max(segment_start, start)
        end = min(segment_end, end)

        if end <= start:
            raise ValueError(
                f"Invalid normalized word timing in segment {index}"
            )

        normalized_word = dict(word)
        normalized_word["start"] = start
        normalized_word["end"] = end
        normalized_word["word"] = text

        validated.append(normalized_word)
        previous_end = end

    return validated


def _semantic_similarity(left: str, right: str) -> float:
    left_words = set(_words(left)) - CONNECTORS
    right_words = set(_words(right)) - CONNECTORS

    if not left_words or not right_words:
        return 0.0

    overlap = len(left_words & right_words)
    union = len(left_words | right_words)

    return overlap / union if union else 0.0


def _should_split(
    current: list[dict[str, Any]],
    next_segment: dict[str, Any],
    current_start: float,
    next_end: float,
    pauses: list[float],
) -> bool:
    current_text = " ".join(
        str(item.get("text", ""))
        for item in current
    )

    next_text = str(next_segment.get("text", ""))

    current_duration = next_end - current_start
    similarity = _semantic_similarity(
        current_text,
        next_text,
    )

    previous_text = str(
        current[-1].get("text", "")
    ).strip()

    previous_words = _words(previous_text)
    next_words = _words(next_text)

    next_start = _time(
        next_segment.get("start"),
        "segment start",
    )

    pause_before = any(
        current[-1]["end"] <= pause <= next_start
        for pause in pauses
    )

    if current_duration >= 7.0:
        return True

    if current_duration >= 4.0 and pause_before:
        return True

    if (
        current_duration >= 4.0
        and previous_text.endswith(("!", "?", "."))
        and similarity < 0.12
    ):
        return True

    if current_duration >= 5.5:
        return True

    if previous_words and next_words:
        if previous_words[-1] in COMPARISON_TERMS:
            return False

        if next_words[0] in CONNECTORS:
            return False

    return False


def _visual_category(text: str) -> str:
    """Choose a visual form using conservative narration heuristics."""
    normalized = " ".join(_words(text))
    words = set(_words(text))

    has_year = bool(YEAR_PATTERN.search(text))
    has_comparison = bool(words & COMPARISON_TERMS)
    has_process = bool(words & PROCESS_TERMS)
    has_question = bool(words & QUESTION_TERMS)

    if has_comparison:
        return "comparison"

    if has_process:
        return "sequence"

    if has_year:
        return "timeline"

    data_context = {
        "percent", "percentage", "rate", "ratio", "total", "average",
        "population", "revenue", "income", "statistics", "statistic",
        "data", "score", "million", "billion",
    }

    trend_context = {
        "increase", "increased", "increasing", "decrease", "decreased",
        "decreasing", "growth", "grew", "rise", "rose", "fall", "fell",
        "decline", "declined",
    }

    has_number = bool(NUMBER_PATTERN.search(text))

    if has_number and words & data_context:
        return "data_visualization"

    if words & trend_context:
        return "animated_chart"

    if has_question or any(
        term in normalized.split()
        for term in ("because", "means", "explains", "refers")
    ):
        return "concept_explanation"

    return "conceptual_graphic"


def _extract_highlights(text: str) -> list[str]:
    words = _words(text)
    candidates: list[str] = []

    for word in words:
        if (
            len(word) >= 6
            and word not in CONNECTORS
            and word not in candidates
        ):
            candidates.append(word)

    return candidates[:4]


def _build_visual_plan(
    text: str,
    style: dict[str, Any],
    emphasis_times: list[float],
    start: float,
    end: float,
) -> dict[str, Any]:
    """Build a narration-led graphics plan informed by reference observations."""
    category = _visual_category(text)
    highlights = _extract_highlights(text)

    style_profile = style.get("style_profile", {})

    if not isinstance(style_profile, dict):
        style_profile = {}

    palette = (
        style_profile.get("palette")
        or style.get("palette_css")
        or []
    )

    if not isinstance(palette, list):
        palette = []

    scene_emphasis = [
        round(value, 3)
        for value in emphasis_times
        if start <= value <= end
    ]

    numeric_values = NUMBER_PATTERN.findall(text)
    years = YEAR_PATTERN.findall(text)

    if category == "comparison":
        primary_type = "comparison_layout"
        support = [
            "two_column_comparison",
            "animated_highlight",
        ]
    elif category == "sequence":
        primary_type = "step_timeline"
        support = [
            "numbered_steps",
            "directional_connector",
        ]
    elif category == "timeline":
        primary_type = "chronological_timeline"
        support = [
            "date_markers",
            "animated_timeline_path",
        ]
    elif category == "data_visualization":
        primary_type = "data_graphic"
        support = [
            "numeric_callouts",
            "chart_or_progress_indicator",
        ]
    elif category == "animated_chart":
        primary_type = "trend_graphic"
        support = [
            "animated_trend_line",
            "change_indicator",
        ]
    elif category == "concept_explanation":
        primary_type = "concept_diagram"
        support = [
            "connected_concept_nodes",
            "kinetic_keywords",
        ]
    else:
        primary_type = "abstract_concept_composition"
        support = [
            "layered_shapes",
            "kinetic_keywords",
        ]

    return {
        "rendering_engine": "hyperframes",
        "implementation": "html_css_javascript",
        "visual_category": category,
        "category": category,
        "primary_element": {
            "type": primary_type,
            "subject": text,
            "key_terms": highlights,
            "numeric_values": numeric_values,
            "years": years,
        },
        "supporting_elements": [
            {
                "type": item,
                "content": highlights,
                "label": ", ".join(highlights),
                "text": ", ".join(highlights),
            }
            for item in support
        ] + [
            {
                "type": "animated_background",
                "purpose": "visual_depth_and_scene_continuity",
            }
        ],
        "composition": {
            "orientation": "portrait",
            "layout_reference": "reference_informed_adaptive",
            "visual_hierarchy": [
                "primary_visual",
                "supporting_graphics",
                "caption",
            ],
            "reference_edge_centroid": style_profile.get(
                "composition",
                {},
            ),
        },
        "style": {
            "palette": palette,
            "motion_reference": style_profile.get(
                "motion",
                "unknown",
            ),
            "transition_reference": style_profile.get(
                "transition",
                style.get("transition_preference", "adaptive"),
            ),
            "caption_style_reference": style_profile.get(
                "caption_style",
                "adaptive",
            ),
            "typography": style_profile.get(
                "typography",
                {
                    "family": "licensed_local_font_or_system_fallback",
                    "weight": "adaptive",
                    "size": "adaptive",
                },
            ),
            "reference_confidence": style_profile.get(
                "confidence",
                0.0,
            ),
        },
        "animation": {
            "sync_to_voiceover": True,
            "emphasis_timestamps": scene_emphasis,
            "entrance": "context_aware",
            "exit": "transition_aware",
        },
        "generation_constraints": {
            "use_external_video": False,
            "use_external_images": False,
            "use_reference_footage": False,
            "use_reference_audio": False,
            "avoid_generic_text_card": True,
            "narration_is_source_of_truth": True,
            "do_not_invent_data": True,
            "charts_require_explicit_data_context": True,
        },
    }


def plan_scenes(
    transcript: dict[str, Any],
    style: dict[str, Any],
    audio_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(transcript, dict):
        raise TypeError("Transcript must be a dictionary")

    if not isinstance(style, dict):
        raise TypeError("Style profile must be a dictionary")

    duration = _time(
        transcript.get("duration", 0),
        "transcript duration",
    )

    segments = transcript.get("segments", [])

    if (
        duration <= 0
        or not isinstance(segments, list)
        or not segments
    ):
        raise ValueError(
            "Transcript must have positive duration and timed segments"
        )

    pauses, emphasis_times = _audio_events(
        audio_analysis,
        duration,
    )

    normalized: list[dict[str, Any]] = []
    previous_start = -1.0
    previous_end = 0.0

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(
                f"Invalid transcript segment at index {index}"
            )

        start = _time(
            segment.get("start"),
            "segment start",
        )

        end = _time(
            segment.get("end"),
            "segment end",
        )

        text = str(segment.get("text", "")).strip()

        if (
            start < 0
            or end <= start
            or end > duration + 0.05
        ):
            raise ValueError(
                f"Invalid transcript segment timing at index {index}"
            )

        if start < previous_start:
            raise ValueError(
                f"Transcript segments are out of order at index {index}"
            )

        if start < previous_end - 0.05:
            raise ValueError(
                f"Overlapping transcript segments at index {index}"
            )

        start = min(start, duration)
        end = min(end, duration)

        if end <= start:
            raise ValueError(
                f"Segment outside usable voiceover duration at index {index}"
            )

        words = _validate_words(
            segment.get("words", []),
            start,
            end,
            index,
        )

        if text:
            normalized.append({
                "start": start,
                "end": end,
                "text": text,
                "words": words,
            })

        previous_start = start
        previous_end = end

    if not normalized:
        raise ValueError(
            "Transcript contains no usable narration segments"
        )

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [normalized[0]]
    current_start = normalized[0]["start"]

    for segment in normalized[1:]:
        if _should_split(
            current,
            segment,
            current_start,
            segment["end"],
            pauses,
        ):
            groups.append(current)
            current = [segment]
            current_start = segment["start"]
        else:
            current.append(segment)

    groups.append(current)

    # Scenes cover the complete voiceover, including silent intervals.
    boundaries = [0.0]

    for left, right in zip(groups, groups[1:]):
        left_end = left[-1]["end"]
        right_start = right[0]["start"]

        boundary = (left_end + right_start) / 2

        boundaries.append(
            max(
                boundaries[-1],
                min(duration, boundary),
            )
        )

    boundaries.append(duration)

    scenes: list[dict[str, Any]] = []

    caption_style = style.get("caption_style", "unknown")
    transition_style = style.get(
        "transition_preference",
        "unknown",
    )

    for index, group in enumerate(groups):
        start = boundaries[index]
        end = boundaries[index + 1]

        scene_text = " ".join(
            item["text"]
            for item in group
        ).strip()

        if end <= start:
            raise ValueError(
                f"Invalid generated scene timing at index {index}"
            )

        highlights = _extract_highlights(scene_text)

        visual_plan = _build_visual_plan(
            scene_text,
            style,
            emphasis_times,
            start,
            end,
        )

        word_timestamps = [
            word
            for item in group
            for word in item["words"]
        ]

        scenes.append({
            "scene_id": index + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "transcript": scene_text,
            "visual_idea": (
                f"Create a {visual_plan['visual_category']} "
                f"motion-graphics composition communicating: "
                f"{scene_text}"
            ),
            "visual_required": True,
            "visual_search_query": "",
            "visual_requirements": {
                "orientation": "portrait",
                "generated_with_html_css_javascript": True,
                "avoid_reference_audio": True,
            },
            "visual_plan": visual_plan,
            "caption": {
                "enabled": True,
                "text": scene_text,
                "word_timestamps": word_timestamps,
                "animation": (
                    "kinetic"
                    if caption_style == "kinetic"
                    else "adaptive"
                ),
            },
            "typography": {
                "font_family": "reference_informed",
                "weight": 800,
                "position": "style_adaptive",
                "highlight_terms": highlights,
                "confidence": (
                    "reference_observation_or_runtime_fallback"
                ),
            },
            "motion": {
                "enabled": True,
                "type": "context_aware",
                "sync_to_voiceover": True,
            },
            "transition": {
                "type": (
                    transition_style
                    if transition_style != "unknown"
                    else "adaptive"
                ),
            },
            "sound_effect": {
                "enabled": False,
            },
            "audio": {
                "music_enabled": False,
                "sound_effect_enabled": False,
            },
            "external_media": {
                "required": False,
                "query": "",
            },
        })

    return scenes
