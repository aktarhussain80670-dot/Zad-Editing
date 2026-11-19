"""Semantic, narration-driven scene planning for universal motion graphics."""

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
    "with", "from", "into", "when", "where", "after", "before",
}

COMPARISON_TERMS = {
    "versus", "vs", "compared", "comparison", "difference",
    "different", "similar", "unlike", "whereas", "rather",
    "more", "less", "greater", "smaller", "higher", "lower",
}

PROCESS_TERMS = {
    "first", "second", "third", "next", "finally", "step",
    "process", "stage", "method", "sequence", "before", "after",
    "begins", "started", "start", "continues", "ends",
}

QUESTION_TERMS = {
    "why", "how", "what", "when", "where", "who", "which",
}

DATA_CONTEXT = {
    "percent", "percentage", "rate", "ratio", "total", "average",
    "population", "revenue", "income", "statistics", "statistic",
    "data", "score", "million", "billion", "thousand", "amount",
    "quantity", "measurement", "measured", "count", "number",
}

TREND_TERMS = {
    "increase", "increased", "increasing", "decrease", "decreased",
    "decreasing", "growth", "grew", "rise", "rose", "fall", "fell",
    "decline", "declined", "improved", "worsened", "expand",
    "expanded", "shrunk", "dropped",
}

CAUSE_TERMS = {
    "because", "caused", "cause", "causes", "result", "resulted",
    "consequence", "consequences", "leads", "leading", "reason",
    "reasons", "effect", "effects", "impact",
}

FIXED_PALETTE = [
    "#FFFFFF",
    "#000000",
    "#2453D6",
    "#22C55E",
]


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
    """Extract validated silence and emphasis timestamps."""

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
    """Validate word timing and preserve source metadata."""

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

        text = str(
            word.get("word", word.get("text", ""))
        ).strip()

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
    """Conservative lexical similarity used only as one split signal."""

    left_words = set(_words(left)) - CONNECTORS
    right_words = set(_words(right)) - CONNECTORS

    if not left_words or not right_words:
        return 0.0

    overlap = len(left_words & right_words)
    union = len(left_words | right_words)

    return overlap / union if union else 0.0


def _sentence_boundary(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…"))


def _pause_between(
    previous_end: float,
    next_start: float,
    pauses: list[float],
) -> bool:
    return any(
        previous_end <= pause <= next_start
        for pause in pauses
    )


def _should_split(
    current: list[dict[str, Any]],
    next_segment: dict[str, Any],
    current_start: float,
    next_end: float,
    pauses: list[float],
) -> bool:
    """Adaptive scene splitting based on timing and narration structure."""

    current_text = " ".join(
        str(item.get("text", ""))
        for item in current
    )

    next_text = str(next_segment.get("text", ""))

    current_duration = next_end - current_start

    previous_text = str(
        current[-1].get("text", "")
    ).strip()

    previous_words = _words(previous_text)
    next_words = _words(next_text)

    next_start = _time(
        next_segment.get("start"),
        "segment start",
    )

    gap = max(0.0, next_start - current[-1]["end"])

    pause_before = _pause_between(
        current[-1]["end"],
        next_start,
        pauses,
    )

    similarity = _semantic_similarity(
        current_text,
        next_text,
    )

    # Avoid excessively long scenes.
    if current_duration >= 7.0:
        return True

    # Strong silence is a natural scene boundary.
    if current_duration >= 2.4 and gap >= 0.65:
        return True

    if current_duration >= 3.0 and pause_before:
        return True

    # Sentence completion with a meaningful contextual shift.
    if (
        current_duration >= 3.0
        and _sentence_boundary(previous_text)
        and similarity < 0.18
    ):
        return True

    # Adaptive target duration, not a rigid fixed duration.
    if current_duration >= 5.2:
        return True

    # Preserve grammatical connections.
    if previous_words and next_words:
        if previous_words[-1] in CONNECTORS:
            return False

        if next_words[0] in CONNECTORS:
            return False

    # Prefer a meaningful change when the current scene is already long.
    if current_duration >= 4.0 and similarity < 0.08:
        return True

    return False


def _visual_category(text: str) -> str:
    """Conservatively classify a visual based on explicit narration clues."""

    words = set(_words(text))
    normalized = " ".join(_words(text))

    has_year = bool(YEAR_PATTERN.search(text))
    has_comparison = bool(words & COMPARISON_TERMS)
    has_process = bool(words & PROCESS_TERMS)
    has_question = bool(words & QUESTION_TERMS)
    has_cause = bool(words & CAUSE_TERMS)

    numbers = NUMBER_PATTERN.findall(text)
    has_numeric_data_context = bool(
        numbers and words & DATA_CONTEXT
    )

    has_trend = bool(words & TREND_TERMS)

    if has_comparison:
        return "comparison"

    if has_process:
        return "sequence"

    if has_year:
        return "timeline"

    if has_numeric_data_context:
        return "data_visualization"

    if has_trend and has_numeric_data_context:
        return "animated_chart"

    if has_question or has_cause or any(
        term in normalized.split()
        for term in ("means", "explains", "refers", "works")
    ):
        return "concept_explanation"

    return "conceptual_graphic"


def _extract_highlights(text: str) -> list[str]:
    """Select meaningful words without relying only on word length."""

    words = _words(text)
    candidates: list[str] = []

    for word in words:
        if (
            len(word) >= 4
            and word not in CONNECTORS
            and word not in candidates
        ):
            candidates.append(word)

    # Prefer numbers, proper-name-like terms, and concept-bearing words
    # only when present in the narration. Never invent text.
    return candidates[:4]


def _safe_style_profile(style: dict[str, Any]) -> dict[str, Any]:
    profile = style.get("style_profile", {})

    return profile if isinstance(profile, dict) else {}


def _scene_emphasis(
    emphasis_times: list[float],
    start: float,
    end: float,
) -> list[float]:
    return [
        round(value, 3)
        for value in emphasis_times
        if start <= value <= end
    ]


def _build_visual_plan(
    text: str,
    style: dict[str, Any],
    emphasis_times: list[float],
    start: float,
    end: float,
) -> dict[str, Any]:
    """Generate a structured visual composition brief."""

    category = _visual_category(text)
    highlights = _extract_highlights(text)
    style_profile = _safe_style_profile(style)

    numeric_values = NUMBER_PATTERN.findall(text)
    years = YEAR_PATTERN.findall(text)

    scene_emphasis = _scene_emphasis(
        emphasis_times,
        start,
        end,
    )

    category_definitions = {
        "comparison": {
            "primary": "comparison_layout",
            "support": [
                "balanced_comparison_regions",
                "animated_focus_indicator",
            ],
            "layout": "comparison",
        },
        "sequence": {
            "primary": "process_visual",
            "support": [
                "step_markers",
                "directional_flow",
            ],
            "layout": "sequence",
        },
        "timeline": {
            "primary": "chronological_timeline",
            "support": [
                "date_markers",
                "animated_timeline_path",
            ],
            "layout": "timeline",
        },
        "data_visualization": {
            "primary": "data_graphic",
            "support": [
                "numeric_callouts",
                "data_relationship_visual",
            ],
            "layout": "data",
        },
        "animated_chart": {
            "primary": "trend_graphic",
            "support": [
                "trend_direction_indicator",
                "change_visualization",
            ],
            "layout": "trend",
        },
        "concept_explanation": {
            "primary": "concept_diagram",
            "support": [
                "connected_concept_nodes",
                "relationship_visualization",
            ],
            "layout": "concept",
        },
        "conceptual_graphic": {
            "primary": "contextual_concept_visual",
            "support": [
                "layered_visual_elements",
                "kinetic_keywords",
            ],
            "layout": "adaptive",
        },
    }

    definition = category_definitions[category]

    primary = {
        "type": definition["primary"],
        "subject": text,
        "description": text,
        "key_terms": highlights,
        "numeric_values": numeric_values,
        "years": years,
    }

    supporting = [
        {
            "type": item,
            "content": highlights,
            "label": ", ".join(highlights),
            "text": ", ".join(highlights),
        }
        for item in definition["support"]
    ]

    supporting.append(
        {
            "type": "animated_background",
            "purpose": "visual_depth_and_scene_continuity",
        }
    )

    reference_composition = style_profile.get(
        "composition",
        {},
    )

    if not isinstance(reference_composition, dict):
        reference_composition = {}

    reference_motion = style_profile.get(
        "motion",
        "adaptive",
    )

    reference_transition = style_profile.get(
        "transition",
        style.get("transition_preference", "adaptive"),
    )

    reference_typography = style_profile.get(
        "typography",
        {},
    )

    if not isinstance(reference_typography, dict):
        reference_typography = {}

    return {
        "rendering_engine": "hyperframes",
        "implementation": "html_css_javascript",
        "visual_category": category,
        "category": category,

        "primary_element": primary,
        "supporting_elements": supporting,

        "composition": {
            "orientation": "portrait",
            "layout_type": definition["layout"],
            "layout_reference": "reference_informed_adaptive",
            "visual_hierarchy": [
                "primary_visual",
                "supporting_graphics",
                "caption",
            ],
            "reference_observations": reference_composition,
            "scene_balance": "adaptive",
            "visual_density": "reference_informed",
        },

        "style": {
            # Brand palette is immutable.
            "palette": list(FIXED_PALETTE),
            "brand_palette_locked": True,
            "reference_palette_observation": (
                style_profile.get("palette", [])
            ),
            "motion_reference": reference_motion,
            "transition_reference": reference_transition,
            "caption_style_reference": style_profile.get(
                "caption_style",
                "adaptive",
            ),
            "typography": reference_typography,
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
            "motion_intensity": "audio_and_context_adaptive",
        },

        "semantic_context": {
            "narration": text,
            "key_terms": highlights,
            "explicit_numbers": numeric_values,
            "explicit_years": years,
            "meaning_must_be_preserved": True,
            "relationships_must_be_preserved": True,
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
            "use_only_narration_supported_facts": True,
            "fixed_brand_palette": list(FIXED_PALETTE),
            "generate_original_composition": True,
        },
    }


def _normalize_segments(
    segments: list[Any],
    duration: float,
) -> list[dict[str, Any]]:
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
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": words,
                }
            )

        previous_start = start
        previous_end = end

    if not normalized:
        raise ValueError(
            "Transcript contains no usable narration segments"
        )

    return normalized


def _group_segments(
    segments: list[dict[str, Any]],
    pauses: list[float],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []

    current = [segments[0]]
    current_start = segments[0]["start"]

    for segment in segments[1:]:
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

    return groups


def plan_scenes(
    transcript: dict[str, Any],
    style: dict[str, Any],
    audio_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Plan complete voiceover coverage with adaptive semantic scenes."""

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

    normalized = _normalize_segments(
        segments,
        duration,
    )

    groups = _group_segments(
        normalized,
        pauses,
    )

    # Scene boundaries divide the full voiceover timeline.
    # This preserves pauses and avoids gaps between generated scenes.
    boundaries = [0.0]

    for left, right in zip(groups, groups[1:]):
        left_end = left[-1]["end"]
        right_start = right[0]["start"]

        boundary = (left_end + right_start) / 2.0

        boundary = max(
            boundaries[-1],
            min(duration, boundary),
        )

        boundaries.append(boundary)

    boundaries.append(duration)

    scenes: list[dict[str, Any]] = []

    caption_style = style.get(
        "caption_style",
        "adaptive",
    )

    transition_style = style.get(
        "transition_preference",
        "adaptive",
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

        scenes.append(
            {
                "scene_id": index + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "transcript": scene_text,

                "visual_idea": (
                    "Create an original "
                    f"{visual_plan['visual_category']} "
                    "motion-graphics composition communicating "
                    f"the meaning and relationships of this narration: "
                    f"{scene_text}"
                ),

                "visual_required": True,
                "visual_search_query": "",

                "visual_requirements": {
                    "orientation": "portrait",
                    "generated_with_html_css_javascript": True,
                    "avoid_reference_audio": True,
                    "avoid_reference_footage": True,
                    "no_external_images": True,
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
                    "font_family": "local_reference_informed",
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
                        if transition_style not in (
                            None,
                            "",
                            "unknown",
                        )
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
                    "enabled": False,
                },
            }
        )

    return scenes
