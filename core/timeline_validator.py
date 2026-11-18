"""Strict structural and timing validation for version 2 editing guides."""

from __future__ import annotations

import math
from typing import Any


TIME_TOLERANCE = 0.05


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")

    return result


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")


def _boolean(value: Any, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")


def _validate_words(
    words: Any,
    start: float,
    end: float,
    field: str,
) -> None:
    words = _list(words, field)
    previous_end = start

    for index, item in enumerate(words, 1):
        word = _object(item, f"{field} {index}")

        word_start = _number(word.get("start"), "Word start")
        word_end = _number(word.get("end"), "Word end")

        word_text = word.get("word", word.get("text"))
        _string(word_text, "Word text")

        if (
            word_start < start - TIME_TOLERANCE
            or word_end <= word_start
            or word_end > end + TIME_TOLERANCE
        ):
            raise ValueError(f"{field} {index} has invalid timestamps")

        if word_start < previous_end - TIME_TOLERANCE:
            raise ValueError(f"{field} timestamps are out of order or overlapping")

        previous_end = word_end


def validate_timeline(guide: dict[str, Any]) -> None:
    guide = _object(guide, "Guide")

    if guide.get("schema_version") != 2:
        raise ValueError("Unsupported editing guide schema version")

    duration = _number(guide.get("duration"), "Guide duration")

    if duration <= 0:
        raise ValueError("Guide duration must be positive")

    transcript = _object(guide.get("transcript"), "Transcript")
    _string(transcript.get("text"), "Guide transcript text")

    transcript_duration = _number(
        transcript.get("duration"), "Transcript duration"
    )

    if abs(transcript_duration - duration) > TIME_TOLERANCE:
        raise ValueError("Guide duration must match original voiceover duration")

    segments = _list(transcript.get("segments"), "Transcript segments")

    if not segments:
        raise ValueError("Transcript requires timed segments")

    previous_segment_end = 0.0

    for index, item in enumerate(segments, 1):
        segment = _object(item, f"Transcript segment {index}")
        _string(segment.get("text"), f"Transcript segment {index} text")

        start = _number(segment.get("start"), "Segment start")
        end = _number(segment.get("end"), "Segment end")

        if start < 0 or end <= start or end > duration + TIME_TOLERANCE:
            raise ValueError(f"Transcript segment {index} has invalid timestamps")

        if start < previous_segment_end - TIME_TOLERANCE:
            raise ValueError("Transcript segments are out of order or overlapping")

        previous_segment_end = end

        words = segment.get("words", [])
        _validate_words(
            words,
            start,
            end,
            f"Transcript segment {index} word",
        )

    _object(guide.get("audio_analysis"), "Audio analysis")
    _object(guide.get("style"), "Guide style profile")

    scenes = _list(guide.get("scenes"), "Scenes")

    if not scenes:
        raise ValueError("Guide requires at least one scene")

    cursor = 0.0

    for expected_id, item in enumerate(scenes, 1):
        scene = _object(item, f"Scene {expected_id}")

        scene_id = scene.get("scene_id")
        if isinstance(scene_id, bool) or scene_id != expected_id:
            raise ValueError("Scene IDs must be sequential integers")

        start = _number(scene.get("start"), "Scene start")
        end = _number(scene.get("end"), "Scene end")

        if abs(start - cursor) > TIME_TOLERANCE or end <= start:
            raise ValueError(
                f"Scene {expected_id} has a gap, overlap, or invalid duration"
            )

        if end > duration + TIME_TOLERANCE:
            raise ValueError(f"Scene {expected_id} exceeds guide duration")

        _string(scene.get("transcript"), "Scene transcript")
        _string(scene.get("visual_idea"), "Scene visual idea")

        if scene.get("visual_search_query") != "":
            raise ValueError("External visual search queries must remain empty")

        _boolean(scene.get("visual_required"), "Visual required")

        requirements = _object(
            scene.get("visual_requirements"), "Visual requirements"
        )

        if requirements.get("orientation") != "portrait":
            raise ValueError("Visual requirements must use portrait orientation")

        if requirements.get("generated_with_html_css_javascript") is not True:
            raise ValueError("Visuals must be generated with HTML/CSS/JavaScript")

        if requirements.get("avoid_reference_audio") is not True:
            raise ValueError("Reference audio must not be used")

        caption = _object(scene.get("caption"), "Scene caption")
        typography = _object(scene.get("typography"), "Scene typography")
        motion = _object(scene.get("motion"), "Scene motion")
        transition = _object(scene.get("transition"), "Scene transition")
        sound_effect = _object(scene.get("sound_effect"), "Scene sound effect")
        audio = _object(scene.get("audio"), "Scene audio")
        external_media = _object(scene.get("external_media"), "Scene external media")
        visual_plan = _object(scene.get("visual_plan"), "Scene visual plan")

        _boolean(caption.get("enabled"), "Caption enabled")
        _string(caption.get("text"), "Caption text")
        _string(caption.get("animation"), "Caption animation")

        if "word_timestamps" in caption:
            _validate_words(
                caption["word_timestamps"],
                start,
                end,
                "Caption word",
            )

        _boolean(motion.get("enabled"), "Motion enabled")
        _string(motion.get("type"), "Motion type")

        if motion.get("sync_to_voiceover") is not True:
            raise ValueError("Motion must synchronize to voiceover")

        _string(transition.get("type"), "Transition type")
        _boolean(sound_effect.get("enabled"), "Sound effect enabled")
        _boolean(audio.get("music_enabled"), "Music enabled")
        _boolean(audio.get("sound_effect_enabled"), "Audio sound effect enabled")

        if external_media.get("required") is not False:
            raise ValueError("External media must remain disabled")

        if external_media.get("query") != "":
            raise ValueError("External media query must remain empty")

        if visual_plan.get("rendering_engine") != "hyperframes":
            raise ValueError("Every scene must use HyperFrames")

        if visual_plan.get("implementation") != "html_css_javascript":
            raise ValueError("Scene visuals must use HTML/CSS/JavaScript")

        _string(visual_plan.get("visual_category"), "Visual category")

        primary = _object(
            visual_plan.get("primary_element"), "Primary visual element"
        )
        _string(primary.get("type"), "Primary visual type")
        _string(primary.get("subject"), "Primary visual subject")

        key_terms = _list(primary.get("key_terms"), "Visual key terms")
        for term in key_terms:
            _string(term, "Visual key term")

        supporting = _list(
            visual_plan.get("supporting_elements"),
            "Supporting visual elements",
        )

        for index, element in enumerate(supporting, 1):
            _object(element, f"Supporting visual element {index}")

        composition = _object(
            visual_plan.get("composition"), "Visual composition"
        )

        if composition.get("orientation") != "portrait":
            raise ValueError("Visual composition must use portrait orientation")

        _string(composition.get("layout_reference"), "Layout reference")

        hierarchy = _list(
            composition.get("visual_hierarchy"), "Visual hierarchy"
        )
        for item in hierarchy:
            _string(item, "Visual hierarchy item")

        _object(visual_plan.get("style"), "Visual style")

        animation = _object(
            visual_plan.get("animation"), "Visual animation"
        )

        if animation.get("sync_to_voiceover") is not True:
            raise ValueError("Scene animation must synchronize to voiceover")

        emphasis = _list(
            animation.get("emphasis_timestamps"),
            "Animation emphasis timestamps",
        )

        for timestamp in emphasis:
            timestamp = _number(timestamp, "Emphasis timestamp")
            if timestamp < start or timestamp > end:
                raise ValueError("Emphasis timestamp is outside its scene")

        _string(animation.get("entrance"), "Animation entrance")
        _string(animation.get("exit"), "Animation exit")

        constraints = _object(
            visual_plan.get("generation_constraints"),
            "Visual generation constraints",
        )

        for key in (
            "use_external_video",
            "use_external_images",
            "use_reference_footage",
            "use_reference_audio",
        ):
            if constraints.get(key) is not False:
                raise ValueError(f"Visual constraint {key} must be disabled")

        if constraints.get("avoid_generic_text_card") is not True:
            raise ValueError("Generic text-card fallback is not permitted")

        cursor = end

    if abs(cursor - duration) > TIME_TOLERANCE:
        raise ValueError("Timeline must end at original voiceover duration")
