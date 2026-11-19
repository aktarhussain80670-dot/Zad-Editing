import pytest

from ai.scene_planner import plan_scenes
from ai.timeline_generator import generate_guide
from core.timeline_validator import validate_timeline


def _transcript(duration, text="The final word ends before trailing silence.", words=None):
    segment = {
        "start": 0.0,
        "end": min(duration, 2.0),
        "text": text,
    }
    if words is not None:
        segment["words"] = words
    return {
        "duration": duration,
        "text": text,
        "segments": [segment],
    }


def _guide(duration, transcript=None):
    transcript = transcript or _transcript(duration)
    scenes = plan_scenes(transcript, {})
    return generate_guide(
        transcript,
        {},
        scenes,
        audio_duration=duration,
    )


def test_trailing_silence_is_preserved_as_final_scene_duration():
    guide = _guide(5.75)
    assert guide["duration"] == 5.75
    assert guide["scenes"][-1]["end"] == 5.75


def test_audio_duration_overrides_stale_transcript_duration():
    audio_duration = 6.125
    transcript = _transcript(3.0)
    scenes = plan_scenes(_transcript(audio_duration), {})
    guide = generate_guide(
        transcript,
        {},
        scenes,
        audio_duration=audio_duration,
    )
    assert guide["transcript"]["duration"] == audio_duration
    assert guide["scenes"][-1]["end"] == audio_duration


def test_validator_rejects_short_and_long_timelines():
    short = _guide(4.0)
    short["scenes"][-1]["end"] = 3.0
    with pytest.raises(ValueError, match="Timeline must end"):
        validate_timeline(short)

    long = _guide(4.0)
    long["scenes"][-1]["end"] = 4.01
    with pytest.raises(ValueError):
        validate_timeline(long)


def test_hindi_word_timestamps_remain_in_order_and_scene_aligned():
    words = [
        {"start": 0.0, "end": 0.45, "word": "यह"},
        {"start": 0.46, "end": 0.95, "word": "एक"},
        {"start": 0.96, "end": 1.5, "word": "परीक्षण"},
    ]
    transcript = _transcript(
        2.75,
        text="यह एक परीक्षण",
        words=words,
    )
    transcript["language"] = "hi"
    guide = _guide(2.75, transcript)
    validate_timeline(guide)
    assert guide["scenes"][-1]["end"] == 2.75


def test_small_float_rounding_difference_is_accepted_only_as_precision():
    guide = _guide(3.141592)
    guide["scenes"][-1]["end"] = 3.1415
    validate_timeline(guide)

    guide["scenes"][-1]["end"] = 3.139
    with pytest.raises(ValueError):
        validate_timeline(guide)
