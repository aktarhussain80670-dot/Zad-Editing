import pytest

from ai.scene_planner import plan_scenes
from ai.style_extractor import extract_style
from core.timeline_validator import validate_timeline


def test_style_uses_median_and_unknown_when_unobserved():
    profile = extract_style([{"brightness": 0.2}, {"brightness": 0.8}])
    assert profile["metrics"]["brightness"]["median"] == 0.5
    assert profile["transition_preference"] == "unknown"


def test_planner_covers_voiceover_duration_contiguously():
    transcript = {"duration": 5.0, "segments": [{"start": 0.4, "end": 1.5, "text": "First phrase."}, {"start": 2.0, "end": 4.4, "text": "Second phrase."}]}
    scenes = plan_scenes(transcript, {})
    guide = {"duration": 5.0, "scenes": scenes}
    validate_timeline(guide)
    assert scenes[0]["start"] == 0
    assert scenes[-1]["end"] == 5


def test_timeline_rejects_overlap():
    with pytest.raises(ValueError):
        validate_timeline({"duration": 3, "scenes": [{"scene_id": 1, "start": 0, "end": 2}, {"scene_id": 2, "start": 1, "end": 3}]})
