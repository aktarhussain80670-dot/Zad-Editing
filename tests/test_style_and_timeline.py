import pytest

from ai.scene_planner import plan_scenes
from ai.style_extractor import extract_style
from core.timeline_validator import validate_timeline


def test_style_uses_median_and_unknown_when_unobserved():
    profile = extract_style([{"brightness": 0.2}, {"brightness": 0.8}])
    assert profile["metrics"]["brightness"]["median"] == 0.5
    assert profile["transition_preference"] == "unknown"


def test_planner_covers_voiceover_duration_contiguously():
    transcript = {
        "duration": 5.0,
        "segments": [
            {"start": 0.4, "end": 1.5, "text": "First phrase."},
            {"start": 2.0, "end": 4.4, "text": "Second phrase."},
        ],
    }
    scenes = plan_scenes(transcript, {})
    guide = {
        "schema_version": 1,
        "duration": 5.0,
        "transcript": {**transcript, "text": "First phrase. Second phrase."},
        "style": {},
        "scenes": scenes,
    }
    validate_timeline(guide)
    assert scenes[0]["start"] == 0
    assert scenes[-1]["end"] == 5


def test_timeline_rejects_overlap():
    scene = {
        "transcript": "text",
        "visual_idea": "visual",
        "visual_search_query": "query",
        "typography": {},
        "caption": {"enabled": True, "text": "text", "animation": "fade"},
        "motion": {"type": "zoom"},
        "transition": {"type": "cut"},
        "visual_required": True,
        "external_media": {"required": False, "query": "query"},
        "sound_effect": {"enabled": False},
        "audio": {"music_enabled": False, "sound_effect_enabled": False},
    }
    with pytest.raises(ValueError):
        validate_timeline(
            {
                "schema_version": 1,
                "duration": 3,
                "transcript": {"duration": 3, "text": "test", "segments": [{"start": 0, "end": 3, "text": "test"}]},
                "style": {},
                "scenes": [
                    {**scene, "scene_id": 1, "start": 0, "end": 2},
                    {**scene, "scene_id": 2, "start": 1, "end": 3},
                ],
            }
        )
