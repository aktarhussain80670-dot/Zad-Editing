from ai.scene_planner import _visual_category, plan_scenes
from core.timeline_validator import validate_timeline
from services.motion_graphics_service import MotionGraphicsService
from services.subtitle_service import SubtitleService


def test_trend_data_selects_chart_and_preserves_visual_roles():
    transcript = {
        "duration": 4.0,
        "text": "Revenue increased by 20 percent.",
        "segments": [{
            "start": 0.0,
            "end": 4.0,
            "text": "Revenue increased by 20 percent.",
            "words": [],
        }],
    }
    scenes = plan_scenes(transcript, {})
    assert _visual_category(transcript["text"]) == "animated_chart"
    assert scenes[0]["visual_plan"]["a_roll"]["role"] == "primary"
    assert scenes[0]["visual_plan"]["b_roll"][0]["role"] == "supporting"


def test_generated_concept_scene_is_renderable_and_validated():
    transcript = {
        "duration": 3.0,
        "text": "A connected system explains the concept.",
        "segments": [{
            "start": 0.0,
            "end": 3.0,
            "text": "A connected system explains the concept.",
            "words": [],
        }],
    }
    scenes = plan_scenes(transcript, {})
    guide = {
        "schema_version": 2,
        "duration": 3.0,
        "transcript": transcript,
        "audio_analysis": {},
        "style": {},
        "scenes": scenes,
    }
    validate_timeline(guide)
    markup = MotionGraphicsService().render_scene(scenes[0])
    assert "motion-scene" in markup


def test_subtitle_service_skips_malformed_timestamps():
    transcript = {"segments": [{"start": "bad", "end": 1, "text": "ignored"}]}
    assert SubtitleService().cues(transcript) == []
