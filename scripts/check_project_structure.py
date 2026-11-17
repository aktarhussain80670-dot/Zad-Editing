"""Fail CI when essential editor modules are missing."""

from pathlib import Path

REQUIRED = (
    "ai/reference_analyzer.py",
    "ai/voiceover_analyzer.py",
    "ai/style_extractor.py",
    "ai/editing_director.py",
    "ai/scene_planner.py",
    "ai/timeline_generator.py",
    "core/pipeline.py",
    "core/input_manager.py",
    "core/feature_controller.py",
    "core/timeline_validator.py",
    "core/media_selector.py",
    "core/audio_controller.py",
    "core/cleanup_manager.py",
    "services/ffmpeg_service.py",
    "services/ffprobe_service.py",
    "services/hyperframes_service.py",
    "services/video_renderer.py",
    "services/audio_mixer.py",
    "services/subtitle_service.py",
    "providers/pexels_provider.py",
    "providers/pixabay_provider.py",
    "config/editing_guide.schema.json",
    "core/checkpoint_manager.py",
    ".github/workflows/edit-video.yml",
    ".github/workflows/validate-project.yml",
)

missing = [path for path in REQUIRED if not Path(path).is_file()]
if missing:
    raise SystemExit("Required project files are missing: " + ", ".join(missing))
print(f"Project structure verified ({len(REQUIRED)} required files)")
