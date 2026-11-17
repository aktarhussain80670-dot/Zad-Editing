"""Run local analysis through validated editing-guide generation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ai.editing_director import direct
from ai.style_extractor import extract_style
from ai.timeline_generator import generate_guide
from ai.voiceover_analyzer import VoiceoverAnalyzer
from config.settings import Settings
from core.checkpoint_manager import CheckpointManager
from core.cleanup_manager import CleanupManager
from core.feature_controller import enabled_features
from core.input_manager import validate_inputs
from core.media_selector import MediaSelector
from providers.http import ProviderError
from services.ffmpeg_service import MediaToolError
from services.ffprobe_service import FFprobeService
from services.hyperframes_service import HyperFramesService
from services.media_download_service import MediaDownloadService
from services.media_search_service import MediaSearchService
from services.reference_video_service import ReferenceVideoService
from services.video_renderer import VideoRenderer

logger = logging.getLogger(__name__)


def run_pipeline(
    voiceover: Path,
    references: list[Path],
    settings: Settings | None = None,
    guide_path: Path = Path("storage/editing_guides/editing_guide.json"),
    render: bool = True,
    resume: bool = True,
    music: Path | None = None,
    sound_effects: list[tuple[Path, float]] | None = None,
) -> dict[str, Any]:
    config = settings or Settings.from_env()
    config.validate()
    validate_inputs(voiceover, references, config)
    if music is not None and config.enable_background_music and not music.is_file():
        raise ValueError(f"Optional background music was not found: {music}")
    if config.enable_sound_effects:
        for effect_path, timestamp in sound_effects or []:
            if not effect_path.is_file() or timestamp < 0:
                raise ValueError(f"Optional sound effect is missing or has an invalid timestamp: {effect_path}")
    optional_audio = []
    if config.enable_background_music and music is not None:
        optional_audio.append(music)
    if config.enable_sound_effects:
        optional_audio.extend(path for path, _ in sound_effects or [])
    fingerprint = CheckpointManager().fingerprint(
        [voiceover, *references, *optional_audio],
        {
            "settings": config,
            "guide_path": str(guide_path.resolve()),
            "sound_effects": [(str(path.resolve()), timestamp) for path, timestamp in sound_effects or []],
        },
    )
    checkpoint = CheckpointManager()
    previous = checkpoint.load(fingerprint) if resume else None
    state: dict[str, Any] = previous.get("data", {}) if previous else {}
    generated = [Path(p) for p in previous.get("generated_files", [])] if previous else []
    last_stage = previous.get("last_completed_stage") if previous else None
    run_temp = Path("temp") / f"{fingerprint[:12]}-{uuid.uuid4().hex[:8]}"
    try:
        if "reference_analyses" not in state:
            logger.info("Analyzing %d reference videos", len(references))
            state["reference_analyses"] = ReferenceVideoService().analyze_many(references)
            checkpoint.save(fingerprint, "references", generated, data=state)
        if "transcript" not in state:
            logger.info("Transcribing original voiceover with local Whisper")
            state["transcript"] = VoiceoverAnalyzer(
                config.whisper_model,
                config.whisper_language,
                config.whisper_device,
                config.whisper_download_root,
                config.whisper_allow_model_download,
            ).analyze(voiceover)
            if not config.video_min_duration <= float(state["transcript"]["duration"]) <= config.video_max_duration:
                raise ValueError("Voiceover duration is outside configured VIDEO_MIN_DURATION / VIDEO_MAX_DURATION")
            checkpoint.save(fingerprint, "transcription", generated, data=state)
        if "guide" not in state:
            state["style"] = extract_style(state["reference_analyses"])
            state["scenes"] = direct(state["transcript"], state["style"], enabled_features(config))
            state["guide"] = generate_guide(state["transcript"], state["style"], state["scenes"], guide_path)
            generated.append(guide_path)
            checkpoint.save(fingerprint, "guide", generated, data=state)
        guide = state["guide"]
        logger.info("Editing guide saved to %s", guide_path)
        if not render:
            return guide
        final_path = config.output_directory / f"{voiceover.stem}.{config.output_format}"
        if last_stage == "rendered" and final_path.is_file():
            try:
                FFprobeService().validate(
                    final_path, require_video=True, require_audio=True, expected_duration=float(guide["duration"])
                )
                return guide
            except (MediaToolError, ValueError, OSError) as exc:
                logger.warning("Ignoring invalid cached render and rerendering: %s", exc)
        assets: dict[int, Path] = {}
        metadata_dir = Path("storage/metadata/provider_assets")
        media_search = MediaSearchService(config)
        downloader = MediaDownloadService(config.media_timeout_seconds, config.max_download_bytes)
        for scene in MediaSelector(config).scenes_requiring_clips(guide):
            try:
                candidates = media_search.search_scene(scene)
                if candidates:
                    assets[int(scene["scene_id"])] = downloader.download(
                        candidates[0], run_temp / "media", metadata_dir
                    )
            except (ProviderError, MediaToolError, ValueError, OSError, TimeoutError) as exc:
                logger.warning("Optional visual media skipped for scene %s: %s", scene.get("scene_id"), exc)
        renderer = VideoRenderer(hyperframes=HyperFramesService(config.hyperframes_command))
        if config.enable_background_music and music is None and config.pixabay_api_key:
            logger.warning("Pixabay's documented public API does not support background music search; no bed was added")
        if config.enable_sound_effects and not sound_effects and config.pixabay_api_key:
            logger.warning(
                "Pixabay's documented public API does not support sound-effect search; no effects were added"
            )
        renderer.render(
            guide,
            voiceover,
            assets,
            final_path,
            config,
            music=music,
            effects=sound_effects,
            work_directory=run_temp / "render",
        )
        generated.append(final_path)
        generated.append(Path("storage/metadata") / f"{final_path.stem}.json")
        checkpoint.save(fingerprint, "rendered", generated, data=state)
        cleanup = CleanupManager()
        cleanup.remove_processed_voiceover(config, [run_temp / "render/mix.m4a"], run_temp)
        cleanup.cleanup(config, success=True, run_directory=run_temp)
        return guide
    except Exception as exc:
        checkpoint.save(fingerprint, "failed", generated, status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("Pipeline failed; the last valid checkpoint is retained")
        raise
