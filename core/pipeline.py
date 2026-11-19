"""Run local analysis through validated editing-guide generation."""

from __future__ import annotations

import json
import logging
import math
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
from services.audio_analysis_service import AudioAnalysisService
from services.ffmpeg_service import MediaToolError
from services.ffprobe_service import FFprobeService
from services.hyperframes_service import HyperFramesService
from services.reference_video_service import ReferenceVideoService
from services.transliteration_service import TransliterationService
from services.video_renderer import VideoRenderer

logger = logging.getLogger(__name__)

_STAGE_ORDER = {
    "references": 1,
    "transcription": 2,
    "audio_analysis": 3,
    "guide": 4,
    "rendered": 5,
}


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths while preserving their original order."""
    result: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            result.append(Path(resolved))

    return result


def _checkpoint_is_consistent(checkpoint_data: dict[str, Any]) -> bool:
    """Reject checkpoints whose completed stage disagrees with their state."""
    stage = checkpoint_data.get("last_completed_stage")

    if stage not in _STAGE_ORDER:
        return False

    data = checkpoint_data.get("data")
    if not isinstance(data, dict):
        return False

    required_by_stage = {
        "references": ("reference_analyses",),
        "transcription": ("reference_analyses", "transcript"),
        "audio_analysis": (
            "reference_analyses",
            "transcript",
            "audio_analysis",
        ),
        "guide": (
            "reference_analyses",
            "transcript",
            "audio_analysis",
            "style",
            "scenes",
            "caption_scenes",
            "guide",
        ),
        "rendered": (
            "reference_analyses",
            "transcript",
            "audio_analysis",
            "style",
            "scenes",
            "caption_scenes",
            "guide",
        ),
    }

    return all(key in data for key in required_by_stage[stage])


def _validate_cached_guide(
    guide: Any,
    guide_path: Path,
) -> bool:
    """Ensure cached guide data and its persisted artifact are both valid."""
    if not isinstance(guide, dict) or not guide:
        return False

    if not guide_path.is_file() or guide_path.stat().st_size == 0:
        return False

    try:
        saved_guide = json.loads(guide_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False

    return isinstance(saved_guide, dict) and saved_guide == guide


def _validate_cached_render(
    path: Path,
    guide: dict[str, Any],
    config: Settings,
) -> dict[str, Any]:
    """Validate cached output against the production render specification."""
    info = FFprobeService().validate(
        path,
        require_video=True,
        require_audio=True,
        expected_duration=float(guide["duration"]),
        expected_fps=config.video_fps,
    )

    expected_video = {
        "width": config.video_width,
        "height": config.video_height,
        "video_codec": config.video_codec,
        "pixel_format": config.video_pixel_format,
    }

    for field, expected in expected_video.items():
        actual = info.get(field)
        if actual != expected:
            raise MediaToolError(
                f"Cached render has invalid {field}: "
                f"{actual!r}; expected {expected!r}"
            )

    if not info.get("constant_frame_rate"):
        raise MediaToolError("Cached render failed constant-frame-rate validation")

    if not info.get("has_audio") or not info.get("has_video"):
        raise MediaToolError("Cached render is missing required audio/video streams")

    return info


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
            valid_timestamp = (
                isinstance(timestamp, (int, float))
                and not isinstance(timestamp, bool)
                and math.isfinite(timestamp)
                and timestamp >= 0
            )
            if not effect_path.is_file() or not valid_timestamp:
                raise ValueError(
                    f"Optional sound effect is missing or has an invalid timestamp: "
                    f"{effect_path}"
                )

    optional_audio: list[Path] = []

    if config.enable_background_music and music is not None:
        optional_audio.append(music)

    if config.enable_sound_effects:
        optional_audio.extend(path for path, _ in sound_effects or [])

    fingerprint = CheckpointManager().fingerprint(
        [voiceover, *references, *optional_audio],
        {
            "pipeline_version": 4,
            "settings": config,
            "guide_path": str(guide_path.resolve()),
            "sound_effects": [
                (str(path.resolve()), timestamp)
                for path, timestamp in sound_effects or []
            ],
        },
    )

    checkpoint = CheckpointManager()
    previous = checkpoint.load(fingerprint) if resume else None

    if previous and not _checkpoint_is_consistent(previous):
        logger.warning("Discarding inconsistent checkpoint state")
        previous = None

    state: dict[str, Any] = (
        previous.get("data", {}).copy()
        if previous
        else {}
    )

    generated = _unique_paths(
        [Path(p) for p in previous.get("generated_files", [])]
        if previous
        else []
    )

    last_stage = (
        previous.get("last_completed_stage")
        if previous
        else None
    )

    run_temp = Path("temp") / f"{fingerprint[:12]}-{uuid.uuid4().hex[:8]}"

    try:
        if "reference_analyses" not in state:
            logger.info("Analyzing %d reference videos", len(references))
            state["reference_analyses"] = ReferenceVideoService().analyze_many(references)
            checkpoint.save(
                fingerprint,
                "references",
                generated,
                data=state,
            )
            last_stage = "references"

        if "transcript" not in state:
            logger.info("Transcribing original voiceover with local Whisper")
            state["transcript"] = VoiceoverAnalyzer(
                config.whisper_model,
                config.whisper_language,
                config.whisper_device,
                config.whisper_download_root,
                config.whisper_allow_model_download,
            ).analyze(voiceover)

            duration = float(state["transcript"]["duration"])
            if not config.video_min_duration <= duration <= config.video_max_duration:
                raise ValueError(
                    "Voiceover duration is outside configured "
                    "VIDEO_MIN_DURATION / VIDEO_MAX_DURATION"
                )

            checkpoint.save(
                fingerprint,
                "transcription",
                generated,
                data=state,
            )
            last_stage = "transcription"

        if "audio_analysis" not in state:
            logger.info("Analyzing original voiceover audio characteristics")
            state["audio_analysis"] = AudioAnalysisService().analyze(
                voiceover,
                state["transcript"],
            )
            checkpoint.save(
                fingerprint,
                "audio_analysis",
                generated,
                data=state,
            )
            last_stage = "audio_analysis"

        original_audio_duration = float(
            FFprobeService().probe(voiceover)["duration"]
        )
        transcript_duration = float(state["transcript"]["duration"])
        if abs(transcript_duration - original_audio_duration) > 0.001:
            logger.warning(
                "Overriding transcript duration %.6f with original voiceover duration %.6f",
                transcript_duration,
                original_audio_duration,
            )
        state["transcript"] = dict(state["transcript"])
        state["transcript"]["duration"] = original_audio_duration

        # Cached guide state is only accepted when its persisted artifact matches.
        cached_duration = (
            state.get("guide", {}).get("duration")
            if isinstance(state.get("guide"), dict)
            else None
        )
        if "guide" in state and (
            not _validate_cached_guide(
                state["guide"],
                guide_path,
            )
            or not isinstance(cached_duration, (int, float))
            or abs(float(cached_duration) - original_audio_duration) > 0.001
        ):
            logger.warning("Invalid cached guide detected; regenerating guide")
            for key in (
                "style",
                "scenes",
                "caption_scenes",
                "guide",
            ):
                state.pop(key, None)

            last_stage = "audio_analysis"

        if "guide" not in state:
            state["style"] = extract_style(state["reference_analyses"])

            state["scenes"] = direct(
                state["transcript"],
                state["style"],
                enabled_features(config),
                audio_analysis=state["audio_analysis"],
            )

            transliteration = TransliterationService()
            caption_transcript = transliteration.prepare(
                state["transcript"],
                config,
            )

            state["caption_scenes"] = transliteration.apply_to_scenes(
                caption_transcript,
                state["scenes"],
            )

            state["guide"] = generate_guide(
                caption_transcript,
                state["style"],
                state["caption_scenes"],
                guide_path,
                audio_analysis=state["audio_analysis"],
                audio_duration=original_audio_duration,
            )

            generated = _unique_paths([*generated, guide_path])

            checkpoint.save(
                fingerprint,
                "guide",
                generated,
                data=state,
            )
            last_stage = "guide"

        guide = state["guide"]

        if not _validate_cached_guide(guide, guide_path):
            raise ValueError(
                "Generated editing guide failed consistency validation"
            )

        logger.info("Editing guide saved to %s", guide_path)

        if not render:
            return guide

        final_path = config.output_directory / f"{voiceover.stem}.{config.output_format}"

        if last_stage == "rendered" and final_path.is_file():
            try:
                _validate_cached_render(
                    final_path,
                    guide,
                    config,
                )
                logger.info("Reusing validated cached render: %s", final_path)
                return guide

            except (MediaToolError, ValueError, OSError, KeyError, TypeError) as exc:
                logger.warning(
                    "Ignoring invalid cached render and rerendering: %s",
                    exc,
                )

        # External clips and images are permanently disabled.
        renderer = VideoRenderer(
            hyperframes=HyperFramesService(config.hyperframes_command)
        )

        renderer.render(
            guide,
            voiceover,
            final_path,
            config,
            music=music,
            effects=sound_effects,
            work_directory=run_temp / "render",
        )

        generated = _unique_paths(
            [
                *generated,
                final_path,
                Path("storage/metadata") / f"{final_path.stem}.json",
            ]
        )

        checkpoint.save(
            fingerprint,
            "rendered",
            generated,
            data=state,
        )

        cleanup = CleanupManager()

        try:
            cleanup.remove_processed_voiceover(
                config,
                [run_temp / "render/mix.m4a"],
                run_temp,
            )
        except Exception:
            logger.exception("Processed voiceover cleanup failed")

        try:
            cleanup.cleanup(
                config,
                success=True,
                run_directory=run_temp,
            )
        except Exception:
            logger.exception("Temporary-file cleanup failed")

        return guide

    except Exception as exc:
        try:
            checkpoint.save(
                fingerprint,
                "failed",
                generated,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            logger.exception("Failed to save failure checkpoint")

        logger.exception("Pipeline failed; the last valid checkpoint is retained")
        raise
