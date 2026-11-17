"""Render HyperFrames visuals, mix narration, mux H.264/AAC, then decode-validate."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.audio_controller import AudioController
from core.timeline_validator import validate_timeline
from services.ffmpeg_service import FFmpegService, MediaToolError
from services.ffprobe_service import FFprobeService
from services.hyperframes_service import HyperFramesService

logger = logging.getLogger(__name__)


class VideoRenderer:
    def __init__(
        self,
        hyperframes: HyperFramesService | None = None,
        ffmpeg: FFmpegService | None = None,
        ffprobe: FFprobeService | None = None,
        audio: AudioController | None = None,
    ) -> None:
        self.hyperframes = hyperframes
        self.ffmpeg = ffmpeg or FFmpegService()
        self.ffprobe = ffprobe or FFprobeService()
        self.audio = audio or AudioController()

    def render(
        self,
        guide: dict[str, Any],
        voiceover: Path,
        assets: dict[int, Path],
        output: Path,
        settings: Settings,
        music: Path | None = None,
        effects: list[tuple[Path, float]] | None = None,
        work_directory: Path = Path("temp/render"),
    ) -> dict[str, Any]:
        duration = float(guide["duration"])
        settings.validate()
        validate_timeline(guide)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output video: {output}")
        metadata_path = Path("storage/metadata") / f"{output.stem}.json"
        if metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output metadata: {metadata_path}")
        source_audio = self.ffprobe.probe(voiceover)
        if not source_audio["has_audio"]:
            raise ValueError("Original voiceover file has no audio stream")
        if abs(source_audio["duration"] - duration) > 0.5:
            raise ValueError("Editing guide duration does not match the original voiceover")
        composition = work_directory / "composition.html"
        visual = work_directory / "visuals.mp4"
        mixed = work_directory / "mix.m4a"
        temporary_output = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.part.mp4")
        work_directory.mkdir(parents=True, exist_ok=True)
        scene_durations = {
            int(scene["scene_id"]): float(scene["end"]) - float(scene["start"]) for scene in guide["scenes"]
        }
        prepared_assets: dict[int, Path] = {}
        for scene_id, asset in assets.items():
            if scene_id not in scene_durations or not asset.is_file():
                continue
            try:
                prepared_assets[scene_id] = self.ffmpeg.prepare_scene(
                    asset,
                    work_directory / "scenes" / f"scene-{scene_id}.mp4",
                    scene_durations[scene_id],
                    settings.video_width,
                    settings.video_height,
                    settings.video_fps,
                )
            except (MediaToolError, ValueError, OSError) as exc:
                logger.warning("Optional footage rejected for scene %s: %s", scene_id, exc)
        hyperframes = self.hyperframes or HyperFramesService(settings.hyperframes_command)
        hyperframes.render(
            guide, prepared_assets, composition, visual, settings.video_width, settings.video_height, settings.video_fps
        )
        self.audio.render_mix(voiceover, mixed, duration, settings, music, effects)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.run(
            [
                "-y",
                "-i",
                str(visual),
                "-i",
                str(mixed),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                str(duration),
                "-r",
                str(settings.video_fps),
                "-c:v",
                settings.video_codec,
                "-preset",
                settings.video_preset,
                "-crf",
                str(settings.video_crf),
                "-pix_fmt",
                settings.video_pixel_format,
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
        )
        result = self.ffprobe.validate(
            temporary_output, require_video=True, require_audio=True, expected_duration=duration
        )
        expected = {
            "width": settings.video_width,
            "height": settings.video_height,
            "pixel_format": settings.video_pixel_format,
            "audio_codec": "aac",
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise RuntimeError(f"Rendered output has unexpected {key}: {result.get(key)!r}; expected {value!r}")
        if abs(float(result["fps"]) - settings.video_fps) > 0.05:
            raise RuntimeError(f"Rendered frame rate {result['fps']} does not match configured {settings.video_fps}")
        if result.get("video_codec") != "h264":
            raise RuntimeError(f"Rendered codec is {result.get('video_codec')!r}, expected H.264")
        os.link(temporary_output, output)
        temporary_output.unlink()
        result.update(
            {
                "path": str(output),
                "target_width": settings.video_width,
                "target_height": settings.video_height,
                "target_fps": settings.video_fps,
                "voiceover_preserved": True,
            }
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(result, indent=2) + "\n")
        return result
