"""Render HyperFrames visuals, mix narration, mux H.264/AAC, then validate."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.audio_controller import AudioController
from core.timeline_validator import validate_timeline
from services.ffmpeg_service import FFmpegService
from services.ffprobe_service import FFprobeService
from services.hyperframes_service import HyperFramesService


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
        output: Path,
        settings: Settings,
        music: Path | None = None,
        effects: list[tuple[Path, float]] | None = None,
        work_directory: Path = Path("temp/render"),
    ) -> dict[str, Any]:
        duration = float(guide["duration"])
        settings.validate()
        validate_timeline(guide)

        output.parent.mkdir(parents=True, exist_ok=True)

        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output video: {output}"
            )

        metadata_path = Path("storage/metadata") / f"{output.stem}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if metadata_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output metadata: {metadata_path}"
            )

        source_audio = self.ffprobe.probe(voiceover)
        if not source_audio["has_audio"]:
            raise ValueError("Original voiceover file has no audio stream")

        if abs(source_audio["duration"] - duration) > 0.5:
            raise ValueError(
                "Editing guide duration does not match the original voiceover"
            )

        composition = work_directory / "composition.html"
        visual = work_directory / "visuals.mp4"
        mixed = work_directory / "mix.m4a"

        unique_id = uuid.uuid4().hex
        temporary_output = output.with_name(
            f".{output.stem}.{unique_id}.part.mp4"
        )
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.stem}.{unique_id}.part.json"
        )

        work_directory.mkdir(parents=True, exist_ok=True)

        hyperframes = self.hyperframes or HyperFramesService(
            settings.hyperframes_command
        )

        output_published = False

        try:
            # External footage and images are intentionally disabled.
            hyperframes.render(
                guide,
                {},
                composition,
                visual,
                settings.video_width,
                settings.video_height,
                settings.video_fps,
            )

            self.audio.render_mix(
                voiceover,
                mixed,
                duration,
                settings,
                music,
                effects,
            )

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
                    "-fps_mode",
                    "cfr",
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
                temporary_output,
                require_video=True,
                require_audio=True,
                expected_duration=duration,
                expected_fps=settings.video_fps,
            )

            expected = {
                "width": settings.video_width,
                "height": settings.video_height,
                "pixel_format": settings.video_pixel_format,
                "audio_codec": "aac",
            }

            for key, value in expected.items():
                if result.get(key) != value:
                    raise RuntimeError(
                        f"Rendered output has unexpected {key}: "
                        f"{result.get(key)!r}; expected {value!r}"
                    )

            if abs(float(result["fps"]) - settings.video_fps) > 0.05:
                raise RuntimeError(
                    f"Rendered frame rate {result['fps']} does not match "
                    f"configured {settings.video_fps}"
                )

            if result.get("video_codec") != "h264":
                raise RuntimeError(
                    f"Rendered codec is {result.get('video_codec')!r}, "
                    "expected H.264"
                )

            result.update(
                {
                    "path": str(output),
                    "target_width": settings.video_width,
                    "target_height": settings.video_height,
                    "target_fps": settings.video_fps,
                    "narration_source": str(voiceover),
                    "original_narration_used": True,
                    "audio_reencoded": True,
                    "bit_identical_to_source_audio": False,
                    "external_video_assets_used": False,
                    "external_image_assets_used": False,
                    "rendering_engine": "hyperframes",
                }
            )

            # Prepare metadata completely before publishing either artifact.
            with temporary_metadata.open("x", encoding="utf-8") as metadata_file:
                metadata_file.write(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                metadata_file.flush()
                os.fsync(metadata_file.fileno())

            # Hard links provide atomic no-overwrite publication.
            os.link(temporary_output, output)
            output_published = True

            try:
                os.link(temporary_metadata, metadata_path)
            except Exception:
                # Roll back only the output created by this render.
                output.unlink(missing_ok=True)
                output_published = False
                raise

            return result

        finally:
            temporary_output.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

            # Defensive rollback if publication failed unexpectedly.
            if output_published and not metadata_path.is_file():
                output.unlink(missing_ok=True)
