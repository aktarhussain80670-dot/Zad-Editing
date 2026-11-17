"""Typed settings for the local video-editing pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean value")
    return normalized in {"true", "1", "yes"}


@dataclass(frozen=True)
class Settings:
    whisper_model: str = "small"
    whisper_language: str = "auto"
    whisper_device: str = "auto"
    pixabay_api_key: str = ""
    pexels_api_key: str = ""
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 60
    video_codec: str = "libx264"
    video_preset: str = "slow"
    video_crf: int = 16
    video_pixel_format: str = "yuv420p"
    video_min_duration: float = 1.0
    video_max_duration: float = 180.0
    enable_subtitles: bool = True
    enable_motion_graphics: bool = True
    enable_transitions: bool = True
    enable_sound_effects: bool = True
    enable_background_music: bool = True
    enable_external_clips: bool = True
    enable_external_images: bool = True
    preserve_original_voiceover: bool = True
    audio_normalization: bool = True
    audio_target_lufs: float = -14.0
    audio_true_peak: float = -1.0
    delete_temp_files: bool = True
    delete_processed_voiceover: bool = False
    preserve_reference_videos: bool = True
    output_format: str = "mp4"
    output_directory: Path = Path("storage/final_video")
    min_references: int = 3
    max_references: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        values: dict[str, object] = {}
        for key in defaults.__dataclass_fields__:
            env_key = key.upper()
            if env_key not in os.environ:
                continue
            current = getattr(defaults, key)
            raw = os.environ[env_key]
            if isinstance(current, bool):
                values[key] = _bool(env_key, current)
            elif isinstance(current, int):
                values[key] = int(raw)
            elif isinstance(current, float):
                values[key] = float(raw)
            elif isinstance(current, Path):
                values[key] = Path(raw)
            else:
                values[key] = raw
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.video_width < 16 or self.video_height < 16 or self.video_fps <= 0:
            raise ValueError("Video dimensions and FPS must be positive and usable")
        if not 0 <= self.video_crf <= 51:
            raise ValueError("VIDEO_CRF must be between 0 and 51")
        if self.video_min_duration <= 0 or self.video_max_duration < self.video_min_duration:
            raise ValueError("Video duration limits are invalid")
        if self.min_references < 1 or self.max_references < self.min_references:
            raise ValueError("Reference video count limits are invalid")

    @property
    def use_sound_effects(self) -> bool:
        return self.enable_sound_effects and bool(self.pixabay_api_key)

    @property
    def use_background_music(self) -> bool:
        return self.enable_background_music and bool(self.pixabay_api_key)

    @property
    def use_external_clips(self) -> bool:
        return self.enable_external_clips and bool(self.pexels_api_key)
