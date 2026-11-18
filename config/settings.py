"""Production configuration for the local motion-graphics video editor."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load project .env values without overriding existing environment variables."""
    env_file = Path.cwd() / ".env"

    if not env_file.is_file():
        return

    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            raise ValueError(f"Invalid .env entry at line {line_number}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid .env key at line {line_number}")

        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or not value.endswith(quote):
                raise ValueError(f"Unclosed quote in .env at line {line_number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()

        os.environ.setdefault(key, value)


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
    whisper_download_root: Path = Path.home() / ".cache" / "whisper"
    whisper_allow_model_download: bool = False

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
    hindi_caption_romanize: bool = True
    caption_preserve_other_scripts: bool = True
    enable_motion_graphics: bool = True
    enable_transitions: bool = True
    enable_sound_effects: bool = False
    enable_background_music: bool = False
    enable_external_clips: bool = False
    enable_external_images: bool = False

    preserve_original_voiceover: bool = True
    audio_normalization: bool = True
    audio_target_lufs: float = -14.0
    audio_true_peak: float = -1.0
    audio_fade_seconds: float = 0.35
    music_volume: float = 0.12
    sound_effect_volume: float = 0.25

    media_timeout_seconds: int = 30
    max_download_bytes: int = 250_000_000
    max_input_bytes: int = 2_000_000_000

    hyperframes_command: str = "hyperframes"

    delete_temp_files: bool = True
    delete_processed_voiceover: bool = False
    preserve_reference_videos: bool = True

    output_format: str = "mp4"
    output_directory: Path = Path("storage/final_video")

    min_references: int = 1
    max_references: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()

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
                values[key] = Path(raw).expanduser()
            else:
                values[key] = raw.strip()

        # Production output profile is fixed.
        fixed_values = {
            "video_width": 1080,
            "video_height": 1920,
            "video_fps": 60,
            "video_codec": "libx264",
            "video_pixel_format": "yuv420p",
        }
        for key, expected in fixed_values.items():
            if key in values and values[key] != expected:
                raise ValueError(
                    f"{key.upper()} is fixed to {expected!r} in production"
                )

        # These features cannot be enabled/disabled through .env.
        values["enable_external_clips"] = False
        values["enable_external_images"] = False
        values["enable_motion_graphics"] = True

        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.video_width != 1080 or self.video_height != 1920:
            raise ValueError("Production output resolution must be 1080x1920")

        if self.video_width % 2 or self.video_height % 2:
            raise ValueError("Video dimensions must be even")

        if self.video_fps != 60:
            raise ValueError("Production output frame rate must be 60 FPS")

        if not 0 <= self.video_crf <= 51:
            raise ValueError("VIDEO_CRF must be between 0 and 51")

        if (
            not math.isfinite(self.video_min_duration)
            or not math.isfinite(self.video_max_duration)
            or self.video_min_duration <= 0
            or self.video_max_duration < self.video_min_duration
        ):
            raise ValueError("Video duration limits are invalid")

        if self.output_format.lower() != "mp4":
            raise ValueError("Only MP4 is supported by the final renderer")

        if self.video_codec != "libx264":
            raise ValueError("Production video codec must be libx264 (H.264)")

        if self.video_pixel_format != "yuv420p":
            raise ValueError("Production pixel format must be yuv420p")

        if not self.video_preset:
            raise ValueError("VIDEO_PRESET cannot be empty")

        if not self.enable_motion_graphics:
            raise ValueError("Motion graphics are mandatory in production")

        if self.enable_external_clips or self.enable_external_images:
            raise ValueError("External clips and images are disabled in production")

        if self.min_references != 1 or self.max_references != 4:
            raise ValueError("The editor requires between 1 and 4 reference videos")

        if self.whisper_device not in {"auto", "cpu", "cuda", "mps"}:
            if not self.whisper_device.startswith("cuda:"):
                raise ValueError("WHISPER_DEVICE must be auto, cpu, cuda, mps, or cuda:N")

        if not self.whisper_model.strip():
            raise ValueError("WHISPER_MODEL cannot be empty")

        if not self.hyperframes_command.strip():
            raise ValueError("HYPERFRAMES_COMMAND cannot be empty")

        numeric_values = {
            "AUDIO_TARGET_LUFS": self.audio_target_lufs,
            "AUDIO_TRUE_PEAK": self.audio_true_peak,
            "AUDIO_FADE_SECONDS": self.audio_fade_seconds,
            "MUSIC_VOLUME": self.music_volume,
            "SOUND_EFFECT_VOLUME": self.sound_effect_volume,
        }

        for name, value in numeric_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")

        if not -70 <= self.audio_target_lufs <= 0:
            raise ValueError("AUDIO_TARGET_LUFS must be between -70 and 0")

        if not -9 <= self.audio_true_peak <= 0:
            raise ValueError("AUDIO_TRUE_PEAK must be between -9 and 0")

        if self.audio_fade_seconds < 0:
            raise ValueError("AUDIO_FADE_SECONDS cannot be negative")

        if not 0 <= self.music_volume <= 1:
            raise ValueError("MUSIC_VOLUME must be between 0 and 1")

        if not 0 <= self.sound_effect_volume <= 1:
            raise ValueError("SOUND_EFFECT_VOLUME must be between 0 and 1")

        if self.media_timeout_seconds < 1:
            raise ValueError("MEDIA_TIMEOUT_SECONDS must be positive")

        if self.max_download_bytes < 1 or self.max_input_bytes < 1:
            raise ValueError("File size limits must be positive")

        if not self.whisper_language.strip():
            raise ValueError("WHISPER_LANGUAGE cannot be empty")

    @property
    def use_sound_effects(self) -> bool:
        return self.enable_sound_effects

    @property
    def use_background_music(self) -> bool:
        return self.enable_background_music

    @property
    def use_external_clips(self) -> bool:
        return self.enable_external_clips and bool(self.pexels_api_key)

    @property
    def use_external_images(self) -> bool:
        return self.enable_external_images and bool(self.pexels_api_key)
