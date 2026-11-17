"""Enforce primary narration preservation before optional audio is mixed."""

from pathlib import Path

from config.settings import Settings
from services.audio_mixer import AudioMixer


class AudioController:
    def __init__(self, mixer: AudioMixer | None = None) -> None:
        self.mixer = mixer or AudioMixer()

    def render_mix(
        self,
        voiceover: Path,
        output: Path,
        duration: float,
        settings: Settings,
        music: Path | None = None,
        effects: list[tuple[Path, float]] | None = None,
    ) -> Path:
        if not settings.preserve_original_voiceover:
            raise ValueError("Production output requires PRESERVE_ORIGINAL_VOICEOVER=true")
        return self.mixer.mix(voiceover, output, duration, settings, music, effects)
