"""Mix optional audio layers beneath the original, duration-governing narration."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from services.ffmpeg_service import FFmpegService


class AudioMixer:
    def __init__(self, ffmpeg: FFmpegService | None = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegService()

    def mix(
        self,
        voiceover: Path,
        destination: Path,
        duration: float,
        settings: Settings,
        music: Path | None = None,
        effects: list[tuple[Path, float]] | None = None,
    ) -> Path:
        if not voiceover.is_file() or duration <= 0:
            raise ValueError("A readable original voiceover and positive duration are required")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing mixed audio: {destination}")
        music = music if settings.enable_background_music and music and music.is_file() else None
        effect_layers = [
            (p, at) for p, at in (effects or []) if settings.enable_sound_effects and p.is_file() and 0 <= at < duration
        ]
        inputs = ["-i", str(voiceover)]
        chains = [
            "[0:a]"
            + (
                f"loudnorm=I={settings.audio_target_lufs}:TP={settings.audio_true_peak}:LRA=11,"
                if settings.audio_normalization
                else ""
            )
            + "volume=1[voice]"
        ]
        labels = ["[voice]"]
        mix_weights = [1.0]
        index = 1
        if music:
            inputs.extend(["-stream_loop", "-1", "-i", str(music)])
            chains.append("[voice]asplit=2[mainvoice][duckvoice]")
            labels = ["[mainvoice]"]
            fade = min(settings.audio_fade_seconds, duration / 2)
            fade_chain = f",afade=t=in:st=0:d={fade},afade=t=out:st={max(0, duration - fade)}:d={fade}" if fade else ""
            chains.extend(
                [
                    f"[{index}:a]atrim=duration={duration}{fade_chain}[rawmusic]",
                    "[rawmusic][duckvoice]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=300[music]",
                ]
            )
            labels.append("[music]")
            mix_weights.append(settings.music_volume)
            index += 1
        for fx_index, (path, start) in enumerate(effect_layers):
            delay_ms = int(start * 1000)
            inputs.extend(["-i", str(path)])
            chains.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[fx{fx_index}]")
            labels.append(f"[fx{fx_index}]")
            mix_weights.append(settings.sound_effect_volume)
            index += 1
        weight_string = " ".join(map(str, mix_weights))
        chains.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:dropout_transition=2:weights='{weight_string}'[outa]"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.run(
            [
                "-y",
                *inputs,
                "-filter_complex",
                ";".join(chains),
                "-map",
                "[outa]",
                "-t",
                str(duration),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(destination),
            ]
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("FFmpeg reported success but the audio output is missing")
        return destination
