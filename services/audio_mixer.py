"""Mix optional audio layers beneath the original, duration-governing narration."""

from __future__ import annotations

import math
import os
import uuid
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
        if not voiceover.is_file() or voiceover.stat().st_size == 0:
            raise ValueError("A readable original voiceover is required")

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Audio duration must be finite and positive")

        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing mixed audio: {destination}"
            )

        settings.validate()

        active_music: Path | None = None
        if settings.enable_background_music and music is not None:
            if not music.is_file() or music.stat().st_size == 0:
                raise FileNotFoundError(f"Background music is missing or empty: {music}")
            active_music = music

        effect_layers: list[tuple[Path, float]] = []

        if settings.enable_sound_effects:
            for effect in effects or []:
                if not isinstance(effect, (tuple, list)) or len(effect) != 2:
                    raise ValueError("Each sound effect must contain a path and timestamp")

                effect_path, start = effect

                try:
                    timestamp = float(start)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Sound-effect timestamps must be numeric") from exc

                if not math.isfinite(timestamp) or timestamp < 0 or timestamp >= duration:
                    raise ValueError(
                        f"Invalid sound-effect timestamp: {start!r}"
                    )

                effect_path = Path(effect_path)
                if not effect_path.is_file() or effect_path.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Sound-effect file is missing or empty: {effect_path}"
                    )

                effect_layers.append((effect_path, timestamp))

        inputs = ["-i", str(voiceover)]

        voice_chain = (
            "[0:a]aresample=48000,aformat=sample_fmts=fltp:"
            "sample_rates=48000:channel_layouts=stereo,"
        )

        if settings.audio_normalization:
            voice_chain += (
                f"loudnorm=I={settings.audio_target_lufs}:"
                f"TP={settings.audio_true_peak}:LRA=11,"
            )

        voice_chain += "volume=1[voice]"
        chains = [voice_chain]

        labels = ["[voice]"]
        mix_weights = [1.0]
        input_index = 1

        if active_music is not None:
            inputs.extend(["-stream_loop", "-1", "-i", str(active_music)])

            chains.append("[voice]asplit=2[mainvoice][duckvoice]")
            labels = ["[mainvoice]"]

            fade = min(settings.audio_fade_seconds, duration / 2)
            fade_chain = ""

            if fade > 0:
                fade_chain = (
                    f",afade=t=in:st=0:d={fade},"
                    f"afade=t=out:st={max(0.0, duration - fade)}:d={fade}"
                )

            chains.extend(
                [
                    f"[{input_index}:a]"
                    f"aresample=48000,"
                    f"aformat=sample_fmts=fltp:sample_rates=48000:"
                    f"channel_layouts=stereo,"
                    f"atrim=duration={duration}"
                    f"{fade_chain}[rawmusic]",
                    "[rawmusic][duckvoice]"
                    "sidechaincompress=threshold=0.03:"
                    "ratio=8:attack=15:release=300[music]",
                ]
            )

            labels.append("[music]")
            mix_weights.append(settings.music_volume)
            input_index += 1

        for effect_index, (effect_path, start) in enumerate(effect_layers):
            delay_ms = round(start * 1000)

            inputs.extend(["-i", str(effect_path)])

            chains.append(
                f"[{input_index}:a]"
                f"aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:"
                f"channel_layouts=stereo,"
                f"adelay={delay_ms}|{delay_ms}"
                f"[fx{effect_index}]"
            )

            labels.append(f"[fx{effect_index}]")
            mix_weights.append(settings.sound_effect_volume)
            input_index += 1

        weight_string = " ".join(str(value) for value in mix_weights)

        chains.append(
            f"{''.join(labels)}"
            f"amix=inputs={len(labels)}:"
            f"duration=first:dropout_transition=2:"
            f"weights='{weight_string}'[mixed]"
        )

        chains.append("[mixed]alimiter=limit=0.95[outa]")

        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary_output = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.part.m4a"
        )

        try:
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
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ]
            )

            if (
                not temporary_output.is_file()
                or temporary_output.stat().st_size == 0
            ):
                raise RuntimeError(
                    "FFmpeg reported success but the mixed audio output is missing"
                )

            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing mixed audio: {destination}"
                )

            os.link(temporary_output, destination)

        finally:
            if temporary_output.exists():
                temporary_output.unlink()

        return destination
