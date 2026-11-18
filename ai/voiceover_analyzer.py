"""Analyze a voiceover using only a locally installed Whisper model."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import urllib.parse
from pathlib import Path
from typing import Any, ClassVar

from services.ffprobe_service import FFprobeService


class WhisperResourceError(RuntimeError):
    """Local Whisper cannot run with the available memory or disk resources."""


class VoiceoverAnalyzer:
    _MINIMUM_MEMORY_GIB: ClassVar[dict[str, float]] = {
        "tiny": 0.8,
        "tiny.en": 0.8,
        "base": 1.0,
        "base.en": 1.0,
        "small": 1.7,
        "small.en": 1.7,
        "medium": 4.0,
        "medium.en": 4.0,
        "large": 8.0,
        "large-v1": 8.0,
        "large-v2": 8.0,
        "large-v3": 8.0,
        "large-v3-turbo": 5.0,
        "turbo": 5.0,
    }

    _DOWNLOAD_GIB: ClassVar[dict[str, float]] = {
        "tiny": 0.08,
        "tiny.en": 0.08,
        "base": 0.15,
        "base.en": 0.15,
        "small": 0.5,
        "small.en": 0.5,
        "medium": 1.6,
        "medium.en": 1.6,
        "large": 3.1,
        "large-v1": 3.1,
        "large-v2": 3.1,
        "large-v3": 3.1,
        "large-v3-turbo": 1.7,
        "turbo": 1.7,
    }

    def __init__(
        self,
        model_name: str = "small",
        language: str = "auto",
        device: str = "auto",
        download_root: Path | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.download_root = (
            download_root or Path.home() / ".cache" / "whisper"
        ).expanduser()
        self.allow_model_download = allow_model_download

    @staticmethod
    def _available_memory_bytes() -> int | None:
        available = None

        try:
            available = (
                os.sysconf("SC_AVPHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE")
            )
        except (AttributeError, OSError, ValueError):
            pass

        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            current = int(
                Path("/sys/fs/cgroup/memory.current").read_text().strip()
            )

            if limit != "max":
                cgroup_available = max(0, int(limit) - current)
                available = (
                    cgroup_available
                    if available is None
                    else min(available, cgroup_available)
                )
        except (OSError, ValueError):
            pass

        return available

    def _resolve_device(self, torch: Any) -> str:
        if self.device not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(
            r"cuda:\d+", self.device
        ):
            raise ValueError(
                "WHISPER_DEVICE must be auto, cpu, cuda, cuda:N, or mps"
            )

        cuda_available = bool(torch.cuda.is_available())

        mps_backend = getattr(
            getattr(torch.backends, "mps", None),
            "is_available",
            lambda: False,
        )
        mps_available = bool(mps_backend())

        if self.device.startswith("cuda"):
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was selected, but CUDA is unavailable"
                )

            if self.device.startswith("cuda:"):
                index = int(self.device.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    raise ValueError(
                        f"CUDA device index {index} is unavailable"
                    )

            return self.device

        if self.device == "mps":
            if not mps_available:
                raise RuntimeError(
                    "MPS was selected, but the PyTorch MPS backend is unavailable"
                )
            return "mps"

        if self.device == "cpu":
            return "cpu"

        return "cuda" if cuda_available else "mps" if mps_available else "cpu"

    @staticmethod
    def _check_torch_version(torch: Any) -> None:
        raw_version = str(getattr(torch, "__version__", ""))

        try:
            version = tuple(
                int(part)
                for part in raw_version.split("+", 1)[0].split(".")[:2]
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not parse PyTorch version {raw_version!r}"
            ) from exc

        if len(version) != 2 or version < (1, 13):
            raise RuntimeError(
                f"Local Whisper requires PyTorch 1.13 or newer; "
                f"found {raw_version or 'unknown'}"
            )

    def _check_resources(
        self,
        torch: Any,
        device: str,
        audio_duration: float,
    ) -> None:
        if self.model_name not in self._MINIMUM_MEMORY_GIB:
            raise ValueError(
                f"Unsupported Whisper model {self.model_name!r}. "
                f"Select one of: {', '.join(self._MINIMUM_MEMORY_GIB)}"
            )

        minimum = self._MINIMUM_MEMORY_GIB[self.model_name] * 1024**3

        if device.startswith("cuda"):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
            except Exception as exc:
                raise WhisperResourceError(
                    "Could not determine free CUDA memory"
                ) from exc

            if free_bytes < minimum:
                raise WhisperResourceError(
                    f"Whisper model {self.model_name!r} needs about "
                    f"{self._MINIMUM_MEMORY_GIB[self.model_name]:.1f} GiB "
                    f"free GPU memory; only {free_bytes / 1024**3:.1f} GiB "
                    "is available."
                )
            return

        available = self._available_memory_bytes()
        waveform_bytes = int(audio_duration * 16000 * 4 * 1.5)
        required = minimum + waveform_bytes

        if available is not None and available < required:
            raise WhisperResourceError(
                f"Whisper model {self.model_name!r} and decoded audio "
                f"need about {required / 1024**3:.1f} GiB free memory; "
                f"only {available / 1024**3:.1f} GiB is available. "
                "Choose a smaller model or shorter input."
            )

    def _model_path(self, whisper: Any) -> Path:
        models = getattr(whisper, "_MODELS", None)

        if not isinstance(models, dict) or self.model_name not in models:
            raise ValueError(
                f"Unsupported Whisper model {self.model_name!r}"
            )

        model_url = models[self.model_name]
        return (
            self.download_root
            / Path(urllib.parse.urlparse(model_url).path).name
        )

    def _check_model_cache(self, whisper: Any) -> Path:
        model_path = self._model_path(whisper)

        if model_path.is_file() and model_path.stat().st_size > 0:
            return model_path

        if not self.allow_model_download:
            raise FileNotFoundError(
                f"Local Whisper model {self.model_name!r} is not cached "
                f"at {model_path}. Enable WHISPER_ALLOW_MODEL_DOWNLOAD "
                "explicitly to permit downloading."
            )

        required = self._DOWNLOAD_GIB[self.model_name] * 1024**3
        self.download_root.mkdir(parents=True, exist_ok=True)

        free_disk = shutil.disk_usage(self.download_root).free

        if free_disk < required:
            raise WhisperResourceError(
                f"Whisper model {self.model_name!r} needs about "
                f"{self._DOWNLOAD_GIB[self.model_name]:.1f} GiB disk space; "
                f"only {free_disk / 1024**3:.1f} GiB is available."
            )

        return model_path

    @staticmethod
    def _sentence_segments(
        raw_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        sentences: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        def flush() -> None:
            if not pending:
                return

            tokens = [str(word.get("word", "")) for word in pending]
            text = ""

            for token in tokens:
                if (
                    text
                    and token
                    and not token[0].isspace()
                    and not re.match(r"^[.,!?;:]", token)
                ):
                    text += " "
                text += token

            sentences.append(
                {
                    "start": pending[0]["start"],
                    "end": pending[-1]["end"],
                    "text": text.strip(),
                    "words": list(pending),
                }
            )
            pending.clear()

        for segment in raw_segments:
            word_list = segment.get("words") or []

            if not word_list:
                flush()
                sentences.append(segment)
                continue

            for word in word_list:
                pending.append(word)

                token = str(word.get("word", "")).strip()
                word_duration = word["end"] - pending[0]["start"]

                if (
                    re.search(r"[.!?][\"']?$", token)
                    or word_duration >= 8.0
                ):
                    flush()

        flush()
        return sentences

    @staticmethod
    def _validate_timestamps(
        result: dict[str, Any],
        duration: float,
    ) -> list[dict[str, Any]]:
        raw_segments = result.get("segments", [])

        if not isinstance(raw_segments, list):
            raise ValueError("Whisper returned invalid segment data")

        validated: list[dict[str, Any]] = []
        previous_segment_end = 0.0

        for index, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                continue

            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Whisper returned invalid timestamps in segment {index}"
                ) from exc

            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
                or end > duration + 0.5
                or start < previous_segment_end - 0.5
            ):
                raise ValueError(
                    f"Whisper returned invalid or overlapping segment "
                    f"timestamps in segment {index}: "
                    f"start={start!r}, end={end!r}, "
                    f"previous_segment_end={previous_segment_end!r}, "
                    f"duration={duration!r}"
                )

            end = min(end, duration)
            words: list[dict[str, Any]] = []

            raw_words = segment.get("words") or []

            if not isinstance(raw_words, list):
                raw_words = []

            previous_word_end = start

            for word_index, word in enumerate(raw_words):
                if not isinstance(word, dict):
                    continue

                try:
                    word_start = float(word["start"])
                    word_end = float(word["end"])
                except (KeyError, TypeError, ValueError):
                    continue

                if (
                    not math.isfinite(word_start)
                    or not math.isfinite(word_end)
                    or word_start < start - 0.5
                    or word_end > end + 0.5
                    or word_end <= word_start
                    or word_start < previous_word_end - 0.05
                ):
                    continue

                word_start = max(start, word_start)
                word_end = min(end, word_end)

                if word_end <= word_start:
                    continue

                words.append(
                    {
                        "word": str(word.get("word", "")),
                        "start": word_start,
                        "end": word_end,
                        "probability": word.get("probability"),
                    }
                )
                previous_word_end = word_end

            validated.append(
                {
                    "start": start,
                    "end": end,
                    "text": str(segment.get("text", "")).strip(),
                    "words": words,
                }
            )

            previous_segment_end = end

        if not validated:
            raise ValueError("Whisper returned no valid timed speech segments")

        return validated

    def analyze(self, path: Path) -> dict[str, Any]:
        path = Path(path)

        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Voiceover is missing or empty: {path}")

        if sys.version_info < (3, 11):
            raise RuntimeError(
                "Local Whisper requires Python 3.11 or newer"
            )

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "Local Whisper requires FFmpeg on PATH"
            )

        media = FFprobeService().probe(path)

        if not media["has_audio"] or media["duration"] <= 0:
            raise ValueError(
                f"Voiceover has no readable audio stream: {path}"
            )

        try:
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "PyTorch is missing or incompatible with this platform."
            ) from exc

        try:
            import whisper
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Local Whisper is unavailable. Check openai-whisper installation."
            ) from exc

        if not hasattr(torch, "__version__") or not hasattr(torch, "cuda"):
            raise RuntimeError(
                "The installed PyTorch package is incomplete or incompatible"
            )

        self._check_torch_version(torch)

        device = self._resolve_device(torch)
        duration = float(media["duration"])

        self._check_resources(torch, device, duration)
        model_path = self._check_model_cache(whisper)

        try:
            model = whisper.load_model(
                self.model_name,
                device=device,
                download_root=str(self.download_root),
            )
        except MemoryError as exc:
            raise WhisperResourceError(
                "Not enough memory to load the selected Whisper model"
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()

            if "out of memory" in message or "cannot allocate memory" in message:
                raise WhisperResourceError(
                    "Insufficient memory while loading Whisper model"
                ) from exc

            raise RuntimeError(
                f"Could not load local Whisper model: {exc}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not read/download Whisper model at {model_path}"
            ) from exc

        options: dict[str, Any] = {
            "word_timestamps": True,
            "verbose": False,
        }

        if self.language.lower() != "auto":
            options["language"] = self.language

        try:
            audio = whisper.load_audio(str(path))

            if len(audio) == 0:
                raise ValueError("Decoded voiceover audio is empty")

            result = model.transcribe(audio, **options)

        except MemoryError as exc:
            raise WhisperResourceError(
                "Insufficient memory while transcribing the voiceover"
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()

            if "out of memory" in message or "cannot allocate memory" in message:
                raise WhisperResourceError(
                    "Whisper ran out of memory during transcription"
                ) from exc

            raise RuntimeError(
                f"Local Whisper transcription failed: {exc}"
            ) from exc

        raw_segments = self._validate_timestamps(result, duration)
        segments = self._sentence_segments(raw_segments)

        words_with_times = [
            word
            for segment in raw_segments
            for word in segment.get("words", [])
        ]

        word_count = (
            len(words_with_times)
            or sum(
                len(re.findall(r"\S+", segment["text"]))
                for segment in segments
            )
        )

        return {
            "source": path.name,
            "language": result.get("language", "unknown"),
            "duration": duration,
            "text": str(result.get("text", "")).strip(),
            "segments": segments,
            "words_per_minute": round(
                word_count / duration * 60, 2
            ),
            "engine": "local-whisper",
            "model": self.model_name,
            "device": device,
        }
