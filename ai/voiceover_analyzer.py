"""Analyze a voiceover using only a locally installed Whisper model."""

from __future__ import annotations

import os
import re
import shutil
import sys
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
        allow_model_download: bool = True,
    ) -> None:
        self.model_name, self.language, self.device = model_name, language, device
        self.download_root = (download_root or (Path.home() / ".cache" / "whisper")).expanduser()
        self.allow_model_download = allow_model_download

    @staticmethod
    def _available_memory_bytes() -> int | None:
        available = None
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            pass
        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            if limit != "max":
                cgroup_available = max(0, int(limit) - current)
                available = cgroup_available if available is None else min(available, cgroup_available)
        except (OSError, ValueError):
            pass
        return available

    def _resolve_device(self, torch: Any) -> str:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("WHISPER_DEVICE must be auto, cpu, cuda, or mps")
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)
        mps_available = bool(mps_backend())
        if self.device == "cuda" and not cuda_available:
            raise RuntimeError("WHISPER_DEVICE=cuda was selected, but CUDA is unavailable in this PyTorch installation")
        if self.device == "mps" and not mps_available:
            raise RuntimeError("WHISPER_DEVICE=mps was selected, but the PyTorch MPS backend is unavailable")
        if self.device != "auto":
            return self.device
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"

    @staticmethod
    def _check_torch_version(torch: Any) -> None:
        raw_version = str(getattr(torch, "__version__", ""))
        try:
            version = tuple(int(part) for part in raw_version.split("+", 1)[0].split(".")[:2])
        except ValueError as exc:
            raise RuntimeError(f"Could not parse PyTorch version {raw_version!r}") from exc
        if len(version) != 2 or version < (1, 13):
            raise RuntimeError(f"Local Whisper requires PyTorch 1.13 or newer; found {raw_version or 'unknown'}")

    def _check_resources(self, torch: Any, device: str, audio_duration: float) -> None:
        if self.model_name not in self._MINIMUM_MEMORY_GIB:
            raise ValueError(
                f"Unsupported Whisper model {self.model_name!r}. Select one of: {', '.join(self._MINIMUM_MEMORY_GIB)}"
            )
        minimum = self._MINIMUM_MEMORY_GIB[self.model_name] * 1024**3
        if device == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except Exception as exc:
                raise WhisperResourceError("Could not determine free CUDA memory for local Whisper") from exc
            if free_bytes < minimum:
                raise WhisperResourceError(
                    f"Whisper model {self.model_name!r} needs about {self._MINIMUM_MEMORY_GIB[self.model_name]:.1f} GiB free GPU memory; only {free_bytes / 1024**3:.1f} GiB is available. Choose a smaller model or WHISPER_DEVICE=cpu."
                )
            return
        available = self._available_memory_bytes()
        waveform_bytes = int(audio_duration * 16000 * 4 * 1.5)
        required = minimum + waveform_bytes
        if available is not None and available < required:
            raise WhisperResourceError(
                f"Whisper model {self.model_name!r} and {audio_duration:.0f}s of decoded audio need about {required / 1024**3:.1f} GiB free memory; only {available / 1024**3:.1f} GiB is available. Choose a smaller WHISPER_MODEL or shorter input."
            )

    def _model_path(self, whisper: Any) -> Path:
        models = getattr(whisper, "_MODELS", None)
        if not isinstance(models, dict) or self.model_name not in models:
            raise ValueError(
                f"Unsupported Whisper model {self.model_name!r}. Select one of: {', '.join(self._MINIMUM_MEMORY_GIB)}"
            )
        import urllib.parse

        return self.download_root / Path(urllib.parse.urlparse(models[self.model_name]).path).name

    def _check_model_cache(self, whisper: Any) -> Path:
        model_path = self._model_path(whisper)
        if model_path.is_file() and model_path.stat().st_size > 0:
            return model_path
        if not self.allow_model_download:
            raise FileNotFoundError(
                f"Local Whisper model {self.model_name!r} is not cached at {model_path}. Enable WHISPER_ALLOW_MODEL_DOWNLOAD or install that model into WHISPER_DOWNLOAD_ROOT."
            )
        required = self._DOWNLOAD_GIB[self.model_name] * 1024**3
        self.download_root.mkdir(parents=True, exist_ok=True)
        free_disk = shutil.disk_usage(self.download_root).free
        if free_disk < required:
            raise WhisperResourceError(
                f"Whisper model {self.model_name!r} needs about {self._DOWNLOAD_GIB[self.model_name]:.1f} GiB disk space; only {free_disk / 1024**3:.1f} GiB is free under {self.download_root}."
            )
        return model_path

    @staticmethod
    def _sentence_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sentences: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        def flush() -> None:
            if not pending:
                return
            tokens = [str(word.get("word", "")) for word in pending]
            text = ""
            for token in tokens:
                if text and token and not token[0].isspace() and not re.match(r"^[.,!?;:]", token):
                    text += " "
                text += token
            sentences.append(
                {
                    "start": float(pending[0]["start"]),
                    "end": float(pending[-1]["end"]),
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
                is_sentence_end = bool(re.search(r"[.!?][\"']?$", token))
                too_long = float(word["end"]) - float(pending[0]["start"]) >= 8
                if is_sentence_end or too_long:
                    flush()
        flush()
        return sentences

    def analyze(self, path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Voiceover is missing or empty: {path}")
        if sys.version_info < (3, 11):  # noqa: UP036 - retain a clear direct-execution diagnostic
            raise RuntimeError("Local Whisper requires Python 3.11 or newer in this project")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Local Whisper requires FFmpeg on PATH to decode audio; install FFmpeg first")
        media = FFprobeService().probe(path)
        if not media["has_audio"] or media["duration"] <= 0:
            raise ValueError(f"Voiceover has no readable audio stream: {path}")
        try:
            import torch  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "PyTorch is missing or binary-incompatible with this Python/platform. Install a matching CPU/GPU PyTorch build, then install requirements.txt; Whisper runs locally and requires no API key."
            ) from exc
        try:
            import whisper  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Local Whisper is unavailable. Install the openai-whisper package from requirements.txt"
            ) from exc
        if not hasattr(torch, "__version__") or not hasattr(torch, "cuda"):
            raise RuntimeError("The installed PyTorch package is incomplete or incompatible with local Whisper")
        self._check_torch_version(torch)
        device = self._resolve_device(torch)
        self._check_resources(torch, device, float(media["duration"]))
        model_path = self._check_model_cache(whisper)
        try:
            model = whisper.load_model(
                self.model_name,
                device=device,
                download_root=str(self.download_root),
            )
        except MemoryError as exc:
            raise WhisperResourceError(
                f"Not enough memory to load Whisper model {self.model_name!r}; choose a smaller model or another device"
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cannot allocate memory" in message:
                raise WhisperResourceError(
                    f"Insufficient memory while loading Whisper model {self.model_name!r}; choose a smaller model or another device"
                ) from exc
            raise RuntimeError(f"Could not load local Whisper model {self.model_name!r}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not read/download local Whisper model {self.model_name!r} at {model_path}; check disk permissions and model-download network access"
            ) from exc
        options: dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if self.language.lower() != "auto":
            options["language"] = self.language
        try:
            audio = whisper.load_audio(str(path))
            result = model.transcribe(audio, **options)
        except MemoryError as exc:
            raise WhisperResourceError("Insufficient system memory while decoding/transcribing the voiceover") from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cannot allocate memory" in message:
                raise WhisperResourceError(
                    "Whisper ran out of memory during transcription; choose a smaller model"
                ) from exc
            raise RuntimeError(f"Local Whisper transcription failed for {path.name}: {exc}") from exc
        raw_segments = [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment.get("text", "").strip(),
                "words": segment.get("words", []),
            }
            for segment in result.get("segments", [])
        ]
        if not raw_segments:
            raise ValueError("Whisper returned no timed speech segments")
        words_with_times = [word for segment in raw_segments for word in segment.get("words", [])]
        segments = self._sentence_segments(raw_segments)
        duration = len(audio) / 16000
        words = len(words_with_times) or sum(len(re.findall(r"\S+", segment["text"])) for segment in segments)
        return {
            "source": path.name,
            "language": result.get("language", "unknown"),
            "duration": duration,
            "text": result.get("text", "").strip(),
            "segments": segments,
            "words_per_minute": round(words / duration * 60, 2) if duration else 0.0,
            "engine": "local-whisper",
            "model": self.model_name,
            "device": device,
            "model_path": str(model_path),
        }
