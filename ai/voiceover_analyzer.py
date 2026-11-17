"""Analyze a voiceover using only a locally installed Whisper model."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VoiceoverAnalyzer:
    def __init__(self, model_name: str = "small", language: str = "auto", device: str = "auto") -> None:
        self.model_name, self.language, self.device = model_name, language, device

    def analyze(self, path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Voiceover is missing or empty: {path}")
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Local Whisper is unavailable. Install the 'whisper' project dependencies.") from exc
        model = whisper.load_model(self.model_name, device=None if self.device == "auto" else self.device)
        options: dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if self.language.lower() != "auto":
            options["language"] = self.language
        result = model.transcribe(str(path), **options)
        segments = [{"start": float(segment["start"]), "end": float(segment["end"]), "text": segment.get("text", "").strip(), "words": segment.get("words", [])} for segment in result.get("segments", [])]
        if not segments:
            raise ValueError("Whisper returned no timed speech segments")
        duration = max(float(segment["end"]) for segment in segments)
        words = sum(len(segment.get("words", [])) for segment in segments)
        return {"source": path.name, "language": result.get("language", "unknown"), "duration": duration, "text": result.get("text", "").strip(), "segments": segments, "words_per_minute": round(words / duration * 60, 2) if duration else 0.0, "engine": "local-whisper"}
