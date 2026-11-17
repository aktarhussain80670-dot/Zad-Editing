"""Analyze a voiceover using only a locally installed Whisper model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class VoiceoverAnalyzer:
    def __init__(self, model_name: str = "small", language: str = "auto", device: str = "auto") -> None:
        self.model_name, self.language, self.device = model_name, language, device

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
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Local Whisper is unavailable. Install the 'whisper' project dependencies.") from exc
        model = whisper.load_model(self.model_name, device=None if self.device == "auto" else self.device)
        options: dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if self.language.lower() != "auto":
            options["language"] = self.language
        audio = whisper.load_audio(str(path))
        result = model.transcribe(audio, **options)
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
        }
