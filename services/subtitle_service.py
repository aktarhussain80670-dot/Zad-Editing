"""Write timed SRT and obtain word-level caption cues when Whisper provides them."""

from pathlib import Path
from typing import Any


def _timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


class SubtitleService:
    def cues(self, transcript: dict[str, Any], word_level: bool = True) -> list[dict[str, Any]]:
        cues = []
        for segment in transcript.get("segments", []):
            words = segment.get("words") or []
            if word_level and words:
                cues.extend(
                    {"start": float(w["start"]), "end": float(w["end"]), "text": str(w.get("word", "")).strip()}
                    for w in words
                    if str(w.get("word", "")).strip()
                )
            elif str(segment.get("text", "")).strip():
                cues.append(
                    {
                        "start": float(segment["start"]),
                        "end": float(segment["end"]),
                        "text": str(segment["text"]).strip(),
                    }
                )
        return cues

    def write_srt(self, transcript: dict[str, Any], destination: Path, word_level: bool = True) -> Path:
        cues = self.cues(transcript, word_level)
        text = "\n\n".join(
            f"{i}\n{_timestamp(c['start'])} --> {_timestamp(c['end'])}\n{c['text']}" for i, c in enumerate(cues, 1)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + ("\n" if text else ""), encoding="utf-8")
        return destination
