"""Generate timed subtitles from original or prepared display transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math


def _timestamp(seconds: float) -> str:
    if not math.isfinite(seconds):
        raise ValueError("Subtitle timestamp must be finite")

    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds_part, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{ms:03}"


class SubtitleService:
    """Create subtitle cues using the prepared display text when available."""

    def cues(
        self,
        transcript: dict[str, Any],
        word_level: bool = True,
    ) -> list[dict[str, Any]]:
        cues: list[dict[str, Any]] = []

        for segment in transcript.get("segments", []):
            words = segment.get("words") or []

            if word_level and words:
                for word in words:
                    text = str(
                        word.get("display_word", word.get("word", ""))
                    ).strip()
                    if not text:
                        continue

                    start = float(word["start"])
                    end = float(word["end"])
                    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                        continue

                    cues.append({"start": start, "end": end, "text": text})
            else:
                text = str(
                    segment.get("display_text", segment.get("text", ""))
                ).strip()
                if not text:
                    continue

                start = float(segment["start"])
                end = float(segment["end"])
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    continue

                cues.append({"start": start, "end": end, "text": text})

        return sorted(cues, key=lambda cue: (cue["start"], cue["end"]))

    def write_srt(
        self,
        transcript: dict[str, Any],
        destination: Path,
        word_level: bool = True,
    ) -> Path:
        cues = self.cues(transcript, word_level)
        content = "\n\n".join(
            f"{index}\n"
            f"{_timestamp(cue['start'])} --> {_timestamp(cue['end'])}\n"
            f"{cue['text']}"
            for index, cue in enumerate(cues, 1)
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return destination
