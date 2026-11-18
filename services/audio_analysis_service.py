"""Memory-conscious voiceover analysis using local FFmpeg filters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import median
from typing import Any

from services.ffmpeg_service import FFmpegService


class AudioAnalysisService:
    def __init__(
        self,
        ffmpeg: FFmpegService | None = None,
        sample_interval: float = 0.1,
        silence_threshold: str = "-35dB",
        minimum_silence: float = 0.25,
    ) -> None:
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("Sample interval must be positive and finite")

        if not math.isfinite(minimum_silence) or minimum_silence <= 0:
            raise ValueError("Minimum silence duration must be positive and finite")

        if not re.fullmatch(r"-?\d+(?:\.\d+)?dB", silence_threshold):
            raise ValueError("Silence threshold must be a valid dB value")

        self.ffmpeg = ffmpeg or FFmpegService()
        self.sample_interval = sample_interval
        self.silence_threshold = silence_threshold
        self.minimum_silence = minimum_silence

    @staticmethod
    def _finite(value: str | float | None) -> float | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        return number if math.isfinite(number) else None

    @staticmethod
    def _parse_silence(log: str) -> list[dict[str, Any]]:
        events = re.finditer(
            r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)"
            r"(?:\s*\|\s*silence_duration:\s*(-?\d+(?:\.\d+)?))?",
            log,
        )

        intervals: list[dict[str, Any]] = []
        pending_start: float | None = None

        for match in events:
            event_type = match.group(1)
            timestamp = AudioAnalysisService._finite(match.group(2))

            if timestamp is None or timestamp < 0:
                continue

            if event_type == "start":
                pending_start = timestamp
                continue

            duration = AudioAnalysisService._finite(match.group(3))

            if duration is not None:
                start = timestamp - duration
            elif pending_start is not None:
                start = pending_start
            else:
                continue

            if start < 0 or timestamp < start:
                pending_start = None
                continue

            intervals.append(
                {
                    "start": round(start, 3),
                    "end": round(timestamp, 3),
                    "duration": round(timestamp - start, 3),
                }
            )
            pending_start = None

        if pending_start is not None:
            intervals.append(
                {
                    "start": round(pending_start, 3),
                    "end": round(pending_start, 3),
                    "duration": 0.0,
                    "open_ended": True,
                }
            )

        return intervals

    def _analyze_energy(self, voiceover: Path) -> list[dict[str, float]]:
        samples = max(1, round(44100 * self.sample_interval))

        result = self.ffmpeg.run(
            [
                "-i", str(voiceover),
                "-af",
                (
                    "aresample=44100,"
                    f"asetnsamples=n={samples}:p=1,"
                    "astats=metadata=1:reset=1,"
                    "ametadata=print:key=lavfi.astats.Overall.RMS_level"
                ),
                "-f", "null", "-",
            ]
        )

        pattern = re.compile(
            r"pts_time:(-?\d+(?:\.\d+)?).*?"
            r"lavfi\.astats\.Overall\.RMS_level="
            r"(-?(?:\d+(?:\.\d*)?|\.\d+)|-inf)",
            re.DOTALL,
        )

        energy: list[dict[str, float]] = []

        for match in pattern.finditer(result.stderr):
            timestamp = self._finite(match.group(1))
            level = self._finite(match.group(2))

            if timestamp is None or level is None or timestamp < 0:
                continue

            energy.append(
                {
                    "time": round(timestamp, 3),
                    "rms_db": round(level, 3),
                }
            )

        return energy

    def _analyze_loudness(self, voiceover: Path) -> dict[str, float | None]:
        result = self.ffmpeg.run(
            [
                "-i", str(voiceover),
                "-af", "ebur128=peak=true",
                "-f", "null", "-",
            ]
        )

        log = result.stderr

        patterns = {
            "integrated_lufs": (
                r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+)\s*LUFS"
            ),
            "loudness_range_lu": (
                r"Loudness range:\s*\n\s*LRA:\s*([\d.]+)\s*LU"
            ),
            "true_peak_dbtp": (
                r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+)\s*dBTP"
            ),
        }

        metrics: dict[str, float | None] = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, log, re.IGNORECASE)
            metrics[key] = (
                self._finite(match.group(1)) if match else None
            )

        return metrics

    def _analyze_silence(self, voiceover: Path) -> list[dict[str, Any]]:
        result = self.ffmpeg.run(
            [
                "-i", str(voiceover),
                "-af",
                (
                    f"silencedetect=noise={self.silence_threshold}:"
                    f"d={self.minimum_silence}"
                ),
                "-f", "null", "-",
            ]
        )

        return self._parse_silence(result.stderr)

    @staticmethod
    def _speaking_pace(
        transcript: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not transcript:
            return {
                "words_per_minute": None,
                "active_speech_words_per_minute": None,
                "total_words": 0,
                "active_speech_seconds": 0.0,
                "segments": [],
                "source": "unavailable",
            }

        segments = transcript.get("segments", [])
        duration = AudioAnalysisService._finite(
            transcript.get("duration")
        )

        if not isinstance(segments, list):
            segments = []

        total_words = 0
        active_seconds = 0.0
        pace_segments = []

        for segment in segments:
            if not isinstance(segment, dict):
                continue

            start = AudioAnalysisService._finite(segment.get("start"))
            end = AudioAnalysisService._finite(segment.get("end"))

            if start is None or end is None or start < 0 or end <= start:
                continue

            words = segment.get("words", [])

            if isinstance(words, list) and words:
                count = len(words)
            else:
                count = len(str(segment.get("text", "")).split())

            if count <= 0:
                continue

            segment_duration = end - start
            total_words += count
            active_seconds += segment_duration

            pace_segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "word_count": count,
                    "words_per_minute": round(
                        count * 60 / segment_duration, 2
                    ),
                }
            )

        overall_wpm = (
            round(total_words * 60 / duration, 2)
            if duration is not None and duration > 0
            else None
        )

        active_wpm = (
            round(total_words * 60 / active_seconds, 2)
            if active_seconds > 0
            else None
        )

        return {
            "words_per_minute": overall_wpm,
            "active_speech_words_per_minute": active_wpm,
            "total_words": total_words,
            "active_speech_seconds": round(active_seconds, 3),
            "segments": pace_segments,
            "source": "whisper_word_timestamps",
        }

    @staticmethod
    def _estimate_emphasis(
        energy: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        if len(energy) < 3:
            return []

        levels = [
            item["rms_db"]
            for item in energy
            if math.isfinite(item["rms_db"])
        ]

        if len(levels) < 3:
            return []

        center = median(levels)
        threshold = center + 3.0

        return [
            {
                "time": item["time"],
                "rms_db": item["rms_db"],
                "above_median_db": round(
                    item["rms_db"] - center, 3
                ),
            }
            for item in energy
            if item["rms_db"] >= threshold
        ]

    def analyze(
        self,
        voiceover: Path,
        transcript: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        voiceover = Path(voiceover)

        if not voiceover.is_file():
            raise FileNotFoundError(voiceover)

        if voiceover.stat().st_size <= 0:
            raise ValueError("Voiceover file is empty")

        self.ffmpeg.ensure_available()

        energy = self._analyze_energy(voiceover)
        loudness = self._analyze_loudness(voiceover)
        silence = self._analyze_silence(voiceover)

        return {
            "source": voiceover.name,
            "sample_interval_seconds": self.sample_interval,
            "energy": {
                "sample_count": len(energy),
                "samples": energy,
                "estimated_emphasis": self._estimate_emphasis(energy),
            },
            "loudness": loudness,
            "silence": {
                "threshold": self.silence_threshold,
                "minimum_duration": self.minimum_silence,
                "intervals": silence,
            },
            "speaking_pace": self._speaking_pace(transcript),
            "analysis_notes": [
                "RMS energy is measured after resampling to 44.1 kHz.",
                "Silence intervals are threshold-based estimates, not semantic pauses.",
                "Emphasis is a heuristic based on relative RMS energy, not emotion recognition.",
                "Overall speaking pace uses total narration duration.",
                "Active speech pace uses the sum of valid Whisper segment durations.",
                "True peak is read from the EBU R128 dBTP report when available.",
                "Audio is analyzed locally without external APIs.",
            ],
        }
