"""Reference-video analysis facade used by the pipeline."""

from pathlib import Path
from typing import Any

from ai.reference_analyzer import ReferenceAnalyzer


class ReferenceVideoService:
    def __init__(self, sample_seconds: float = 2.0, max_dimension: int = 640) -> None:
        self.analyzer = ReferenceAnalyzer(sample_seconds, max_dimension)

    def analyze_many(self, paths: list[Path]) -> list[dict[str, Any]]:
        if not paths:
            raise ValueError("At least one reference video is required")
        return [self.analyzer.analyze(path) for path in paths]
