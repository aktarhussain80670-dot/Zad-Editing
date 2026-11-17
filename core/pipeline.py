"""Run local analysis through validated editing-guide generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai.editing_director import direct
from ai.reference_analyzer import ReferenceAnalyzer
from ai.style_extractor import extract_style
from ai.timeline_generator import generate_guide
from ai.voiceover_analyzer import VoiceoverAnalyzer
from config.settings import Settings
from core.feature_controller import enabled_features
from core.input_manager import validate_inputs

logger = logging.getLogger(__name__)


def run_pipeline(voiceover: Path, references: list[Path], settings: Settings | None = None, guide_path: Path = Path("storage/editing_guides/editing_guide.json")) -> dict[str, Any]:
    config = settings or Settings.from_env()
    config.validate()
    validate_inputs(voiceover, references, config)
    logger.info("Analyzing %d reference videos", len(references))
    reference_analyses = [ReferenceAnalyzer().analyze(path) for path in references]
    style = extract_style(reference_analyses)
    logger.info("Transcribing original voiceover with local Whisper")
    transcript = VoiceoverAnalyzer(config.whisper_model, config.whisper_language, config.whisper_device).analyze(voiceover)
    scenes = direct(transcript, style, enabled_features(config))
    guide = generate_guide(transcript, style, scenes, guide_path)
    logger.info("Editing guide saved to %s", guide_path)
    return guide
