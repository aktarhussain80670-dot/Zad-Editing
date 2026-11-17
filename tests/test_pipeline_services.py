import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai.voiceover_analyzer import VoiceoverAnalyzer
from config.settings import Settings
from core.input_manager import validate_inputs
from core.pipeline import run_pipeline
from providers.pexels_provider import PexelsProvider
from services.subtitle_service import SubtitleService


def test_local_whisper_transcript_keeps_original_words_and_audio_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    voice = tmp_path / "original.wav"
    voice.write_bytes(b"audio")
    audio = [0.0] * 32000

    class Model:
        def transcribe(self, waveform, **options):
            assert waveform is audio
            assert options["word_timestamps"] is True
            return {
                "text": " original words ",
                "language": "en",
                "segments": [
                    {
                        "start": 0.1,
                        "end": 1.2,
                        "text": " original words ",
                        "words": [{"start": 0.1, "end": 0.5, "word": " original"}],
                    }
                ],
            }

    local_whisper = SimpleNamespace(
        _MODELS={"small": "https://example.invalid/models/small.pt"},
        load_audio=lambda path: audio,
        load_model=lambda name, device, download_root: Model(),
    )
    monkeypatch.setitem(sys.modules, "whisper", local_whisper)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.2.0",
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
    )
    monkeypatch.setattr("ai.voiceover_analyzer.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "ai.voiceover_analyzer.FFprobeService",
        lambda: SimpleNamespace(probe=lambda path: {"has_audio": True, "duration": len(audio) / 16000}),
    )
    monkeypatch.setattr(VoiceoverAnalyzer, "_available_memory_bytes", staticmethod(lambda: 8 * 1024**3))
    transcript = VoiceoverAnalyzer().analyze(voice)
    assert transcript["text"] == "original words"
    assert transcript["duration"] == 2
    assert transcript["segments"][0]["words"][0]["word"] == " original"


def test_sentence_segmentation_uses_word_punctuation_without_rewriting():
    raw = [
        {
            "start": 0,
            "end": 1,
            "text": "First. Second!",
            "words": [
                {"start": 0, "end": 0.4, "word": " First."},
                {"start": 0.5, "end": 1, "word": " Second!"},
            ],
        }
    ]
    sentences = VoiceoverAnalyzer._sentence_segments(raw)
    assert [item["text"] for item in sentences] == ["First.", "Second!"]


def test_pexels_results_are_validated_and_filtered():
    response = {
        "videos": [
            {
                "id": 42,
                "url": "https://pexels.com/video/42",
                "duration": 4,
                "user": {"name": "Creator"},
                "video_files": [{"link": "https://videos.pexels.com/v/clip.mp4", "width": 1920, "height": 1080}],
            },
            {"id": 43, "video_files": [{"link": "http://unsafe.invalid/clip.mp4", "width": 1080, "height": 1920}]},
        ]
    }
    with patch("providers.pexels_provider.get_json", return_value=response) as request:
        results = PexelsProvider("secret", timeout=5).search_videos("forest", orientation="landscape", min_width=1280)
    assert len(results) == 1
    assert results[0]["id"] == 42
    assert results[0]["provider"] == "pexels"
    assert request.call_args.args[1] == {"Authorization": "secret"}


def test_input_validator_rejects_duplicate_and_oversized_paths(tmp_path: Path):
    voice = tmp_path / "speech.wav"
    voice.write_bytes(b"voice")
    refs = []
    for index in range(3):
        path = tmp_path / f"reference-{index}.mp4"
        path.write_bytes(b"video")
        refs.append(path)
    with pytest.raises(ValueError, match="distinct"):
        validate_inputs(voice, [refs[0], refs[0], refs[1]], Settings())
    with pytest.raises(ValueError, match="MAX_INPUT_BYTES"):
        validate_inputs(voice, refs, Settings(max_input_bytes=1))


def test_subtitle_service_exports_word_timestamps(tmp_path: Path):
    transcript = {
        "segments": [
            {
                "start": 0.1,
                "end": 0.8,
                "text": "hello world",
                "words": [{"start": 0.1, "end": 0.3, "word": "hello"}, {"start": 0.4, "end": 0.8, "word": "world"}],
            }
        ]
    }
    target = SubtitleService().write_srt(transcript, tmp_path / "captions.srt")
    result = target.read_text()
    assert "00:00:00,100 --> 00:00:00,300" in result
    assert "hello" in result and "world" in result


def test_pipeline_plan_only_resume_reuses_fingerprinted_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    voice = Path("voice.wav")
    voice.write_bytes(b"voice")
    references = []
    for index in range(3):
        item = Path(f"ref-{index}.mp4")
        item.write_bytes(b"reference")
        references.append(item)
    reference_service = SimpleNamespace(analyze_many=lambda paths: [{"brightness": 0.5} for _ in paths])
    transcript = {
        "source": "voice.wav",
        "duration": 2.0,
        "text": "hello world",
        "segments": [{"start": 0.1, "end": 1.9, "text": "hello world"}],
    }
    voiceover_service = SimpleNamespace(analyze=lambda path: transcript)
    with (
        patch("core.pipeline.ReferenceVideoService", return_value=reference_service),
        patch("core.pipeline.VoiceoverAnalyzer", return_value=voiceover_service) as transcriber,
    ):
        first = run_pipeline(
            voice, references, Settings(video_max_duration=5), guide_path=Path("guide.json"), render=False
        )
        second = run_pipeline(
            voice, references, Settings(video_max_duration=5), guide_path=Path("guide.json"), render=False
        )
    assert first == second
    assert Path("guide.json").is_file()
    assert transcriber.call_count == 1
