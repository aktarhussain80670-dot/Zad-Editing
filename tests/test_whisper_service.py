import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.voiceover_analyzer import VoiceoverAnalyzer, WhisperResourceError


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio: list[float] | None = None):
    waveform = audio or [0.0] * 32000

    class Model:
        def transcribe(self, payload, **options):
            assert payload is waveform
            assert options["word_timestamps"] is True
            return {
                "text": " Local speech.",
                "language": "en",
                "segments": [
                    {
                        "start": 0.1,
                        "end": 1.2,
                        "text": " Local speech.",
                        "words": [
                            {"start": 0.1, "end": 0.5, "word": " Local"},
                            {"start": 0.6, "end": 1.2, "word": " speech."},
                        ],
                    }
                ],
            }

    def load_model(name, device, download_root):
        assert name == "tiny.en"
        assert device == "cpu"
        assert download_root == str(tmp_path / "models")
        return Model()

    fake_whisper = SimpleNamespace(
        _MODELS={"tiny.en": "https://example.invalid/models/tiny.en.pt"},
        load_model=load_model,
        load_audio=lambda path: waveform,
    )
    fake_torch = SimpleNamespace(
        __version__="2.2.0+cpu",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "ai.voiceover_analyzer.FFprobeService",
        lambda: SimpleNamespace(probe=lambda _: {"has_audio": True, "duration": len(waveform) / 16000}),
    )
    monkeypatch.setattr(
        "ai.voiceover_analyzer.shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )
    monkeypatch.setattr(VoiceoverAnalyzer, "_available_memory_bytes", staticmethod(lambda: 8 * 1024**3))
    return waveform


def test_local_transcription_preserves_words_and_word_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = _install_fake_runtime(monkeypatch, tmp_path)
    voiceover = tmp_path / "voice.wav"
    voiceover.write_bytes(b"local-audio-fixture")
    analyzer = VoiceoverAnalyzer("tiny.en", "auto", "auto", tmp_path / "models")
    result = analyzer.analyze(voiceover)
    assert result["engine"] == "local-whisper"
    assert result["text"] == "Local speech."
    assert result["duration"] == len(audio) / 16000
    assert result["segments"][0]["words"][1]["word"] == " speech."


def test_missing_pytorch_has_install_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    voiceover = tmp_path / "voice.wav"
    voiceover.write_bytes(b"fixture")
    monkeypatch.setattr("ai.voiceover_analyzer.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "ai.voiceover_analyzer.FFprobeService",
        lambda: SimpleNamespace(probe=lambda _: {"has_audio": True, "duration": 2}),
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch is missing or binary-incompatible"):
        VoiceoverAnalyzer().analyze(voiceover)


def test_missing_cached_model_can_be_disallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_fake_runtime(monkeypatch, tmp_path)
    voiceover = tmp_path / "voice.wav"
    voiceover.write_bytes(b"fixture")
    analyzer = VoiceoverAnalyzer("tiny.en", "auto", "auto", tmp_path / "missing-models", False)
    with pytest.raises(FileNotFoundError, match="not cached"):
        analyzer.analyze(voiceover)


def test_low_memory_refuses_model_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_whisper = SimpleNamespace(_MODELS={"small": "https://example.invalid/models/small.pt"})
    fake_torch = SimpleNamespace(
        __version__="2.2.0",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "ai.voiceover_analyzer.FFprobeService",
        lambda: SimpleNamespace(probe=lambda _: {"has_audio": True, "duration": 2}),
    )
    monkeypatch.setattr("ai.voiceover_analyzer.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(VoiceoverAnalyzer, "_available_memory_bytes", staticmethod(lambda: 128 * 1024**2))
    voiceover = tmp_path / "voice.wav"
    voiceover.write_bytes(b"fixture")
    with pytest.raises(WhisperResourceError, match="Choose a smaller"):
        VoiceoverAnalyzer("small", download_root=tmp_path / "models").analyze(voiceover)
