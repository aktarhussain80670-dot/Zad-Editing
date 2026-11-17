import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config.settings import Settings
from core.audio_controller import AudioController
from core.checkpoint_manager import CheckpointManager
from core.cleanup_manager import CleanupManager
from providers.http import ProviderError, download_file
from providers.pexels_provider import PexelsProvider
from providers.pixabay_provider import PixabayProvider
from services.audio_mixer import AudioMixer
from services.ffmpeg_service import FFmpegService, MediaToolError
from services.ffprobe_service import FFprobeService
from services.hyperframes_service import HyperFramesService
from services.media_search_service import MediaSearchService


def test_missing_ffmpeg_is_reported():
    with pytest.raises(MediaToolError, match="FFmpeg not found"):
        FFmpegService(executable="missing-ffmpeg-binary").run(["-version"])


def test_ffprobe_parses_stream_metadata(tmp_path: Path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not-read-by-mock")
    response = {
        "format": {"duration": "4.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "avg_frame_rate": "60/1",
                "pix_fmt": "yuv420p",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(response), "")
    with (
        patch("services.ffprobe_service.shutil.which", return_value="ffprobe"),
        patch("services.ffprobe_service.subprocess.run", return_value=completed),
    ):
        info = FFprobeService().probe(media)
    assert info["duration"] == 4.5 and info["fps"] == 60
    assert info["has_audio"] and info["video_codec"] == "h264"


def test_audio_mixer_keeps_voiceover_first_and_skips_disabled_beds(tmp_path: Path):
    voice = tmp_path / "voice.wav"
    music = tmp_path / "music.mp3"
    voice.touch()
    music.touch()
    runner = Mock()
    output = tmp_path / "mix.m4a"
    runner.run.side_effect = lambda args: output.write_bytes(b"encoded")
    settings = Settings(enable_background_music=False, enable_sound_effects=False)
    AudioMixer(runner).mix(voice, output, 3, settings, music, [(music, 1)])
    args = runner.run.call_args.args[0]
    assert args[args.index("-i") + 1] == str(voice)
    assert str(music) not in args


def test_local_music_is_ducked_and_mixed_without_provider_credentials(tmp_path: Path):
    voice = tmp_path / "voice.wav"
    music = tmp_path / "local-music.mp3"
    voice.touch()
    music.touch()
    runner = Mock()
    output = tmp_path / "with-music.m4a"
    runner.run.side_effect = lambda args: output.write_bytes(b"encoded")
    settings = Settings(enable_background_music=True, pixabay_api_key="")
    AudioMixer(runner).mix(voice, output, 4, settings, music)
    args = runner.run.call_args.args[0]
    graph = args[args.index("-filter_complex") + 1]
    assert "sidechaincompress" in graph
    assert "weights='1.0 0.12'" in graph


def test_audio_controller_refuses_voiceover_replacement(tmp_path: Path):
    with pytest.raises(ValueError, match="requires PRESERVE_ORIGINAL_VOICEOVER=true"):
        AudioController().render_mix(
            tmp_path / "voice.wav", tmp_path / "out.wav", 2, Settings(preserve_original_voiceover=False)
        )


def test_hyperframes_composition_is_native_html_and_word_timed():
    guide = {
        "duration": 2,
        "transcript": {"segments": [{"words": [{"start": 0, "end": 0.4, "word": "hello"}]}]},
        "scenes": [
            {
                "scene_id": 1,
                "start": 0,
                "end": 2,
                "visual_idea": "opening",
                "transcript": "hello",
                "caption": {"enabled": True, "text": "hello"},
                "motion": {"enabled": True},
                "transition": {"enabled": True, "type": "smooth_fade"},
            }
        ],
    }
    html = HyperFramesService()._composition_html(guide, {}, 1080, 1920, 60)
    assert 'data-composition-id="main"' in html
    assert 'data-duration="2.0"' in html and "word-1-0" in html
    assert "window.__timelines.main=tl" in html


def test_optional_provider_keys_and_scene_gate():
    settings = Settings(pexels_api_key="", enable_external_clips=True)
    with patch("services.media_search_service.logger.warning") as warn:
        assert MediaSearchService(settings).search_scene({"visual_required": True, "visual_search_query": "test"}) == []
    warn.assert_called_once()
    assert (
        MediaSearchService(Settings(pexels_api_key="set")).search_scene(
            {"visual_required": False, "visual_search_query": "test"}
        )
        == []
    )


def test_pexels_missing_key_and_pixabay_audio_api_limit():
    with pytest.raises(ProviderError, match="PEXELS_API_KEY"):
        PexelsProvider("").search_videos("city")
    with pytest.raises(ProviderError, match="does not provide music"):
        PixabayProvider("configured").search_audio("ambient", "music")


def test_preset_redirect_download_rejects_hosts(tmp_path: Path):
    with pytest.raises(ProviderError, match="permitted provider CDN"):
        download_file("https://example.com/file.mp4", tmp_path / "bad.mp4", 1, 100, {"videos.pexels.com"})


def test_checkpoint_invalidates_modified_inputs(tmp_path: Path):
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"a")
    manager = CheckpointManager(tmp_path / "checkpoint.json")
    fingerprint = manager.fingerprint([input_file], {"model": "test"})
    output = tmp_path / "result.json"
    output.write_text("{}")
    manager.save(fingerprint, "guide", [output])
    assert manager.load(fingerprint) is not None
    input_file.write_bytes(b"changed")
    assert manager.load(manager.fingerprint([input_file], {"model": "test"})) is None


def test_cleanup_restricted_to_temp_and_preserves_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    temp = tmp_path / "temp"
    temp.mkdir()
    scratch = temp / "keep.tmp"
    scratch.touch()
    run = temp / "run-a"
    run.mkdir()
    run_scratch = run / "scratch.tmp"
    run_scratch.touch()
    reference = tmp_path / "reference.mp4"
    reference.touch()
    removed = CleanupManager().cleanup(Settings(delete_temp_files=True), success=True, run_directory=run)
    assert run_scratch in removed and not run.exists() and scratch.exists() and reference.exists()


def test_cleanup_rejects_outside_target(tmp_path: Path):
    with pytest.raises(ValueError, match="restricted"):
        CleanupManager(tmp_path)
