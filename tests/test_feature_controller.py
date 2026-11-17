from config.settings import Settings
from core.feature_controller import enabled_features


def test_disabled_features_remain_disabled():
    result = enabled_features(Settings(enable_subtitles=False, enable_transitions=False, pexels_api_key="set"))
    assert result["subtitles"] is False
    assert result["transitions"] is False
    assert result["external_clips"] is True
